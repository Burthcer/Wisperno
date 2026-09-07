"""
Contextual Polish & Transformation Engine for Wisperno.
Wraps llama-cpp-python for local SLM inference (Llama 3.2 / Qwen 2.5) on NVIDIA GPU.
"""

import os
import re
import sys
import time
from typing import Dict, Optional
from loguru import logger

# Ensure CUDA runtime DLLs are found on Windows
if sys.platform == "win32":
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(torch_lib)
            os.environ["PATH"] = torch_lib + os.path.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

from typing import List

from llama_cpp import Llama
from src.config import LLMConfig, get_base_dir
from src.direct_formatter import capitalize_pronoun_i
from src.swear_filter import apply_profanity_filter, profanity_was_censored

# Above this word count, split into sentence-aligned chunks before transforming.
# n_ctx=4096 tokens is comfortable for one utterance, but multi-minute unbounded
# dictation plus a system prompt plus generation budget can approach that
# ceiling - chunking keeps each call well inside it regardless of session length.
LONG_FORM_WORD_THRESHOLD = 400
CHUNK_WORD_TARGET = 400

# Guardrail: prefixes that mean the model drifted into conversational-assistant
# mode (answering/refusing) instead of just cleaning the dictation. Caught here
# rather than trusting the prompt alone, since temperature=0.0 + few-shot
# constraints reduce but don't guarantee this can't happen.
DRIFT_PREFIXES = (
    "i'm sorry", "i am sorry", "i cannot", "i can't", "the system is",
    "as an ai", "i'm an ai", "i am an ai", "here is", "here's", "here are",
)

# Phrases that mean the model spoke TO the user (asked a question back, offered
# help, editorialized) rather than just cleaning what the user said - checked
# anywhere in the output, not just as a prefix, since these can appear mid-
# response too. Skipped if the phrase is already present in the raw input
# (the user may have genuinely dictated it as content).
CONVERSATIONAL_TRIGGERS = (
    "what is your goal", "how can i help", "sure, here", "as an ai",
    "i understand", "could you clarify", "here is the polished", "here's the",
)

# Padded-with-spaces substring markers (see the " {text} " padding at each use
# site) for the voice-shift check below - space-bounded so "you" doesn't match
# inside an unrelated word, and "i'm"/"i am" catch the contraction and the
# expanded form (see _normalize_contractions, not reused here since these are
# whole-word markers, not phrase-prefix ones).
FIRST_PERSON_MARKERS = (" i ", "i'm ", "i am ", " my ", " me ", " me.", " me?", " me,")
SECOND_PERSON_MARKERS = ("you're ", "you are ", " you ", " you.", " you?", " you!", " your ", "yourself")

# Markers of a legitimate mid-sentence self-correction (see the polish
# prompt's own rule 6) - a raw input containing one of these can legitimately
# change shape (question -> statement or vice versa) as part of resolving the
# correction, so the question-answered check below skips it rather than
# risk a false positive on that deliberate rewrite.
SELF_CORRECTION_MARKERS = ("wait no", "wait, no", "actually", "i mean")

# A genuinely pathological LLM failure mode (looping instead of stopping),
# distinct from and much rarer than "rephrased more than expected" - catches
# what GUARDED_WORDS/length-ratio/vocabulary-similarity were removed for
# below: this many *identical* sentences repeated back to back is never
# legitimate polish output, whereas "did the wording change" often is.
REPETITION_LOOP_THRESHOLD = 6

# Catastrophic-only over-generation ceiling: a full multi-paragraph answer to
# a short question (e.g. "can you explain python pointers to me" -> a 230-word
# technical essay - a real case found testing Turbo, caught by none of the
# other checks above since the raw input already contains both "you" and "me")
# is not a paraphrase-style disagreement the removed length-ratio check used
# to (over-eagerly) flag - it's the model answering instead of cleaning, the
# original problem this whole guardrail exists for. 4x is deliberately loose:
# every legitimate polish sample measured across this session (punctuation
# restoration, filler removal, even aggressive stutter-stripping) stays under
# ~1.5x; this only fires on genuine essay-length over-generation.
ESSAY_CEILING_RATIO = 4.0

# Catches a genuine, demonstrated defect the essay ceiling above MISSES: a
# short, direct answer to a question instead of a fabricated essay (e.g.
# Turbo turning "what is your goal for this quarter" into "Your goal ... is
# to achieve [specific target]." - only 1.3x longer than the input, well
# under the essay ceiling, yet a clear answer with a hallucinated filled-in
# placeholder). Rule 2 of the polish prompt itself mandates every dictated
# question survive as a cleaned question - this is that same invariant,
# checked in Python. "Contains '?' anywhere" (not "ends with") is deliberate:
# a correct multi-clause output like "What is the capital of Japan? I
# forgot." legitimately ends on a non-question clause. Skipped on a
# self-correction, which can legitimately change a sentence's shape.
QUESTION_STARTERS = (
    "what ", "who ", "when ", "where ", "why ", "how ",
    "is ", "are ", "do ", "does ", "did ", "can ", "could ", "would ", "will ", "should ",
    "have ", "has ",
)

_CONTRACTION_EXPANSIONS = (
    ("i'm", "i am"), ("here's", "here is"), ("it's", "it is"), ("that's", "that is"),
    ("there's", "there is"), ("can't", "cannot"), ("won't", "will not"),
    ("isn't", "is not"), ("don't", "do not"),
)


def _normalize_contractions(text: str) -> str:
    """The polish prompt's own grammar-fixing rule legitimately expands contractions
    ("here's" -> "here is") - without this, the raw-input exemption below can't tell
    that expanded output apart from genuine drift, since the exact contracted string
    the user said no longer appears in the model's grammar-corrected output."""
    for contracted, expanded in _CONTRACTION_EXPANSIONS:
        text = text.replace(contracted, expanded)
    return text


def _has_repetition_loop(text: str, threshold: int = REPETITION_LOOP_THRESHOLD) -> bool:
    """True if the same sentence appears `threshold` or more times - a real,
    pathological LLM failure (looping instead of stopping) that greedy
    decoding can still occasionally hit, unrelated to whether the wording
    changed from the input (which legitimate polishing does on purpose)."""
    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) < threshold:
        return False
    counts: Dict[str, int] = {}
    for s in sentences:
        counts[s] = counts.get(s, 0) + 1
        if counts[s] >= threshold:
            return True
    return False


# Modes whose prompts explicitly mandate near-verbatim, non-summarizing output
# (see config.yaml's "MANDATORY FIDELITY RULES" for polish). bullets/
# prompt_engineer/custom Transforms-hub prompts are DESIGNED to restructure,
# condense, and drop connective words when converting prose into a list or a
# structured spec - applying a verbatim-fidelity check there is a guaranteed
# false positive on completely correct output, not a safety net.
FIDELITY_STRICT_MODES = frozenset({"polish", "code"})


def sanity_check(raw: str, output: str, mode: str = "polish", profanity_filter: str = "allow") -> Optional[str]:
    """Programmatic guardrail, independent of prompt-following: returns a
    human-readable reason `output` looks wrong for `raw`, or None if it's
    fine. Not a replacement for the prompt/DRIFT_PREFIXES check above - a
    second, cheaper line of defense that doesn't rely on the model obeying
    instructions.

    Deliberately narrow, by design and by evidence: an earlier, stricter
    version of this function also flagged word-count deviation and dropped
    "guarded words" (on/or/and/...), reasoning that legitimate polish output
    tracks the input closely. In real use this was wrong often enough to be
    the dominant source of fallbacks - both checks fired on completely
    correct output ("going to work" -> "will work" tripped the guarded-word
    check; "so wait what was I saying oh right um the deadline moved to
    friday" -> "The deadline moved to Friday." - a clean, correct filler
    removal - tripped the length check). Removed rather than re-tuned again:
    two rounds of narrowing thresholds still left real false positives,
    because there is no clean surface-level signal that distinguishes good
    paraphrase from bad paraphrase - only whether the MEANING changed, which
    word-count/vocabulary-overlap can't measure. What's left below targets
    failure modes that ARE reliably detectable this way: the model speaking
    TO the user, looping, or (for the smaller Turbo model specifically,
    see FIRST_PERSON_MARKERS below) silently changing who's being addressed.
    """
    if not output.strip():
        return None  # empty output already falls back via clean_output/DRIFT_PREFIXES's caller

    if _has_repetition_loop(output):
        return f"repetition loop ({REPETITION_LOOP_THRESHOLD}+ identical sentences)"

    lowered_out = output.lower()
    lowered_raw = raw.lower()
    normalized_raw = _normalize_contractions(lowered_raw)
    for phrase in CONVERSATIONAL_TRIGGERS:
        if phrase in lowered_out and phrase not in lowered_raw and phrase not in normalized_raw:
            return f"conversational leak (contains '{phrase}')"

    # Profanity-integrity check: only meaningful when the user's explicit
    # setting is "allow" (verbatim) - "censor"/"remove" already deterministically
    # post-process the final text regardless of what the model did, so a
    # dropped/euphemized swear word there isn't a fidelity violation. Verified
    # directly this round: despite the polish prompt's own "never censor" rule,
    # the 3B Standard model still silently dropped or substituted a real,
    # user-spoken swear word on multiple ordinary inputs ("i cannot get this
    # fucking thing to work" -> "...this thing to work.", "what the shit is
    # going on" -> "...the heck is going on") - ordinary RLHF alignment
    # overriding an explicit instruction, not a rare adversarial case. Checked
    # mode-agnostically (like the leak check above), since the user's
    # profanity preference applies regardless of which mode/prompt is active.
    if profanity_filter == "allow" and profanity_was_censored(raw, output):
        return "profanity censored by the model despite 'Allow All' setting"

    if mode not in FIDELITY_STRICT_MODES:
        return None  # bullets/prompt_engineer/custom prompts restructure by design - leak check above still applies

    # Voice-shift check: catches a smaller model silently rewriting first-person
    # dictation into second-person address ("I am trying to..." -> "You're
    # trying to..."). Kept as the one exception to this function's otherwise
    # word-count/vocabulary-agnostic design: this is a real, demonstrated,
    # reproducible defect specific to the smaller Turbo Flagship model (not a
    # paraphrase-style disagreement), and it changes WHO the sentence is
    # about, not just how it's worded - the exact distinction the other,
    # removed checks couldn't reliably draw.
    padded_raw, padded_out = f" {lowered_raw} ", f" {lowered_out} "
    if any(m in padded_raw for m in FIRST_PERSON_MARKERS) and not any(m in padded_raw for m in SECOND_PERSON_MARKERS):
        if any(m in padded_out for m in SECOND_PERSON_MARKERS):
            return "voice shift (first-person input rewritten to second-person output)"

    # Catastrophic-only essay ceiling - see ESSAY_CEILING_RATIO's comment.
    # Loose enough that no legitimate polish output should ever trip it.
    raw_words = raw.split()
    if raw_words and len(output.split()) > len(raw_words) * ESSAY_CEILING_RATIO:
        return f"essay-length over-generation ({len(output.split())} words from a {len(raw_words)}-word input)"

    # Answered-instead-of-cleaned check - see QUESTION_STARTERS's comment.
    if lowered_raw.startswith(QUESTION_STARTERS) and not any(m in lowered_raw for m in SELF_CORRECTION_MARKERS):
        if "?" not in output:
            return "question was answered instead of cleaned (no '?' anywhere in output)"

    return None


def basic_capitalize(text: str) -> str:
    """Fallback formatting when the LLM output fails sanity_check: capitalize
    sentence starts on the raw transcript and ensure terminal punctuation,
    instead of injecting nothing or trusting a possibly-hallucinated LLM
    output. Reuses QUESTION_STARTERS (the same heuristic sanity_check's own
    "answered instead of cleaned" check already uses) to end on "?" rather
    than "." when the raw input is clearly a dictated question - without
    this, a guardrail trip on a question-shaped utterance (more likely on a
    smaller/faster model that needs this fallback more often, e.g. Turbo
    Flagship) silently dropped the question mark Whisper itself didn't add,
    even though the whole point of this fallback is to still read as what
    the user actually said, not a flattened statement.
    """
    text = text.strip()
    if not text:
        return text
    sentences = re.split(r"([.!?]\s+)", text)
    capitalized = "".join(
        (s[0].upper() + s[1:] if s and s[0].isalpha() else s) for s in sentences
    )
    capitalized = capitalized[0].upper() + capitalized[1:] if capitalized else capitalized
    if capitalized and capitalized[-1] not in ".!?":
        capitalized += "?" if text.lower().startswith(QUESTION_STARTERS) else "."
    return capitalize_pronoun_i(capitalized)


class Transformer:
    """
    Contextual text transformation engine using local GGUF models on CUDA.
    """

    def __init__(
        self, config: Optional[LLMConfig] = None, prompts: Optional[Dict[str, str]] = None,
        profanity_filter: str = "allow",
    ):
        self.config = config or LLMConfig()
        self.prompts = prompts or {}
        self.active_model_name: Optional[str] = None
        self.active_model_layers: Optional[int] = None
        self.last_fallback_applied = False  # set by _transform_chunk; read by workers.py to warn the user
        # Mutable, not baked into a closure: Settings can flip this live via
        # set_profanity_filter() with no engine/model restart needed - it only
        # affects a post-processing step on the final text, not inference.
        self.profanity_filter = profanity_filter
        self.llm = self._load_llm()

    def set_profanity_filter(self, mode: str) -> None:
        self.profanity_filter = mode

    @staticmethod
    def _total_layers(llm: Llama) -> Optional[int]:
        """Read the model's real decoder layer count from its own GGUF metadata (key varies by
        architecture, e.g. 'llama.block_count' vs 'qwen2.block_count') rather than hardcoding one
        model's layer count, which would silently mislabel the GPU/CPU split for any other model."""
        try:
            for key, value in llm.metadata.items():
                if key.endswith(".block_count"):
                    return int(value)
        except Exception:
            pass
        return None

    def _resolve_path(self, model_path: str) -> str:
        """Resolve a possibly-relative model path against the project root."""
        if os.path.isabs(model_path):
            return model_path
        return str(get_base_dir() / model_path)

    def _load_gguf(self, model_path: str) -> Optional[Llama]:
        """Load a single GGUF file into VRAM, returning None on any failure."""
        logger.info(
            f"Loading LLM '{os.path.basename(model_path)}' (n_gpu_layers={self.config.n_gpu_layers}, "
            f"n_ctx={self.config.n_ctx})..."
        )
        t0 = time.perf_counter()
        # Flash attention always on: cuts attention compute/memory overhead
        # regardless of KV cache mode. Q8_0 KV cache quantization (type_k/type_v=8,
        # ~600MB VRAM saved) additionally requires it - verified locally, not a
        # supported combination without flash_attn in this llama-cpp-python build.
        kv_kwargs = {"type_k": 8, "type_v": 8} if self.config.kv_cache_quantization else {}
        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=self.config.n_gpu_layers,
                n_threads=self.config.n_threads,
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                flash_attn=True,
                verbose=False,
                **kv_kwargs,
            )
            load_time = (time.perf_counter() - t0) * 1000
            logger.info(f"LLM loaded into GPU memory in {load_time:.1f}ms.")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize llama-cpp LLM '{model_path}': {e}")
            return None

    def _load_llm(self) -> Optional[Llama]:
        """
        Load the primary GGUF model into VRAM. If it isn't on disk yet, fall back
        to the lightweight secondary model so the app remains usable while the
        primary model finishes downloading.
        """
        primary_path = self._resolve_path(self.config.model_path)
        if os.path.exists(primary_path):
            llm = self._load_gguf(primary_path)
            if llm is not None:
                self.active_model_name = os.path.basename(primary_path)
                self.active_model_layers = self._total_layers(llm)
                return llm
        else:
            logger.warning(f"Primary LLM model file not found at '{primary_path}'.")

        fallback_path = self._resolve_path(self.config.fallback_model_path)
        if os.path.exists(fallback_path):
            logger.warning(f"Falling back to lightweight LLM '{os.path.basename(fallback_path)}'.")
            llm = self._load_gguf(fallback_path)
            if llm is not None:
                self.active_model_name = os.path.basename(fallback_path)
                self.active_model_layers = self._total_layers(llm)
                return llm

        logger.warning(
            "No LLM model weights available (primary or fallback). "
            "Transformation will fall back to raw transcript until a model is downloaded."
        )
        return None

    def clean_output(self, text: str) -> str:
        """
        Remove markdown code fences, intro conversational fillers, and extraneous quotes.
        """
        if not text:
            return ""

        cleaned = text.strip()

        # Remove markdown code fences ```markdown ... ``` or ``` ... ```
        fence_match = re.match(r"^```(?:[a-zA-Z0-9_\-]+)?\s*\n?(.*?)\n?```$", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # Remove common introductory filler patterns. The "Here(...)...:" pattern is
        # intentionally general (any wording up to the first colon) rather than an
        # exact phrase list - a narrower list previously missed "Here's *a* polished
        # version of your dictation:" (only "Here's *the* ..." was covered), which
        # then fell through to the DRIFT_PREFIXES guardrail below as a false-positive
        # "answered instead of cleaned" case even though the actual cleaning was correct.
        filler_patterns = [
            r"^Here(?:'s|\s+is|\s+are)\b[^\n:]{0,80}:\s*",
            r"^(?:Sure(?: thing)?,? here (?:is|are)[^:\n]*:?\s*)",
            r"^(?:Polished text:?\s*)",
        ]
        for pattern in filler_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # Remove trailing notes or explanations (e.g., "\n\nNote: I removed filler words...")
        cleaned = re.sub(r"\n+(?:Note|Notes|Explanation|Summary|Remarks):.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

        # Remove surrounding quotes if model enclosed entire output in quotes
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            if len(cleaned) >= 2:
                cleaned = cleaned[1:-1].strip()

        # Deterministic backstop for the prompt's own "I" capitalization rule -
        # the model usually gets this right, but this makes it never wrong,
        # regardless of prompt-following.
        return capitalize_pronoun_i(cleaned)

    @staticmethod
    def _split_into_chunks(text: str, max_words: int = CHUNK_WORD_TARGET) -> List[str]:
        """Split text into chunks of up to `max_words`, breaking only at sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: List[str] = []
        current: List[str] = []
        current_words = 0
        for sentence in sentences:
            words_in_sentence = len(sentence.split())
            if current and current_words + words_in_sentence > max_words:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            current.append(sentence)
            current_words += words_in_sentence
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _transform_chunk(self, text: str, mode: str, system_prompt: str) -> str:
        """Run a single LLM call on one chunk (already known to fit the context window)."""
        t0 = time.perf_counter()
        try:
            # Dynamic ceiling scaled to input length, not a flat 4096: without a
            # tight cap and explicit stop tokens, a chunk that doesn't hit a
            # natural stop can run generation out toward the full ceiling -
            # measured directly (a 211-word chunk took ~14s vs <1s for
            # comparable short chunks) before this fix. Floor raised to 512 (was
            # 128) and cap to 2048 (was 1024) - the old tighter numbers were safe
            # for the ~400-word chunk size this app actually sends, but the real
            # backstop against runaway generation is the EOS-only stop list
            # below, not a cramped token ceiling that risked truncating a
            # legitimately long, punctuation-heavy polish of a longer chunk.
            dynamic_max_tokens = min(self.config.max_tokens, 2048, max(512, int(len(text.split()) * 1.6)))
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text.strip()},
                ],
                temperature=self.config.temperature,
                # top_p is a documented no-op here, not dead config: llama-cpp-
                # python's sampler special-cases temperature==0 into true greedy
                # argmax (verified in _internals.py's _LlamaSamplingContext.sample),
                # bypassing top_p/top_k entirely. Deliberately NOT setting
                # repeat_penalty above its neutral 1.0 default, despite that
                # looking like it would help fidelity: the SAME source shows
                # repeat_penalty is applied to logits BEFORE the greedy argmax
                # pick, even at temp=0 - it would actively suppress any word
                # (e.g. "on") that already appeared recently, which is the
                # opposite of what verbatim dictation cleanup needs.
                top_p=1.0,
                max_tokens=dynamic_max_tokens,
                # "User:"/"Raw:" removed as stop strings: both are plausible literal
                # dictated content (discussing CLI/log/prompt text, this app's own
                # target audience), so they could truncate a legitimate polish
                # mid-sentence. Kept "<|im_end|>"/"<|endoftext|>" (the model's real
                # chat-template EOS markers) and "\n\n\n" (three consecutive
                # newlines essentially never occurs in real transcribed speech -
                # Whisper's own output is a single continuous line - so this only
                # catches the model inventing a fake follow-up turn, not real content).
                stop=["<|im_end|>", "<|endoftext|>", "\n\n\n"],
            )
            content = response["choices"][0]["message"]["content"]
            result = self.clean_output(content)
            lowered_result = result.lower()
            lowered_raw = text.lower()
            normalized_raw = _normalize_contractions(lowered_raw)
            # Root cause of the "repeatedly falls back to raw text" defect: this
            # check used to fire on ANY output starting with e.g. "I'm sorry" or
            # "Here is" - including completely legitimate dictation that itself
            # starts that way ("I'm sorry I'm late", "Here is what I found"),
            # which is extremely common real speech, not a rare edge case
            # (reproduced on 6/6 everyday test sentences before this fix). Now
            # exempted exactly like CONVERSATIONAL_TRIGGERS already was: only a
            # drift phrase absent from the user's own raw input counts as drift.
            # normalized_raw additionally expands contractions ("here's" -> "here
            # is") so a legitimate grammar-fix doesn't defeat the exemption.
            drift_prefix = next(
                (p for p in DRIFT_PREFIXES
                 if lowered_result.startswith(p) and p not in lowered_raw and p not in normalized_raw),
                None,
            )
            if drift_prefix:
                logger.warning(
                    f"[FALLBACK_DEBUG] Reason: conversational drift prefix '{drift_prefix}' | "
                    f"Raw: {text!r} | Output: {result!r}"
                )
                result = basic_capitalize(text.strip())
                self.last_fallback_applied = True
            else:
                reason = sanity_check(text, result, mode=mode, profanity_filter=self.profanity_filter)
                if reason:
                    logger.warning(f"[FALLBACK_DEBUG] Reason: {reason} | Raw: {text!r} | Output: {result!r}")
                    result = basic_capitalize(text.strip())
                    self.last_fallback_applied = True
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"Transformed [{mode}] ({len(text)} -> {len(result)} chars) in {elapsed_ms:.1f}ms.")
            if not result.strip() or result.strip() == text.strip():
                # Not necessarily a bug on its own (a guardrail fallback or an
                # already-clean input can legitimately produce this), but it's
                # exactly what "polish did nothing" looks like from the user's
                # side - worth a distinct, greppable line rather than burying
                # it in the [FALLBACK_DEBUG] warning above (which only fires
                # on an actual guardrail trip, not on a plain no-op).
                logger.warning(f"[POLISH] LLM output identical to (or empty vs.) raw input for mode '{mode}'.")
            return result
        except Exception as e:
            logger.error(f"Error during LLM transformation: {e}. Falling back to raw text for this chunk.")
            self.last_fallback_applied = True
            return basic_capitalize(text.strip())

    def transform(self, raw_text: str, mode: str = "polish") -> str:
        """
        Transform raw transcript text according to a config.yaml-defined mode
        ("polish", "prompt_engineer", "bullets", "code", "raw").
        """
        self.last_fallback_applied = False
        if not raw_text or not raw_text.strip():
            return ""
        if mode == "raw" or self.llm is None:
            return apply_profanity_filter(raw_text.strip(), self.profanity_filter)

        system_prompt = self.prompts.get(mode, "")
        if not system_prompt:
            logger.warning(f"No prompt template found for mode '{mode}'. Returning raw text.")
            return apply_profanity_filter(raw_text.strip(), self.profanity_filter)

        separator = "\n" if mode == "bullets" else " "
        return apply_profanity_filter(self._run(raw_text, mode, system_prompt, separator), self.profanity_filter)

    def transform_with_prompt(self, raw_text: str, system_prompt: str, label: str = "custom") -> str:
        """
        Transform raw text with an explicit system prompt (a Transforms-hub
        entry's own `system_prompt`, from the database) rather than looking
        one up by config.yaml mode name - same chunking/fidelity behavior.
        """
        self.last_fallback_applied = False
        if not raw_text or not raw_text.strip():
            return ""
        if self.llm is None or not system_prompt:
            return apply_profanity_filter(raw_text.strip(), self.profanity_filter)
        return apply_profanity_filter(self._run(raw_text, label, system_prompt, " "), self.profanity_filter)

    def _run(self, raw_text: str, label: str, system_prompt: str, separator: str) -> str:
        self.last_fallback_applied = False
        word_count = len(raw_text.split())
        if word_count <= LONG_FORM_WORD_THRESHOLD:
            return self._transform_chunk(raw_text, label, system_prompt)

        chunks = self._split_into_chunks(raw_text)
        logger.info(
            f"Long-form dictation ({word_count} words) - splitting into {len(chunks)} chunk(s) "
            f"of ~{CHUNK_WORD_TARGET} words to stay well within the context window."
        )
        results = [self._transform_chunk(chunk, label, system_prompt) for chunk in chunks if chunk.strip()]
        return separator.join(r for r in results if r)
