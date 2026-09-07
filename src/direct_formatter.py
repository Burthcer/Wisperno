"""
Instant, zero-LLM formatter for Wisperno's default dictation hotkey (Mode 1).

Pure regex/string cleanup - no llama.cpp call, no model, <5ms. Dictionary/
jargon replacement is NOT duplicated here: Vocabulary.apply_replacements()
already runs inside Transcriber.transcribe() before this ever sees the text
(see src/transcriber.py), and Whisper's own punctuation is left as-is except
for the capitalization/terminal-punctuation pass at the end.
"""

import re

# "like" is only a filler when it trails off as a verbal pause (Whisper tends
# to punctuate that with a comma) - stripping every "like" would also eat the
# verb ("I like pizza"), so it's matched separately and only with a comma.
_FILLER_RE = re.compile(r"\b(?:uh|um|ah|er|you know)\b,?\s*", re.IGNORECASE)
_FILLER_LIKE_RE = re.compile(r"\blike,\s*", re.IGNORECASE)

# Stutter: a word immediately repeating itself one or more times ("the the",
# "I I I"), or a sentence-final abbreviation repeated by an STT quirk ("etc.
# etc."). The optional trailing "\.?" is deliberately period-only, not comma -
# a comma before the repeat ("great, great job") is intentional spoken
# emphasis, not a stutter, and must survive untouched.
_STUTTER_RE = re.compile(r"\b(\w+)(\.?)(?:\s+\1\b\.?)+", re.IGNORECASE)

# Spoken developer/terminal shorthand -> the literal symbol.
_SLASH_COMMAND_RE = re.compile(r"\b(?:slash|forward slash)\s*([a-zA-Z0-9_\-]+)\b", re.IGNORECASE)
# Lookaround-guarded (word on both sides) so an isolated "dash" used as a
# spoken bullet-list cue (see _BULLET_CUE below) isn't eaten by this pass.
_DASH_WORD_RE = re.compile(r"(?<=\w)\s+(?:dash|hyphen)\s+(?=\w)", re.IGNORECASE)
_UNDERSCORE_WORD_RE = re.compile(r"(?<=\w)\s+underscore\s+(?=\w)", re.IGNORECASE)
_BACKTICK_RE = re.compile(r"\bbacktick\b", re.IGNORECASE)
# A bare command ("/compact") shouldn't get a terminal "." appended by the
# normal sentence-punctuation pass.
_BARE_COMMAND_RE = re.compile(r"^/[\w\-]+$")

_ORDINAL_CUE = (
    r"firstly|first|secondly|second|thirdly|third|fourthly|fourth|fifthly|fifth|"
    r"number\s+(?:one|two|three|four|five|six|seven|eight|nine|ten)"
)
_NUMBERED_SPLIT_RE = re.compile(rf"\b({_ORDINAL_CUE})\b", re.IGNORECASE)

_BULLET_CUE = r"bullet point|point one|next point|dash"
_BULLET_SPLIT_RE = re.compile(rf"\b({_BULLET_CUE})\b", re.IGNORECASE)

# Below this many cues, treat it as an incidental word in normal speech
# ("I first went to..."), not an actual spoken enumeration.
_MIN_LIST_CUES = 2

# Whisper frequently leaves the first-person pronoun lowercase mid-utterance
# ("i think", "i'm not sure"). \b matches on BOTH sides of a bare "i" even
# right before an apostrophe (the apostrophe is a non-word char), so this one
# pattern already covers "i"/"i'm"/"i've"/"i'll"/"i'd" - no separate
# contraction regex needed. Case-sensitive on purpose: an already-correct "I"
# is left untouched rather than needlessly rewritten.
LOWERCASE_I_RE = re.compile(r"\bi\b")


def capitalize_pronoun_i(text: str) -> str:
    return LOWERCASE_I_RE.sub("I", text)


def _apply_spoken_commands(text: str) -> str:
    """Spoken developer/terminal shorthand -> literal symbols ("slash compact"
    -> "/compact", "my dash file" -> "my-file", "backtick" -> "`")."""
    text = _SLASH_COMMAND_RE.sub(r"/\1", text)
    text = _DASH_WORD_RE.sub("-", text)
    text = _UNDERSCORE_WORD_RE.sub("_", text)
    text = _BACKTICK_RE.sub("`", text)
    return text


def _strip_fillers(text: str) -> str:
    text = _FILLER_RE.sub("", text)
    text = _FILLER_LIKE_RE.sub("", text)
    text = _STUTTER_RE.sub(r"\1\2", text)
    text = re.sub(r"[ \t]+([,.!?])", r"\1", text)
    # Collapse runs of plain spaces/tabs only - a "\n\n" paragraph break
    # (src/transcriber.py's pause-based split) must survive this pass, so
    # newlines are deliberately excluded from what counts as collapsible
    # whitespace here.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]*\n\n[ \t]*", "\n\n", text)
    return re.sub(r"^[,\s]+", "", text).strip()


def _cap_punct(item: str) -> str:
    item = item.strip(" ,")
    if not item:
        return item
    item = item[0].upper() + item[1:]
    if item[-1] not in ".!?":
        item += "."
    return item


def _cap_sentences(text: str) -> str:
    """Capitalize the first letter of the text and of every sentence following
    a '.'/'!'/'?' - independent of src.transformer so this module never pulls
    in the (heavy, CUDA-touching) llama.cpp import chain."""
    sentences = re.split(r"([.!?]\s+)", text)
    capitalized = "".join(
        (s[0].upper() + s[1:] if s and s[0].isalpha() else s) for s in sentences
    )
    return capitalized[0].upper() + capitalized[1:] if capitalized else capitalized


def _split_list(text: str, split_re: re.Pattern) -> list:
    """Split on a cue pattern; return the item segments (text after each cue),
    dropping the pre-first-cue lead-in, or [] if fewer than _MIN_LIST_CUES cues found."""
    parts = split_re.split(text)
    cues = parts[1::2]
    if len(cues) < _MIN_LIST_CUES:
        return []
    items = parts[2::2]
    return [i.strip(" ,") for i in items if i.strip(" ,")]


def format_direct(text: str) -> str:
    """Zero-LLM dictation cleanup: fillers/stutters out, spoken lists formatted,
    capitalization/terminal punctuation applied. Runs in well under 10ms."""
    text = text.strip()
    if not text:
        return ""

    text = _strip_fillers(text)
    if not text:
        return ""
    text = capitalize_pronoun_i(text)
    text = _apply_spoken_commands(text)

    numbered_items = _split_list(text, _NUMBERED_SPLIT_RE)
    if numbered_items:
        return "\n".join(f"{i + 1}. {_cap_punct(item)}" for i, item in enumerate(numbered_items))

    bullet_items = _split_list(text, _BULLET_SPLIT_RE)
    if bullet_items:
        return "\n".join(f"* {_cap_punct(item)}" for item in bullet_items)

    if _BARE_COMMAND_RE.match(text.strip()):
        return text.strip()

    # Paragraph breaks (src/transcriber.py's pause-based split) are
    # capitalized/punctuated independently, same as the very first sentence
    # of the whole utterance - each one starts a fresh thought.
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    return "\n\n".join(_cap_punct(_cap_sentences(p)) for p in paragraphs)


def _demo() -> None:
    import time

    r1 = format_direct("um I think we should uh go with option A")
    assert r1 == "I think we should go with option A.", r1

    r2 = format_direct("we need three things first buy milk second get eggs third go home")
    lines = r2.split("\n")
    assert lines == ["1. Buy milk.", "2. Get eggs.", "3. Go home."], r2

    r3 = format_direct("the the client wants changes")
    assert r3 == "The client wants changes.", r3

    r4 = format_direct("bullet point buy milk bullet point get eggs")
    assert r4.split("\n") == ["* Buy milk.", "* Get eggs."], r4

    r5 = format_direct("I first went to the store")
    assert "\n" not in r5 and r5 == "I first went to the store.", r5

    r6 = format_direct("i think i'm going to head out, i've already told them i'll be late")
    assert r6 == "I think I'm going to head out, I've already told them I'll be late.", r6

    r7 = format_direct("slash compact")
    assert r7 == "/compact", r7

    r8 = format_direct("run slash help to see options")
    assert r8 == "Run /help to see options.", r8

    r9 = format_direct("we're all done here etc. etc.")
    assert "etc. etc." not in r9 and r9.rstrip(".").endswith("etc"), r9

    r10 = format_direct("great, great job team")
    assert r10 == "Great, great job team.", r10  # comma-separated emphasis preserved

    r11 = format_direct("my project dash file underscore name")
    assert r11 == "My project-file_name.", r11

    r12 = format_direct("this is the opening thought\n\nhere is another one after a pause")
    assert r12 == "This is the opening thought.\n\nHere is another one after a pause.", r12

    r13 = format_direct("um so uh this is one thought\n\n  the the other thought here")
    assert r13 == "So this is one thought.\n\nThe other thought here.", r13

    t0 = time.perf_counter()
    for _ in range(100):
        format_direct("um so uh I was thinking we could uh go with the the second option")
    elapsed_ms = (time.perf_counter() - t0) * 1000 / 100
    assert elapsed_ms < 10, f"format_direct averaged {elapsed_ms:.3f}ms/call, expected <10ms"

    print(f"PASS: filler/stutter stripping, numbered + bullet list detection, capitalization all verified "
          f"({elapsed_ms:.3f}ms/call avg).")


if __name__ == "__main__":
    _demo()
