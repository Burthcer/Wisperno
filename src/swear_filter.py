"""
Configurable profanity handling for polished dictation output.
Three modes (see config.py's AppConfig.profanity_filter):
  "allow"  - leave verbatim, the default (a dictation tool should not
             silently alter what the user actually said).
  "censor" - replace all but the first letter with asterisks (e.g. "fuck" -> "f***").
  "remove" - drop the word entirely (surrounding whitespace/punctuation collapsed).

A compiled-once regex over a small, common English profanity list - deliberately
whole-word-boundary matched so it never touches a word that merely contains one
as a substring (e.g. "class", "assess", "Scunthorpe").
"""

import re

# Common English profanity - not exhaustive (a determined user can always defeat
# a client-side word list), just the frequent, unambiguous cases. Word-boundary
# matched below, case-insensitive.
_PROFANITY_WORDS = (
    "fuck", "fucking", "fucked", "fucker",
    "shit", "shitty", "shitting",
    "bitch", "bitching",
    "ass", "asshole",
    "damn", "goddamn",
    "bastard",
    "crap",
    "piss", "pissed",
    "dick", "dickhead",
    "cock",
    "pussy",
    "cunt",
    "whore",
    "slut",
    "hell",
)

_PROFANITY_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _PROFANITY_WORDS) + r")\b",
    re.IGNORECASE,
)


def _censor_word(word: str) -> str:
    """"fuck" -> "f***", "SHIT" -> "S***" - first letter kept (preserves the
    reader's ability to recognize what was said), case of the first letter
    preserved, rest replaced one-for-one with asterisks."""
    if len(word) <= 1:
        return word
    return word[0] + "*" * (len(word) - 1)


def find_profanity_words(text: str) -> set:
    """Lowercase set of known profanity words present in `text` (word-boundary matched)."""
    return {m.group(0).lower() for m in _PROFANITY_RE.finditer(text or "")}


def profanity_was_censored(raw: str, output: str) -> bool:
    """True if `raw` contains a known profanity word that no longer appears
    anywhere in `output` (verbatim, case-insensitive) - catches the base
    model's own RLHF safety alignment silently dropping, euphemizing, or
    asterisking a swear word despite an explicit 'allow' (never censor)
    instruction in the prompt. Prompt instructions alone are not reliable
    here (verified directly: the same polish prompt's rule against censoring
    still lost words like "fucking" on some inputs) - this is the
    deterministic backstop, checked only when profanity_filter == "allow"
    (see transformer.py's sanity_check)."""
    raw_words = find_profanity_words(raw)
    if not raw_words:
        return False
    output_words = find_profanity_words(output)
    return not raw_words.issubset(output_words)


def apply_profanity_filter(text: str, mode: str) -> str:
    """Apply the configured profanity mode to already-polished text. `mode` is
    one of "allow" (no-op), "censor", "remove". An unrecognized mode is
    treated as "allow" - never silently censor without an explicit setting."""
    if not text or mode == "allow" or mode not in ("censor", "remove"):
        return text

    if mode == "censor":
        return _PROFANITY_RE.sub(lambda m: _censor_word(m.group(0)), text)

    # mode == "remove": drop the word; collapse the resulting double space
    # (and a lone leading/trailing space) so removal doesn't leave "He is  a"
    # or a dangling space before punctuation like "He is  ."
    removed = _PROFANITY_RE.sub("", text)
    removed = re.sub(r" {2,}", " ", removed)
    removed = re.sub(r" +([.,!?;:])", r"\1", removed)
    return removed.strip()
