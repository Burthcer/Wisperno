"""
"AI Relay" transform (Alt+R) verification for Wisperno.

Real LLM, real DB migration path (seed_ai_relay_transform() - the one-time
insert that reaches an EXISTING install's transforms table, since
DEFAULT_TRANSFORMS itself is only ever applied to a brand-new database - same
pattern as test_grammar_correct.py).

The whole point of this transform is that it does LESS than Polish (no
sentence-splitting/restructuring, since the output goes to another AI, not a
human reader) while still doing filler removal, self-correction resolution,
capitalization, and drift-guarding exactly like Polish does. Every test here
either verifies "does the same thing as Polish" or "does deliberately less
than Polish" - never "does something new".

Run: python tests/test_ai_relay.py
"""

import os
import sys
import tempfile
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# A deliberately long, rambling, multi-topic, filler-heavy, self-correcting
# sample - the exact shape of input this transform exists for (a long spoken
# ramble headed to another AI chatbot, not a short one-liner).
RAMBLE = (
    "um so basically what I am trying to say is that I think we should uh go with the "
    "the second approach because it seems more scalable and you know it also fits "
    "better with what the team already knows and uh actually wait no I mean the first "
    "approach fits better with what the team knows and also I wanted to mention that "
    "the deployment last week broke the staging environment and nobody noticed for like "
    "six hours which is kind of a big deal and we should probably set up better alerting "
    "for that"
)

SELF_CORRECTION_SAMPLE = "call him at 3 wait no 4 pm"
PROFANITY_SAMPLE = "this fucking build keeps failing and I don't know why"

CONVERSATIONAL_TRAPS = [
    "this is not working",
    "i don't know why this is failing",
    "can you explain python pointers to me",
    "why is this not working",
]
CONVERSATIONAL_LEAKS = (
    "i can help", "i'd be happy", "i would be happy", "here is an explanation",
    "here's an explanation", "let me explain", "as an ai", "i'm sorry", "i am sorry",
    "sure, here", "sure thing", "great question", "to answer your question",
)


def _sentence_count(text: str) -> int:
    return sum(text.count(p) for p in (".", "!", "?"))


def test_ai_relay_migration_seeds_for_existing_install() -> None:
    from src.database import WispernoDB

    logger.info("--- Testing AI Relay migration path (existing-install seeding) ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(db_path)
    try:
        db = WispernoDB(db_path)
        existing_row = db.get_transform("ai_relay")
        assert existing_row is not None, "Fresh DB should already have ai_relay from DEFAULT_TRANSFORMS"

        # Simulate an EXISTING install that predates this transform.
        with db._cursor() as cur:
            cur.execute("DELETE FROM transforms WHERE id = 'ai_relay'")
        assert db.get_transform("ai_relay") is None

        seeded = db.seed_ai_relay_transform()
        assert seeded is True
        transform = db.get_transform("ai_relay")
        assert transform is not None
        assert transform["shortcut"] == "alt+r"
        assert transform["is_active"] == 1
        assert db.seed_ai_relay_transform() is False, "Must not re-seed once it already exists"

        db.close()
        logger.success("PASS: AI Relay migration seeds correctly and is idempotent.")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_ai_relay_preserves_structure_vs_polish() -> None:
    """The core differentiator: on the SAME rambling input, AI Relay must
    remove filler words like Polish does, but must NOT fragment the rambling
    into as many separate sentences as Polish does."""
    from src.config import load_config
    from src.database import WispernoDB
    from src.transformer import Transformer

    logger.info("--- Testing AI Relay preserves rambling structure (vs Polish) ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(db_path)
    try:
        db = WispernoDB(db_path)
        ai_relay = db.get_transform("ai_relay")
        assert ai_relay is not None

        config = load_config()
        transformer = Transformer(config=config.llm, prompts=config.prompts)
        if transformer.llm is None:
            logger.warning("No LLM weights on disk - skipping (nothing to test the transform against).")
            db.close()
            return

        relay_out = transformer.transform_with_prompt(RAMBLE, ai_relay["system_prompt"], label="ai_relay")
        polish_out = transformer.transform(RAMBLE, mode="polish")
        logger.info(f"Input:      {RAMBLE!r}")
        logger.info(f"AI Relay:   {relay_out!r}")
        logger.info(f"Polish:     {polish_out!r}")

        assert relay_out.strip(), "AI Relay produced empty output"
        lowered = relay_out.lower()

        # Filler words removed, same as Polish.
        for filler in ("um ", "uh ", "you know"):
            assert filler not in lowered, f"Filler word {filler!r} survived: {relay_out!r}"

        # Self-correction resolved, same as Polish - the corrected content
        # ("first approach") must win, and the raw markers must not survive.
        assert "wait" not in lowered and " no " not in f" {lowered} ", f"Correction marker leaked: {relay_out!r}"
        assert "first approach" in lowered, f"Self-correction wasn't resolved to the right value: {relay_out!r}"

        # Facts preserved (fidelity) - nothing summarized away.
        assert "staging" in lowered and "alerting" in lowered, f"Content was dropped: {relay_out!r}"

        # The actual differentiator: AI Relay must not fragment the ramble
        # into materially more sentences than it started as - Polish is
        # EXPECTED to add more sentence breaks (that's its whole job), so
        # relay's sentence count must be strictly lower than polish's.
        relay_sentences = _sentence_count(relay_out)
        polish_sentences = _sentence_count(polish_out)
        logger.info(f"Sentence-ending punctuation - AI Relay: {relay_sentences}, Polish: {polish_sentences}")
        assert relay_sentences < polish_sentences, (
            f"AI Relay restructured as much as Polish did (relay={relay_sentences} vs "
            f"polish={polish_sentences} sentence breaks) - it should preserve more of the "
            f"original run-on structure than Polish does."
        )

        db.close()
        logger.success("PASS: AI Relay cleans filler/self-correction but preserves structure Polish would fragment.")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_ai_relay_preserves_profanity_and_capitalizes_i() -> None:
    from src.config import load_config
    from src.database import WispernoDB
    from src.transformer import Transformer

    logger.info("--- Testing AI Relay: profanity untouched, 'I' capitalized ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(db_path)
    try:
        db = WispernoDB(db_path)
        ai_relay = db.get_transform("ai_relay")
        config = load_config()
        transformer = Transformer(config=config.llm, prompts=config.prompts)
        if transformer.llm is None:
            logger.warning("No LLM weights on disk - skipping.")
            db.close()
            return

        output = transformer.transform_with_prompt(PROFANITY_SAMPLE, ai_relay["system_prompt"], label="ai_relay")
        logger.info(f"Input:  {PROFANITY_SAMPLE!r}\nOutput: {output!r}")
        assert "fuck" in output.lower(), f"Profanity was censored: {output!r}"
        assert " i " not in f" {output} ".replace(" I ", " Z "), f"Lowercase standalone 'i' survived: {output!r}"

        db.close()
        logger.success(f"PASS: profanity preserved uncensored: {output!r}")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_ai_relay_no_conversational_drift() -> None:
    from src.config import load_config
    from src.database import WispernoDB
    from src.transformer import Transformer

    logger.info("--- Testing AI Relay conversational-drift guard (adversarial prompts) ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(db_path)
    try:
        db = WispernoDB(db_path)
        ai_relay = db.get_transform("ai_relay")
        config = load_config()
        transformer = Transformer(config=config.llm, prompts=config.prompts)
        if transformer.llm is None:
            logger.warning("No LLM weights on disk - skipping.")
            db.close()
            return

        for trap in CONVERSATIONAL_TRAPS:
            output = transformer.transform_with_prompt(trap, ai_relay["system_prompt"], label="ai_relay")
            lowered = output.lower()
            leaked = [phrase for phrase in CONVERSATIONAL_LEAKS if phrase in lowered]
            bloat = len(trap.split()) <= 15 and len(output.split()) > len(trap.split()) * 3
            logger.info(f"  Trap: {trap!r} -> {output!r}")
            assert not leaked, f"Conversational drift on {trap!r}: leaked phrases {leaked} in {output!r}"
            assert not bloat, f"Conversational drift (bloat) on {trap!r}: {output!r}"

        db.close()
        logger.success("PASS: all conversational traps survived with no drift.")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO AI RELAY TEST SUITE")
    logger.info("=========================================================")
    test_ai_relay_migration_seeds_for_existing_install()
    test_ai_relay_preserves_structure_vs_polish()
    test_ai_relay_preserves_profanity_and_capitalizes_i()
    test_ai_relay_no_conversational_drift()
    logger.success("=========================================================")
    logger.success(" ALL AI RELAY TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
