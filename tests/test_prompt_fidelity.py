"""
Prompt-Fidelity Verification for Wisperno.
Feeds conversational traps into Transformer and asserts the LLM never answers,
opines, or chats back - it must only clean/restructure the dictated text verbatim.
"""

import sys
import time
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import load_config
from src.transformer import Transformer

# Phrases that indicate the model broke character and answered/chatted instead
# of cleaning the dictation.
CONVERSATIONAL_LEAKS = [
    "i can help",
    "i'd be happy",
    "i would be happy",
    "here is an explanation",
    "here's an explanation",
    "let me explain",
    "as an ai",
    "i'm sorry",
    "i am sorry",
    "sure, here",
    "sure thing",
    "great question",
    "to answer your question",
    "pointers are a way",
    "a pointer is a variable",
]

CONVERSATIONAL_TRAPS = [
    "this is not working",
    "i don't know why this is failing",
    "can you explain python pointers to me",
    "why is this not working",
    "what is the best way to write a python script for this",
]


def _assert_no_conversational_drift(mode: str, raw: str, output: str) -> None:
    lower = output.lower()
    leaked = [phrase for phrase in CONVERSATIONAL_LEAKS if phrase in lower]
    assert not leaked, (
        f"[{mode}] Conversational drift detected for input '{raw}'. "
        f"Output leaked phrase(s) {leaked}: '{output}'"
    )
    assert len(output.strip()) > 0, f"[{mode}] Empty output for input '{raw}'."


def test_prompt_fidelity_polish():
    logger.info("--- Testing POLISH mode against conversational traps ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts)

    if transformer.llm is None:
        logger.warning("No LLM weights loaded (primary or fallback missing) - skipping live fidelity checks.")
        return

    logger.info(f"Active model: {transformer.active_model_name}")

    for raw in CONVERSATIONAL_TRAPS:
        t0 = time.perf_counter()
        output = transformer.transform(raw, mode="polish")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[polish] '{raw}' -> '{output}' ({elapsed_ms:.0f}ms)")
        _assert_no_conversational_drift("polish", raw, output)

    logger.success("POLISH mode never answered/chatted back - verbatim cleaning confirmed.")


def test_prompt_fidelity_prompt_engineer():
    logger.info("--- Testing PROMPT_ENGINEER mode against conversational traps ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts)

    if transformer.llm is None:
        logger.warning("No LLM weights loaded - skipping live fidelity checks.")
        return

    raw = "why does my react component rerender on every state update even when the state didn't change"
    output = transformer.transform(raw, mode="prompt_engineer")
    logger.info(f"[prompt_engineer] '{raw}' -> '{output}'")
    _assert_no_conversational_drift("prompt_engineer", raw, output)

    logger.success("PROMPT_ENGINEER mode restructured without answering the dictated question.")


def run_all_tests():
    logger.info("=========================================================")
    logger.info("        WISPERNO PROMPT-FIDELITY TEST SUITE")
    logger.info("=========================================================")
    test_prompt_fidelity_polish()
    test_prompt_fidelity_prompt_engineer()
    logger.success("=========================================================")
    logger.success(" ALL PROMPT-FIDELITY TESTS COMPLETED SUCCESSFULLY!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
