"""
Round 24 Sanitizer Verification for Wisperno.
Mission's literal ask: feed 10 diverse spoken passages (questions, stream-of-
consciousness, technical jargon, long paragraphs) and assert 0/10 trigger a
fallback, with punctuation/capitalization correctly applied. Run against both
Standard (the default) and Turbo (the tier that motivated this round's whole
guardrail simplification) - Standard alone would not have caught the real
production problem.

Run: python tests/test_transformer_sanitizer.py
"""

import sys
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Deliberately diverse: questions, stream-of-consciousness with false starts,
# technical jargon, a longer paragraph, common sentence openers that used to
# false-positive (see Round 24's DRIFT_PREFIXES fix), and legitimate paraphrase
# patterns that used to trip the since-removed guarded-word/length checks.
PASSAGES = [
    "can we push the release by a day to make sure qa has enough time",
    "so um yeah i was talking to john and he said the the client wants changes",
    "the api is returning a 500 error when i hit the slash users endpoint and i cannot figure out why",
    "i'm sorry i'm running late to the meeting today",
    "here is what i found after looking through the logs from last night",
    "honestly i am not sure this design is going to work for mobile users",
    "so wait what was i saying oh right um the deadline moved to friday",
    "i was reading this article about machine learning and it mentioned transformers",
    "so um yeah i wanted to give an update on the project uh basically we finished the backend "
    "api last week and now we are working on integrating it with the frontend uh there were a "
    "few issues with authentication that took longer than expected but we got those sorted out",
    "when is the soonest we could get this deployed to production",
]


def _run_suite(label: str, transformer) -> None:
    if transformer.llm is None:
        logger.warning(f"{label}: no LLM weights on disk - skipping.")
        return

    fallback_count = 0
    for raw in PASSAGES:
        out = transformer.transform(raw, mode="polish")
        fell_back = transformer.last_fallback_applied
        if fell_back:
            fallback_count += 1
            logger.error(f"  [FALLBACK] {raw!r} -> {out!r}")
        else:
            starts_capitalized = bool(out) and out[0].isupper()
            has_terminal_punct = bool(out) and out.rstrip()[-1] in ".!?"
            assert starts_capitalized, f"{label}: output not capitalized: {out!r}"
            assert has_terminal_punct, f"{label}: output missing terminal punctuation: {out!r}"
            logger.info(f"  [ok] {raw!r} -> {out!r}")

    assert fallback_count == 0, f"{label}: {fallback_count}/{len(PASSAGES)} passages triggered a fallback."
    logger.success(f"PASS ({label}): 0/{len(PASSAGES)} fallbacks, punctuation/capitalization correct on all.")


def test_standard_zero_fallbacks() -> None:
    from src.config import load_config
    from src.transformer import Transformer

    logger.info("--- Testing Standard preset: 0/10 diverse passages should fall back ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts)
    _run_suite("Standard", transformer)


def test_turbo_zero_fallbacks() -> None:
    """The tier that motivated this round: production reports of the fallback
    warning firing on "virtually every utterance" trace to Turbo's higher
    baseline drift rate (a smaller model) combined with an over-eager
    guardrail. Both were addressed this round - verify the fix here, not just
    on the already-reliable Standard tier."""
    from src.config import load_config, get_base_dir, LLMConfig
    from src.transformer import Transformer

    logger.info("--- Testing Turbo Flagship preset: 0/10 diverse passages should fall back ---")
    config = load_config()
    turbo_path = get_base_dir() / "models" / "Qwen2.5-1.5B-Instruct-Q8_0.gguf"
    if not turbo_path.exists():
        logger.warning(f"'{turbo_path}' not found on disk - skipping (Turbo model not downloaded).")
        return

    turbo_cfg = LLMConfig(
        model_path="models/Qwen2.5-1.5B-Instruct-Q8_0.gguf", n_gpu_layers=-1,
        n_threads=config.llm.n_threads, n_ctx=config.llm.n_ctx, n_batch=config.llm.n_batch,
        kv_cache_quantization=False,
    )
    transformer = Transformer(config=turbo_cfg, prompts=config.prompts)
    _run_suite("Turbo", transformer)


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO SANITIZER FALSE-POSITIVE TEST SUITE")
    logger.info("=========================================================")
    test_standard_zero_fallbacks()
    test_turbo_zero_fallbacks()
    logger.success("=========================================================")
    logger.success(" ALL SANITIZER TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
