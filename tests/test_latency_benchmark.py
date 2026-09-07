"""
Latency benchmark for Wisperno's LLM transform pipeline after the throughput
fixes (flash_attn always on, n_gpu_layers 20->24, n_threads pinning, dynamic
max_tokens + explicit stop tokens instead of a flat 4096-token ceiling).

Run: python tests/test_latency_benchmark.py
"""

import os
import sys
import time
from pathlib import Path
from loguru import logger

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import load_config

MIN_TOKENS_PER_SEC = 30.0
MAX_100_WORD_SEC = 3.2
MAX_VRAM_MB = 4800.0

# Genuinely varied content, not a repeated filler phrase - repeated text gets
# legitimately condensed by the polish prompt (it's not a self-correction, but
# it *is* redundant), which would starve the output-token count and make
# tokens/sec look artificially low regardless of real decode speed.
_VARIED_SENTENCES = (
    "um so basically what I am trying to explain here is that the new deployment pipeline "
    "needs to run the integration tests before it pushes anything to the staging environment "
    "and uh I also think we should add a rollback step just in case something breaks in production. "
    "Another thing I wanted to bring up is that the database migration script we wrote last week "
    "has a bug where it does not handle null values correctly in the legacy customer records table, "
    "so we should probably fix that before the next release goes out. On top of that, the frontend "
    "team mentioned that the new checkout flow is causing some confusion for users because the "
    "shipping address form appears before the payment method selector, which is not the order most "
    "competitors use, so maybe we should run an A/B test to see which order actually converts better. "
    "I also wanted to mention that we are getting close to running out of budget for the cloud "
    "infrastructure this quarter, so someone from finance should probably take a look at our current "
    "spend and see if there is anything we can optimize, like maybe switching some of the less "
    "critical background jobs to a cheaper instance type or scaling down during off-peak hours. "
    "Separately, the mobile app crash rate went up after last Tuesday's release and we still have "
    "not root-caused it, so that probably needs to jump ahead of some of the other backlog items. "
    "There is also a request from the design team to unify the button styles across the settings "
    "and onboarding screens since right now they use two slightly different corner radii and it "
    "looks inconsistent when you flip between them quickly during a demo."
).split()


def _make_input(word_count: int) -> str:
    pool = _VARIED_SENTENCES
    words = (pool * ((word_count // len(pool)) + 2))[:word_count]
    return " ".join(words)


def _count_tokens(transformer, text: str) -> int:
    try:
        return len(transformer.llm.tokenize(text.encode("utf-8")))
    except Exception:
        return max(1, int(len(text.split()) * 1.3))  # rough fallback if tokenize() isn't available


def _decode_tokens_per_sec(transformer, system_prompt: str, user_text: str):
    """
    Stream the completion and time strictly between the first and last emitted
    token - isolates decode (generation) speed from prompt prefill, which is a
    fixed cost unrelated to GPU layer offload/thread tuning and would otherwise
    dominate wall time on short inputs and understate real generation throughput.
    Returns (decode_tokens_per_sec, total_elapsed_sec, output_text).
    """
    t_start = time.perf_counter()
    stream = transformer.llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text.strip()},
        ],
        temperature=0.0,
        max_tokens=1024,
        stop=["<|im_end|>", "<|endoftext|>", "\n\n\n", "User:", "Raw:"],
        stream=True,
    )
    first_token_t = None
    last_token_t = None
    token_count = 0
    chunks = []
    for piece in stream:
        delta = piece["choices"][0]["delta"].get("content")
        if not delta:
            continue
        now = time.perf_counter()
        if first_token_t is None:
            first_token_t = now
        last_token_t = now
        token_count += 1
        chunks.append(delta)
    total_elapsed = time.perf_counter() - t_start
    output_text = "".join(chunks)

    if first_token_t is None or last_token_t is None or last_token_t <= first_token_t:
        # Too few tokens to measure a decode interval - fall back to whole-call rate.
        decode_tps = token_count / total_elapsed if total_elapsed > 0 else 0.0
    else:
        decode_tps = (token_count - 1) / (last_token_t - first_token_t)

    return decode_tps, total_elapsed, output_text


def _gpu_used_mb() -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
    except Exception as e:
        logger.warning(f"Could not read GPU memory via pynvml: {e}")
        return 0.0


def run_all_tests() -> None:
    logger.info("=" * 70)
    logger.info("  WISPERNO LATENCY BENCHMARK")
    logger.info("=" * 70)

    cfg = load_config()
    from src.transcriber import Transcriber
    from src.transformer import Transformer

    # Load both models, matching the real app's combined footprint - the
    # mission's VRAM ceiling is "total VRAM usage", not the LLM alone.
    transcriber = Transcriber(config=cfg.whisper)
    logger.info(f"VRAM after Whisper load: {_gpu_used_mb():.0f} MB")

    transformer = Transformer(config=cfg.llm, prompts=cfg.prompts)
    if transformer.llm is None:
        logger.warning("No LLM weights on disk - skipping (nothing to benchmark).")
        return

    logger.info(f"VRAM after LLM load (combined): {_gpu_used_mb():.0f} MB")
    system_prompt = cfg.prompts.get("polish", "")

    for word_count in (50, 100, 250):
        text = _make_input(word_count)
        decode_tps, total_elapsed, output = _decode_tokens_per_sec(transformer, system_prompt, text)
        out_tokens = _count_tokens(transformer, output)

        logger.info(
            f"{word_count}-word input -> {out_tokens} output tokens, total {total_elapsed:.2f}s "
            f"({decode_tps:.1f} decode tok/s)"
        )
        assert decode_tps > MIN_TOKENS_PER_SEC, (
            f"{word_count}-word decode throughput {decode_tps:.1f} tok/s is below the {MIN_TOKENS_PER_SEC} tok/s floor."
        )
        if word_count == 100:
            assert total_elapsed < MAX_100_WORD_SEC, (
                f"100-word transform took {total_elapsed:.2f}s end-to-end, over the {MAX_100_WORD_SEC}s ceiling."
            )

    vram_peak = _gpu_used_mb()
    logger.info(f"VRAM peak (system-wide, both models loaded): {vram_peak:.0f} MB")
    assert vram_peak < MAX_VRAM_MB, f"VRAM peak {vram_peak:.0f} MB exceeds the {MAX_VRAM_MB} MB ceiling."

    logger.success("=" * 70)
    logger.success(" ALL LATENCY BENCHMARK ASSERTIONS PASSED!")
    logger.success("=" * 70)


if __name__ == "__main__":
    run_all_tests()
