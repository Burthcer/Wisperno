"""
Head-to-head benchmark: Qwen2.5-1.5B-Instruct-Q8_0 (Turbo Flagship, shipped
default) vs. Gemma-2-2B-it-Q4_K_M (the experimental "Gemma 2 2B (Beta)"
preset) - real GPU measurements, not vendor-quoted numbers, across the 4
pillars this evaluation was scoped to: grammar/spelling repair, typography
normalization, style rewriting, and throughput/latency/VRAM.

Requires both GGUF files present under models/ (download_models.py fetches
the primary Turbo file; the Gemma-2 file is a one-off manual download - see
download_models.download_llm_model("bartowski/gemma-2-2b-it-GGUF",
"gemma-2-2b-it-Q4_K_M.gguf", ...)).

Run: python tests/benchmark_engine_candidates.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

GRAMMAR_SAMPLES = [
    "he go to store.yesterday he buyed fruit",
    "she dont know nothing about it",
]
TYPOGRAPHY_SAMPLES = [
    "first.second,third",
    "end of sentence.next sentence",
    "Hello.How are you?",
]
STYLE_SAMPLE = "the meeting got pushed back again and now i have to redo the whole schedule"
LATENCY_SAMPLES = {
    "10w": "um so i think we should uh go with the the second option",
    "50w": (
        "um so i think we should uh go with the the second option because it seems more "
        "scalable and easier to maintain long term you know and also the team already has "
        "experience with this kind of architecture from the previous project we did last year"
    ),
    "200w": (
        "so basically what happened is the deployment pipeline failed again last night around "
        "2am and i got paged and had to wake up and look at the logs and it turns out the issue "
        "was related to a stale cache entry that was not being invalidated properly when the "
        "config changed so what i think we need to do is add a proper cache busting mechanism "
        "that uses a hash of the config file contents instead of relying on a manual version "
        "bump because clearly people keep forgetting to bump it and this has caused at least "
        "three incidents now in the past two months and each time it takes about an hour to "
        "diagnose because the error message is completely misleading it just says connection "
        "refused which makes you think its a network issue but really its the stale cache "
        "serving an old endpoint that no longer exists so we should also improve the error "
        "message to include which cache key was stale and when it was last refreshed so the on "
        "call person does not have to go spelunking through the logs every single time this "
        "happens again in the future"
    ),
}


def _vram_used_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    return int(out.splitlines()[0]) if out else -1


def _build_transformer(gemma: bool):
    from src.config import load_config, LLMConfig
    from src.transformer import Transformer

    config = load_config()
    if not gemma:
        return Transformer(config=config.llm, prompts=config.prompts), config

    llm_cfg = LLMConfig(
        repo_id="bartowski/gemma-2-2b-it-GGUF",
        filename="gemma-2-2b-it-Q4_K_M.gguf",
        model_path="models/gemma-2-2b-it-Q4_K_M.gguf",
        n_gpu_layers=-1, n_threads=config.llm.n_threads, n_ctx=4096, n_batch=config.llm.n_batch,
        kv_cache_quantization=False, flash_attn=False, chat_style="gemma",
        temperature=0.0, max_tokens=4096,
        fallback_repo_id=config.llm.fallback_repo_id, fallback_filename=config.llm.fallback_filename,
        fallback_model_path=config.llm.fallback_model_path,
    )
    return Transformer(config=llm_cfg, prompts=config.prompts), config


def _run_battery(transformer, config, label: str) -> dict:
    results = {"label": label}

    grammar_out = [transformer.transform(s, mode="polish") for s in GRAMMAR_SAMPLES]
    results["grammar"] = list(zip(GRAMMAR_SAMPLES, grammar_out))

    typo_out = [transformer.transform(s, mode="polish") for s in TYPOGRAPHY_SAMPLES]
    results["typography"] = list(zip(TYPOGRAPHY_SAMPLES, typo_out))

    from src.writing_styles import WRITING_STYLES
    style_out = {}
    for style in WRITING_STYLES[:4]:  # Professional/Polite/Casual/Emojify, per the mission's own 4-style ask
        style_out[style.title] = transformer.transform_with_prompt(STYLE_SAMPLE, style.system_prompt, label=style.id)
    results["styles"] = style_out

    latency = {}
    for tag, text in LATENCY_SAMPLES.items():
        t0 = time.perf_counter()
        out = transformer.transform(text, mode="polish")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latency[tag] = {"ms": elapsed_ms, "words_out": len(out.split())}
    results["latency"] = latency

    return results


def run_comparison() -> None:
    logger.info("=========================================================")
    logger.info("   ENGINE CANDIDATE BENCHMARK: Turbo Flagship vs. Gemma 2 2B (Beta)")
    logger.info("=========================================================")

    gemma_path = BASE_DIR / "models" / "gemma-2-2b-it-Q4_K_M.gguf"
    if not gemma_path.exists():
        logger.warning(f"'{gemma_path}' not found - skipping the comparison (baseline-only checks still run).")
        return

    vram_before = _vram_used_mb()
    turbo_t, config = _build_transformer(gemma=False)
    assert turbo_t.llm is not None, "Turbo Flagship failed to load - cannot benchmark"
    vram_turbo = _vram_used_mb()
    turbo_results = _run_battery(turbo_t, config, "Turbo Flagship (Qwen2.5-1.5B-Q8_0)")
    del turbo_t

    gemma_t, config = _build_transformer(gemma=True)
    assert gemma_t.llm is not None, "Gemma 2 2B (Beta) failed to load"
    vram_combined = _vram_used_mb()
    gemma_results = _run_battery(gemma_t, config, "Gemma 2 2B Beta (gemma-2-2b-it-Q4_K_M)")

    logger.info("")
    logger.info("--- Grammar & Spelling Repair ---")
    for (raw, t_out), (_, g_out) in zip(turbo_results["grammar"], gemma_results["grammar"]):
        logger.info(f"  Input:  {raw!r}")
        logger.info(f"  Turbo:  {t_out!r}")
        logger.info(f"  Gemma:  {g_out!r}")

    logger.info("")
    logger.info("--- Typography Normalization ---")
    for (raw, t_out), (_, g_out) in zip(turbo_results["typography"], gemma_results["typography"]):
        logger.info(f"  Input:  {raw!r}")
        logger.info(f"  Turbo:  {t_out!r}")
        logger.info(f"  Gemma:  {g_out!r}")

    logger.info("")
    logger.info("--- Style Rewriting ---")
    for style_name in turbo_results["styles"]:
        logger.info(f"  [{style_name}]")
        logger.info(f"    Turbo: {turbo_results['styles'][style_name]!r}")
        logger.info(f"    Gemma: {gemma_results['styles'][style_name]!r}")

    logger.info("")
    logger.info("--- Throughput & Latency ---")
    logger.info(f"{'Input':<8} {'Turbo ms':<12} {'Gemma ms':<12}")
    for tag in LATENCY_SAMPLES:
        logger.info(
            f"{tag:<8} {turbo_results['latency'][tag]['ms']:<12.1f} {gemma_results['latency'][tag]['ms']:<12.1f}"
        )

    logger.info("")
    logger.info("--- VRAM ---")
    logger.info(f"  Baseline (nothing loaded): {vram_before}MB")
    logger.info(f"  + Turbo Flagship LLM:      {vram_turbo}MB")
    logger.info(f"  + Gemma 2 2B (replacing):  {vram_combined}MB")

    logger.info("=========================================================")
    logger.success(" COMPARISON COMPLETE - see the tables above for the verdict.")
    logger.info("=========================================================")


def test_gemma_beta_produces_non_empty_output_on_every_category() -> None:
    """Not a 'Gemma must win' assertion (it measurably doesn't, on speed - see
    run_comparison()'s printed numbers) - this only confirms the beta preset
    is functionally alive: loads, and produces real, non-empty, non-crashing
    output across every category, which is the actual bar for shipping it as
    a labeled, opt-in Beta (not the default)."""
    gemma_path = BASE_DIR / "models" / "gemma-2-2b-it-Q4_K_M.gguf"
    if not gemma_path.exists():
        logger.warning("Gemma-2 GGUF not present - skipping.")
        return

    logger.info("--- Testing Gemma 2 2B (Beta): produces real output on every category ---")
    transformer, config = _build_transformer(gemma=True)
    assert transformer.llm is not None

    results = _run_battery(transformer, config, "gemma_beta")
    for raw, out in results["grammar"]:
        assert out.strip(), f"Empty output for grammar sample {raw!r}"
    for raw, out in results["typography"]:
        assert out.strip(), f"Empty output for typography sample {raw!r}"
        assert out[0].isupper() or not out[0].isalpha(), f"Sentence start not capitalized: {out!r}"
    for style_name, out in results["styles"].items():
        assert out.strip(), f"Empty output for style '{style_name}'"
    logger.success("PASS: Gemma 2 2B (Beta) produces real, non-empty output across all 4 categories.")


def test_combined_vram_stays_under_5gb_ceiling() -> None:
    gemma_path = BASE_DIR / "models" / "gemma-2-2b-it-Q4_K_M.gguf"
    if not gemma_path.exists():
        logger.warning("Gemma-2 GGUF not present - skipping.")
        return

    logger.info("--- Testing combined VRAM (Whisper + Gemma 2 2B) stays under the 5.0GB ceiling ---")
    from src.config import load_config
    from src.transcriber import Transcriber
    from src.transformer import Transformer
    from src.vocabulary import Vocabulary

    config = load_config()
    _ = Transcriber(config=config.whisper, vocabulary=Vocabulary(db=None))
    transformer, _ = _build_transformer(gemma=True)
    assert transformer.llm is not None

    used_mb = _vram_used_mb()
    assert used_mb < 5000, f"Combined VRAM {used_mb}MB exceeds the 5.0GB ceiling"
    logger.success(f"PASS: combined VRAM {used_mb}MB is under the 5.0GB ceiling.")


def run_all_tests() -> None:
    test_combined_vram_stays_under_5gb_ceiling()
    test_gemma_beta_produces_non_empty_output_on_every_category()
    run_comparison()


if __name__ == "__main__":
    run_all_tests()
