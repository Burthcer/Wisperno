"""
Model Asset Acquisition Script for Wisperno.
Automates downloading and verifying Whisper STT and GGUF SLM model weights into models/.
"""

import os
import sys
import time
from pathlib import Path
from loguru import logger
from huggingface_hub import hf_hub_download
from faster_whisper import WhisperModel

from src.config import load_config, get_base_dir


def ensure_models_dir() -> Path:
    """Ensure models directory exists."""
    models_dir = get_base_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def download_whisper_model(model_name: str = "large-v3-turbo") -> None:
    """
    Download and cache the faster-whisper CTranslate2 model.
    """
    logger.info(f"Checking / downloading Whisper model: '{model_name}'...")
    t0 = time.perf_counter()
    try:
        # Initializing WhisperModel with download_root downloads/verifies model assets
        WhisperModel(model_name, device="cpu", compute_type="int8", download_root="models")
        elapsed = time.perf_counter() - t0
        logger.success(f"Whisper model '{model_name}' is ready ({elapsed:.1f}s).")
    except Exception as e:
        logger.error(f"Failed to download Whisper model '{model_name}': {e}")
        raise


def download_llm_model(repo_id: str, filename: str, local_dir: Path) -> Path:
    """
    Download GGUF weights from Hugging Face with resume capability.
    """
    target_path = local_dir / filename
    if target_path.exists() and target_path.stat().st_size > 100 * 1024 * 1024:
        logger.success(
            f"LLM model '{filename}' already exists ({target_path.stat().st_size / (1024*1024):.1f} MB)."
        )
        return target_path

    logger.info(f"Downloading LLM '{filename}' from repo '{repo_id}' to '{local_dir}'...")
    t0 = time.perf_counter()
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        elapsed = time.perf_counter() - t0
        size_mb = Path(downloaded_path).stat().st_size / (1024 * 1024)
        logger.success(f"LLM model downloaded successfully: {size_mb:.1f} MB in {elapsed:.1f}s.")
        return Path(downloaded_path)
    except Exception as e:
        logger.error(f"Failed to download LLM model '{filename}': {e}")
        raise


def main():
    logger.info("=" * 60)
    logger.info("Wisperno Model Downloader & Verification")
    logger.info("=" * 60)

    config = load_config()
    models_dir = ensure_models_dir()

    # 1. Download Whisper model
    download_whisper_model(config.whisper.model_name)

    # 2. Download primary LLM model (e.g. Qwen 2.5 7B Instruct)
    download_llm_model(
        repo_id=config.llm.repo_id,
        filename=config.llm.filename,
        local_dir=models_dir,
    )

    # 3. Download lightweight fallback LLM model, used automatically if the
    #    primary model isn't on disk yet (e.g. still mid-download).
    download_llm_model(
        repo_id=config.llm.fallback_repo_id,
        filename=config.llm.fallback_filename,
        local_dir=models_dir,
    )

    logger.success("All required models are verified and ready for offline inference!")


if __name__ == "__main__":
    main()
