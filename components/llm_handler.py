"""
llm_handler.py
--------------
Loads and runs the open-source LLM (Mistral-7B GGUF) via llama-cpp-python.

Design decisions
----------------
* GGUF 4-bit quantisation (Q4_K_M) keeps RAM usage under 6 GB — fits the
  HuggingFace Spaces free tier (16 GB CPU RAM).
* The model is downloaded from HuggingFace Hub at first startup and cached
  locally under data/models/ so subsequent starts are fast.
* GPU layers default to 0 (CPU-only) but can be set via the LLM_N_GPU_LAYERS
  environment variable for local development with a GPU.
"""

import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from app.config import (
    LLM_CACHE_DIR,
    LLM_CONTEXT_LEN,
    LLM_MAX_TOKENS,
    LLM_MODEL_FILE,
    LLM_MODEL_REPO,
    LLM_N_GPU_LAYERS,
    LLM_N_THREADS,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class LLMHandler:
    """
    Wraps llama-cpp-python to provide a simple generate(prompt) interface.

    The model is lazily loaded on the first call to generate() to avoid
    blocking the UI startup.
    """

    def __init__(self) -> None:
        self._llm: Llama | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """
        Run inference on the given prompt and return the generated text.

        Args:
            prompt: Fully formatted RAG prompt string.

        Returns:
            Generated answer string (stripped of leading/trailing whitespace).
        """
        llm = self._get_or_load_model()
        logger.debug("Running LLM inference (prompt length=%d chars) …", len(prompt))

        output = llm(
            prompt,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            stop=["[INST]", "</s>"],
            echo=False,
        )
        answer = output["choices"][0]["text"].strip()
        logger.debug("LLM generated %d chars.", len(answer))
        return answer

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get_or_load_model(self) -> Llama:
        if self._llm is None:
            model_path = self._download_model()
            logger.info("Loading LLM from '%s' …", model_path)
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=LLM_CONTEXT_LEN,
                n_threads=LLM_N_THREADS,
                n_gpu_layers=LLM_N_GPU_LAYERS,
                verbose=False,
            )
            logger.info("LLM ready.")
        return self._llm

    @staticmethod
    def _download_model() -> Path:
        """
        Download the GGUF model file from HuggingFace Hub if not cached.

        Returns:
            Local path to the model file.
        """
        cache_dir = Path(LLM_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / LLM_MODEL_FILE

        if local_path.exists():
            logger.info("Model already cached at '%s'.", local_path)
            return local_path

        logger.info(
            "Downloading '%s' from '%s' (this may take a few minutes) …",
            LLM_MODEL_FILE,
            LLM_MODEL_REPO,
        )
        downloaded = hf_hub_download(
            repo_id=LLM_MODEL_REPO,
            filename=LLM_MODEL_FILE,
            local_dir=str(cache_dir),
            token=os.environ.get("HF_TOKEN"),   # optional; needed for gated models
        )
        logger.info("Model downloaded to '%s'.", downloaded)
        return Path(downloaded)
