"""
config.py
---------
Central configuration for the RAG chatbot.
All paths, model names, and hyperparameters live here.
Import this module instead of hardcoding values anywhere else.
"""

import os
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent.parent

# ── Use /tmp on HF Spaces (read-only filesystem), local path otherwise ────────
IS_HF_SPACE  = os.environ.get("SPACE_ID") is not None

if IS_HF_SPACE:
    DATA_RAW_DIR   = Path("/tmp/data/raw")
    VECTOR_DB_DIR  = Path("/tmp/data/vector_db")
else:
    DATA_RAW_DIR   = BASE_DIR / "data" / "raw"
    VECTOR_DB_DIR  = BASE_DIR / "data" / "vector_db"

VECTOR_DB_PATH = str(VECTOR_DB_DIR / "faiss_index")

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL_REPO   = "TheBloke/phi-2-GGUF"
LLM_MODEL_FILE   = "phi-2.Q4_K_M.gguf"

if IS_HF_SPACE:
    LLM_CACHE_DIR = Path("/tmp/data/models")
else:
    LLM_CACHE_DIR = BASE_DIR / "data" / "models"

LLM_CONTEXT_LEN  = 2048
LLM_MAX_TOKENS   = 512
LLM_TEMPERATURE  = 0.1
LLM_N_THREADS    = int(os.environ.get("LLM_N_THREADS", 4))
LLM_N_GPU_LAYERS = int(os.environ.get("LLM_N_GPU_LAYERS", 0))

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K           = 4
SCORE_THRESHOLD = 0.0

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE       = "RAG Chatbot"
APP_DESCRIPTION = "Ask questions about your uploaded documents."
