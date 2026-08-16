"""
ingest.py
---------
One-time CLI script to load all documents from data/raw/,
split them into chunks, embed, and persist the FAISS index.

Usage
-----
    python scripts/ingest.py
    python scripts/ingest.py --data-dir /path/to/docs
    python scripts/ingest.py --chunk-size 600 --chunk-overlap 60
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_RAW_DIR, VECTOR_DB_PATH
from components.document_loader import load_documents_from_directory
from components.embedder import HuggingFaceEmbedder
from components.text_splitter import split_documents
from components.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into FAISS vector store.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_RAW_DIR),
        help="Directory containing source documents (default: data/raw/)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Characters per chunk (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Overlap between chunks (default: {CHUNK_OVERLAP})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  RAG Chatbot — Document Ingestion")
    print("=" * 60)
    print(f"  Source dir  : {data_dir}")
    print(f"  Chunk size  : {args.chunk_size} chars")
    print(f"  Overlap     : {args.chunk_overlap} chars")
    print(f"  Index path  : {VECTOR_DB_PATH}")
    print("=" * 60 + "\n")

    # ── Step 1: Load documents ────────────────────────────────────────────────
    t0 = time.time()
    print("📄 Step 1/4  Loading documents …")
    docs = load_documents_from_directory(data_dir)
    if not docs:
        logger.error("No supported documents found in '%s'.", data_dir)
        sys.exit(1)
    print(f"   Loaded {len(docs)} page(s) in {time.time()-t0:.1f}s\n")

    # ── Step 2: Split ─────────────────────────────────────────────────────────
    print("✂️  Step 2/4  Splitting into chunks …")
    t1 = time.time()
    chunks = split_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"   Created {len(chunks)} chunks in {time.time()-t1:.1f}s\n")

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    print("🔢 Step 3/4  Loading embedding model …")
    t2 = time.time()
    embedder = HuggingFaceEmbedder()
    print(f"   Model ready in {time.time()-t2:.1f}s\n")

    # ── Step 4: Build & persist vector store ──────────────────────────────────
    print("🗄️  Step 4/4  Building FAISS index …")
    t3 = time.time()
    store = VectorStore(embedder=embedder, index_path=VECTOR_DB_PATH)
    store.build(chunks)
    print(f"   Index saved in {time.time()-t3:.1f}s\n")

    total = time.time() - t0
    print("=" * 60)
    print(f"  ✅ Ingestion complete in {total:.1f}s")
    print(f"     {len(chunks)} chunks indexed and saved to '{VECTOR_DB_PATH}'")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
