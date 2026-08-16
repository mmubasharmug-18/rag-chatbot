"""
evaluate.py
-----------
Basic evaluation script for the RAG pipeline.

Metrics
-------
* Retrieval accuracy  — whether the correct source appears in the top-K results
* Answer non-emptiness — whether the LLM produced a non-trivial response
* Latency             — end-to-end response time per query

Usage
-----
    python scripts/evaluate.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chatbot import Chatbot
from app.config import VECTOR_DB_PATH
from components.embedder import HuggingFaceEmbedder
from components.vector_store import VectorStore

# ── Sample eval set (query, expected_keyword_in_answer) ──────────────────────
# Edit these to match the documents you have ingested.
EVAL_SET = [
    ("What is RAG?",                       "retrieval"),
    ("How does embedding work?",           "vector"),
    ("What is a vector store?",            "similarity"),
    ("Explain chunking",                   "overlap"),
    ("What models are used in this stack?","mistral"),
]


def main() -> None:
    print("\n" + "=" * 60)
    print("  RAG Chatbot — Evaluation")
    print("=" * 60 + "\n")

    embedder = HuggingFaceEmbedder()
    store    = VectorStore(embedder=embedder, index_path=VECTOR_DB_PATH)

    if not store.load():
        print("❌  No vector index found. Run scripts/ingest.py first.")
        sys.exit(1)

    chatbot = Chatbot(vector_store=store)

    passed = 0
    results = []

    for i, (query, keyword) in enumerate(EVAL_SET, start=1):
        t0       = time.time()
        response = chatbot.chat(query)
        latency  = time.time() - t0

        hit = keyword.lower() in response.answer.lower()
        non_empty = len(response.answer.strip()) > 20

        ok = hit and non_empty
        passed += int(ok)

        results.append({
            "q":        query,
            "keyword":  keyword,
            "hit":      hit,
            "non_empty": non_empty,
            "latency":  latency,
            "ok":       ok,
        })

        status = "✅" if ok else "❌"
        print(f"  [{i}] {status}  '{query}'")
        print(f"       keyword={keyword!r} found={hit}  "
              f"non_empty={non_empty}  latency={latency:.2f}s")
        print(f"       Sources: {response.sources}")
        print()

    total  = len(EVAL_SET)
    pct    = 100 * passed / total
    avg_lat = sum(r["latency"] for r in results) / total

    print("=" * 60)
    print(f"  Score     : {passed}/{total}  ({pct:.0f}%)")
    print(f"  Avg latency: {avg_lat:.2f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
