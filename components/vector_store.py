"""
vector_store.py
---------------
Creates, persists, and loads a FAISS vector index.

Key behaviours
--------------
* build()  — embeds a list of Document chunks and writes the index to disk.
* load()   — reads a previously saved index from disk (avoids re-embedding).
* search() — similarity search returning top-K (Document, score) pairs.

The index is stored under data/vector_db/ (configurable via config.py).
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from app.config import VECTOR_DB_PATH
from components.embedder import HuggingFaceEmbedder

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Thin wrapper around LangChain's FAISS integration.

    Args:
        embedder: An initialised HuggingFaceEmbedder (or any Embeddings impl).
        index_path: Directory path where the FAISS index is saved/loaded.
    """

    def __init__(
        self,
        embedder: HuggingFaceEmbedder | None = None,
        index_path: str = VECTOR_DB_PATH,
    ) -> None:
        self.embedder    = embedder or HuggingFaceEmbedder()
        self.index_path  = index_path
        self._store: FAISS | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def build(self, documents: List[Document]) -> None:
        """
        Embed documents and create a new FAISS index, overwriting any existing one.

        Args:
            documents: List of chunked Document objects to index.
        """
        if not documents:
            raise ValueError("Cannot build a vector store from an empty document list.")

        logger.info("Building FAISS index from %d chunks …", len(documents))
        self._store = FAISS.from_documents(documents, self.embedder)
        self._persist()
        logger.info("FAISS index saved to '%s'.", self.index_path)

    def load(self) -> bool:
        """
        Load a persisted FAISS index from disk.

        Returns:
            True if the index was loaded successfully, False if not found.
        """
        index_file = Path(self.index_path) / "index.faiss"
        if not index_file.exists():
            logger.info("No existing FAISS index found at '%s'.", self.index_path)
            return False

        logger.info("Loading FAISS index from '%s' …", self.index_path)
        self._store = FAISS.load_local(
            self.index_path,
            self.embedder,
            allow_dangerous_deserialization=True,  # safe: we created the file
        )
        logger.info("FAISS index loaded (%d vectors).", self._store.index.ntotal)
        return True

    def search(
        self,
        query: str,
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search and return top-K (Document, score) pairs.

        Args:
            query: User query string.
            k:     Number of results to return.

        Returns:
            List of (Document, relevance_score) tuples, best match first.

        Raises:
            RuntimeError: If the index has not been built or loaded yet.
        """
        self._require_store()
        results = self._store.similarity_search_with_relevance_scores(query, k=k)
        return results

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to an already-loaded index and persist.

        Args:
            documents: New chunks to add.
        """
        self._require_store()
        self._store.add_documents(documents)
        self._persist()
        logger.info("Added %d chunks to existing index.", len(documents))

    @property
    def is_ready(self) -> bool:
        """True if the store has been built or loaded."""
        return self._store is not None

    # ── Private helpers ──────────────────────────────────────────────────────

    def _persist(self) -> None:
        # Ensure the directory exists (critical on HF Spaces /tmp)
        Path(self.index_path).mkdir(parents=True, exist_ok=True)
        self._store.save_local(self.index_path)

    def _require_store(self) -> None:
        if self._store is None:
            raise RuntimeError(
                "Vector store is not ready. Call build() or load() first."
            )
