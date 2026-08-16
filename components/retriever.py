"""
retriever.py
------------
Wraps VectorStore to provide a clean retrieve(query, k) interface.

Handles edge cases:
  - Empty result set (no documents indexed)
  - Score thresholding to suppress low-confidence chunks
  - Deduplication of chunks with identical content
"""

import logging
from typing import List, Tuple

from langchain.schema import Document

from app.config import TOP_K, SCORE_THRESHOLD
from components.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves the most relevant document chunks for a given query.

    Args:
        vector_store:     An initialised (built or loaded) VectorStore.
        top_k:            Default number of chunks to return.
        score_threshold:  Minimum relevance score; chunks below this are dropped.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = TOP_K,
        score_threshold: float = SCORE_THRESHOLD,
    ) -> None:
        self.vector_store    = vector_store
        self.top_k           = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve the top-K most relevant chunks for a query.

        Args:
            query: User's natural-language question.
            k:     Override for the number of results (falls back to self.top_k).

        Returns:
            List of (Document, score) tuples sorted by descending relevance.
            Empty list if the vector store has no documents or nothing passes
            the score threshold.
        """
        if not self.vector_store.is_ready:
            logger.warning("Retriever called before vector store is ready.")
            return []

        k = k or self.top_k

        try:
            results = self.vector_store.search(query, k=k)
        except Exception as exc:
            logger.error("Vector store search failed: %s", exc)
            return []

        # Apply score threshold
        filtered = [(doc, score) for doc, score in results if score >= self.score_threshold]

        if not filtered:
            logger.info("No chunks passed the score threshold (%.2f).", self.score_threshold)
            return []

        # Deduplicate by content (keep first occurrence = highest score)
        seen: set[str] = set()
        unique: List[Tuple[Document, float]] = []
        for doc, score in filtered:
            content_key = doc.page_content.strip()
            if content_key not in seen:
                seen.add(content_key)
                unique.append((doc, score))

        logger.debug("Retrieved %d unique chunks for query: '%s'", len(unique), query[:80])
        return unique

    def retrieve_documents(self, query: str, k: int | None = None) -> List[Document]:
        """
        Convenience wrapper — returns only Document objects (no scores).
        """
        return [doc for doc, _ in self.retrieve(query, k=k)]
