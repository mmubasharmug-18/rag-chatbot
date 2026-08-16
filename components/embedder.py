"""
embedder.py
-----------
Wraps a HuggingFace SentenceTransformer model to produce dense vector
embeddings for document chunks and user queries.

The class implements LangChain's Embeddings interface so it can be passed
directly to FAISS, Chroma, or any other LangChain vector store.
"""

import logging
from typing import List

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class HuggingFaceEmbedder(Embeddings):
    """
    Sentence-transformer embedding wrapper compatible with LangChain.

    Args:
        model_name: HuggingFace model identifier.
                    Defaults to config.EMBEDDING_MODEL_NAME.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")

    # ── LangChain Embeddings interface ───────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document strings.

        Args:
            texts: Raw text strings (document chunks).

        Returns:
            List of float vectors, one per input text.
        """
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit-norm → cosine ≡ dot product
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            text: User query string.

        Returns:
            Float vector representing the query.
        """
        vector = self._model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.tolist()
