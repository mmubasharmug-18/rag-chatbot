"""
text_splitter.py
----------------
Splits LangChain Document objects into smaller, overlapping chunks.
Chunk size and overlap are driven by config.py to keep the logic
configurable without touching source code.
"""

import logging
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks with overlap.

    Each output chunk inherits the metadata of its parent document
    (source, page number, etc.) so that source attribution is preserved
    after retrieval.

    Args:
        documents:    List of LangChain Document objects to split.
        chunk_size:   Maximum number of characters per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    if not documents:
        logger.warning("split_documents called with an empty document list.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer splitting at paragraph/sentence boundaries before characters
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,   # adds 'start_index' to metadata for debugging
    )

    chunks = splitter.split_documents(documents)

    logger.info(
        "Split %d document(s) → %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
