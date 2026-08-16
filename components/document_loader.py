"""
document_loader.py
------------------
Loads and parses documents from a file path or directory.
Supports PDF, TXT, DOCX, and HTML.
Returns a list of LangChain Document objects with metadata
(source filename, page number where available).
"""

import logging
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
)

logger = logging.getLogger(__name__)

# Map file extensions → loader class
LOADER_MAP = {
    ".pdf":  PyMuPDFLoader,
    ".txt":  TextLoader,
    ".docx": Docx2txtLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm":  UnstructuredHTMLLoader,
}


def load_document(file_path: str | Path) -> List[Document]:
    """
    Load a single document and return a list of LangChain Document objects.

    Args:
        file_path: Absolute or relative path to the document.

    Returns:
        List of Document objects with `.page_content` and `.metadata`.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    loader_cls = LOADER_MAP.get(ext)

    if loader_cls is None:
        supported = ", ".join(LOADER_MAP.keys())
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: {supported}"
        )

    logger.info("Loading %s with %s", file_path.name, loader_cls.__name__)
    loader = loader_cls(str(file_path))
    docs = loader.load()

    # Normalise metadata: ensure 'source' key is always present
    for doc in docs:
        doc.metadata.setdefault("source", file_path.name)

    logger.info("Loaded %d page(s) from %s", len(docs), file_path.name)
    return docs


def load_documents_from_directory(directory: str | Path) -> List[Document]:
    """
    Recursively load all supported documents from a directory.

    Args:
        directory: Path to the folder containing source documents.

    Returns:
        Flat list of Document objects from all files found.
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    all_docs: List[Document] = []
    supported_exts = set(LOADER_MAP.keys())

    for file_path in sorted(directory.rglob("*")):
        if file_path.suffix.lower() in supported_exts:
            try:
                docs = load_document(file_path)
                all_docs.extend(docs)
            except Exception as exc:
                logger.warning("Skipping %s — %s", file_path.name, exc)

    logger.info(
        "Loaded %d document chunk(s) from directory '%s'",
        len(all_docs),
        directory,
    )
    return all_docs
