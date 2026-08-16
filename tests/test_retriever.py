"""
test_retriever.py
-----------------
Unit tests for components/retriever.py
Uses a mock VectorStore to avoid loading real models in CI.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.schema import Document
from components.retriever import Retriever
from components.vector_store import VectorStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_mock_store(results=None, is_ready=True):
    store = MagicMock(spec=VectorStore)
    store.is_ready = is_ready
    store.search.return_value = results or []
    return store


def make_doc(content, source="test.txt"):
    return Document(page_content=content, metadata={"source": source})


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRetriever:

    def test_returns_results_above_threshold(self):
        docs = [
            (make_doc("relevant content"), 0.8),
            (make_doc("also relevant"),    0.7),
        ]
        store     = make_mock_store(results=docs)
        retriever = Retriever(store, top_k=4, score_threshold=0.5)
        results   = retriever.retrieve("test query")
        assert len(results) == 2

    def test_filters_below_threshold(self):
        docs = [
            (make_doc("great match"),  0.9),
            (make_doc("poor match"),   0.1),
        ]
        store     = make_mock_store(results=docs)
        retriever = Retriever(store, top_k=4, score_threshold=0.5)
        results   = retriever.retrieve("test query")
        assert len(results) == 1
        assert results[0][0].page_content == "great match"

    def test_deduplicates_identical_content(self):
        doc = make_doc("duplicate content")
        docs = [(doc, 0.9), (doc, 0.8)]
        store     = make_mock_store(results=docs)
        retriever = Retriever(store, score_threshold=0.0)
        results   = retriever.retrieve("query")
        assert len(results) == 1

    def test_returns_empty_when_store_not_ready(self):
        store     = make_mock_store(is_ready=False)
        retriever = Retriever(store)
        results   = retriever.retrieve("anything")
        assert results == []

    def test_retrieve_documents_strips_scores(self):
        docs = [(make_doc("content"), 0.75)]
        store     = make_mock_store(results=docs)
        retriever = Retriever(store, score_threshold=0.0)
        plain     = retriever.retrieve_documents("query")
        assert isinstance(plain[0], Document)
        assert plain[0].page_content == "content"

    def test_search_exception_returns_empty(self):
        store = make_mock_store()
        store.search.side_effect = RuntimeError("index error")
        retriever = Retriever(store)
        results   = retriever.retrieve("query")
        assert results == []
