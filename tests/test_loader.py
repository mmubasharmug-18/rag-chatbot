"""
test_loader.py
--------------
Unit tests for components/document_loader.py
"""

import os
import tempfile
from pathlib import Path

import pytest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.document_loader import load_document, load_documents_from_directory


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def txt_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello, this is a test document.\nIt has two lines.")
    return f


@pytest.fixture
def empty_dir(tmp_path):
    return tmp_path


@pytest.fixture
def docs_dir(tmp_path):
    (tmp_path / "a.txt").write_text("Document A content.")
    (tmp_path / "b.txt").write_text("Document B content.")
    (tmp_path / "ignored.csv").write_text("col1,col2\n1,2")
    return tmp_path


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLoadDocument:

    def test_loads_txt_file(self, txt_file):
        docs = load_document(txt_file)
        assert len(docs) >= 1
        assert "Hello" in docs[0].page_content

    def test_metadata_has_source(self, txt_file):
        docs = load_document(txt_file)
        assert "source" in docs[0].metadata
        assert docs[0].metadata["source"] == "sample.txt"

    def test_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/path/file.txt")

    def test_raises_for_unsupported_type(self, tmp_path):
        bad = tmp_path / "data.csv"
        bad.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(bad)


class TestLoadDocumentsFromDirectory:

    def test_loads_supported_files(self, docs_dir):
        docs = load_documents_from_directory(docs_dir)
        sources = {d.metadata["source"] for d in docs}
        assert "a.txt" in sources
        assert "b.txt" in sources
        assert not any("csv" in s for s in sources), "CSV should be ignored"

    def test_empty_dir_returns_empty_list(self, empty_dir):
        docs = load_documents_from_directory(empty_dir)
        assert docs == []

    def test_raises_for_non_directory(self, txt_file):
        with pytest.raises(NotADirectoryError):
            load_documents_from_directory(txt_file)
