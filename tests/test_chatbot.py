"""
test_chatbot.py
---------------
Unit tests for app/chatbot.py
All heavy components (VectorStore, Retriever, LLMHandler) are mocked
so tests run instantly without GPU or model downloads.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.schema import Document
from app.chatbot import Chatbot, ChatResponse
from components.vector_store import VectorStore
from components.retriever import Retriever
from components.llm_handler import LLMHandler


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_mock_retriever(docs=None):
    retriever = MagicMock(spec=Retriever)
    docs = docs or []
    retriever.retrieve.return_value = [
        (Document(page_content=d, metadata={"source": "doc.txt"}), 0.8)
        for d in docs
    ]
    return retriever


def make_mock_llm(answer="This is the answer."):
    llm = MagicMock(spec=LLMHandler)
    llm.generate.return_value = answer
    return llm


def make_mock_store(is_ready=True):
    store = MagicMock(spec=VectorStore)
    store.is_ready = is_ready
    return store


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestChatbot:

    def test_returns_chat_response_object(self):
        store     = make_mock_store()
        retriever = make_mock_retriever(["Some context about the topic."])
        llm       = make_mock_llm("The answer is 42.")
        bot       = Chatbot(store, retriever=retriever, llm=llm)

        resp = bot.chat("What is the answer?")
        assert isinstance(resp, ChatResponse)

    def test_answer_comes_from_llm(self):
        store     = make_mock_store()
        retriever = make_mock_retriever(["context"])
        llm       = make_mock_llm("Specific LLM answer here.")
        bot       = Chatbot(store, retriever=retriever, llm=llm)

        resp = bot.chat("question")
        assert resp.answer == "Specific LLM answer here."

    def test_sources_are_populated(self):
        store     = make_mock_store()
        retriever = make_mock_retriever(["context chunk"])
        llm       = make_mock_llm("answer")
        bot       = Chatbot(store, retriever=retriever, llm=llm)

        resp = bot.chat("question")
        assert len(resp.sources) >= 1
        assert "doc.txt" in resp.sources[0]

    def test_empty_query_returns_prompt_message(self):
        store = make_mock_store()
        bot   = Chatbot(store, retriever=make_mock_retriever(), llm=make_mock_llm())
        resp  = bot.chat("")
        assert "please enter" in resp.answer.lower()

    def test_store_not_ready_returns_warning(self):
        store     = make_mock_store(is_ready=False)
        retriever = make_mock_retriever()
        llm       = make_mock_llm()
        bot       = Chatbot(store, retriever=retriever, llm=llm)

        resp = bot.chat("Any question")
        assert "no documents" in resp.answer.lower() or "ingest" in resp.answer.lower()

    def test_no_retrieved_docs_returns_fallback(self):
        store     = make_mock_store()
        retriever = make_mock_retriever([])   # empty retrieval
        llm       = make_mock_llm()
        bot       = Chatbot(store, retriever=retriever, llm=llm)

        resp = bot.chat("unanswerable question")
        assert "couldn't find" in resp.answer.lower() or "no relevant" in resp.answer.lower()
        assert resp.sources == []

    def test_query_whitespace_is_stripped(self):
        store     = make_mock_store()
        retriever = make_mock_retriever(["context"])
        llm       = make_mock_llm("answer")
        bot       = Chatbot(store, retriever=retriever, llm=llm)

        resp = bot.chat("   padded query   ")
        assert retriever.retrieve.called
        called_query = retriever.retrieve.call_args[0][0]
        assert called_query == "padded query"
