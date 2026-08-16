"""
chatbot.py
----------
Orchestrates the full RAG pipeline for a single user query.
Pipeline
--------
  1. Receive user query
  2. Retrieve top-K relevant chunks from the vector store
  3. Build a RAG prompt (context + question)
  4. Generate an answer with the LLM
  5. Return answer text + list of source references with similarity scores
This module is the sole interface used by app/main.py; the UI layer
never touches individual components directly.
"""

import logging
from dataclasses import dataclass, field
from typing import List

from components.llm_handler import LLMHandler
from components.prompt_template import build_prompt, format_sources
from components.retriever import Retriever
from components.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Container for a single chatbot turn."""
    answer:  str
    sources: List[str] = field(default_factory=list)
    query:   str = ""


class Chatbot:
    """
    End-to-end RAG chatbot.
    Args:
        vector_store: Pre-built or loaded VectorStore instance.
        retriever:    Retriever wrapping the vector store (created automatically
                      if not provided).
        llm:          LLMHandler for text generation (created automatically if
                      not provided).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        retriever: Retriever | None = None,
        llm: LLMHandler | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.retriever    = retriever or Retriever(vector_store)
        self.llm          = llm or LLMHandler()

    # ── Public API ───────────────────────────────────────────────────────────

    def chat(self, query: str, top_k: int | None = None) -> ChatResponse:
        """
        Process a user query through the full RAG pipeline.
        Args:
            query:  The user's natural-language question.
            top_k:  Number of chunks to retrieve. Overrides config default when set.
        Returns:
            ChatResponse containing the answer and source references with scores.
        """
        query = query.strip()
        if not query:
            return ChatResponse(
                answer="Please enter a question.",
                sources=[],
                query=query,
            )

        if not self.vector_store.is_ready:
            return ChatResponse(
                answer=(
                    "No documents have been ingested yet. "
                    "Please upload documents and run ingestion first."
                ),
                sources=[],
                query=query,
            )

        logger.info("Processing query: '%s'", query[:100])

        # Step 1 — Retrieve (pass top_k if provided)
        results      = self.retriever.retrieve(query, k=top_k)
        context_docs = [doc for doc, _ in results]

        if not context_docs:
            return ChatResponse(
                answer=(
                    "I couldn't find any relevant information in the knowledge base "
                    "to answer your question."
                ),
                sources=[],
                query=query,
            )

        # Step 2 — Build prompt
        prompt = build_prompt(query, context_docs)

        # Step 3 — Generate
        answer = self.llm.generate(prompt)

        # Step 4 — Collect sources with similarity scores
        sources = format_sources(results)

        logger.info(
            "Query answered. Sources: %s | Answer length: %d chars",
            sources,
            len(answer),
        )

        return ChatResponse(answer=answer, sources=sources, query=query)
