"""
prompt_template.py
------------------
Defines and builds the RAG prompt sent to the LLM.
Updated in Enhancement 2:
- format_sources() now accepts (Document, float) tuples and includes
  similarity scores in the output, e.g. "Pakistan.pdf (score: 0.84)"
"""

from typing import List, Tuple

from langchain.schema import Document

# ── System instruction ────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the "
    "provided context documents. "
    "If the answer is not contained in the context, say: "
    "'I don't have enough information in the provided documents to answer that.' "
    "Always be concise and accurate. "
    "At the end of your answer, list the source document(s) you used."
)

# ── Template ─────────────────────────────────────────────────────────────────
_TEMPLATE = """\
### System
{system_prompt}
### Context Documents
{context}
### Question
{question}
### Answer
"""


def build_prompt(question: str, context_docs: List[Document]) -> str:
    """
    Build the full prompt string for the LLM.
    Args:
        question:     The user's natural-language question.
        context_docs: Retrieved Document chunks to inject as context.
    Returns:
        Formatted prompt string ready to pass to the LLM.
    """
    if not context_docs:
        context_str = "No relevant documents were found."
    else:
        parts = []
        for i, doc in enumerate(context_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page   = doc.metadata.get("page", "")
            loc    = f"{source}, page {page}" if page != "" else source
            parts.append(f"[{i}] (Source: {loc})\n{doc.page_content.strip()}")
        context_str = "\n\n".join(parts)

    return _TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=context_str,
        question=question.strip(),
    )


def format_sources(results: List[Tuple[Document, float]]) -> List[str]:
    """
    Build a human-readable list of source references from retrieved chunks.
    Includes similarity scores so users can judge retrieval confidence.
    Args:
        results: List of (Document, score) tuples from the retriever.
    Returns:
        Deduplicated list of source strings,
        e.g. ["Pakistan.pdf, page 0  (score: 0.84)"]
    """
    seen: set[str] = set()
    sources: List[str] = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        ref    = f"{source}, page {page}" if page != "" else source
        ref_with_score = f"{ref}  (score: {score:.2f})"
        if ref not in seen:
            seen.add(ref)
            sources.append(ref_with_score)
    return sources
