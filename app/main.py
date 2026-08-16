"""
main.py
streamlit run app/main.py
http://localhost:7860
-------
Streamlit UI for the RAG chatbot.
Features
--------
* Chat interface with conversation history
* Sidebar: upload documents + trigger ingestion
* Source references with similarity scores displayed below each answer
* Clear Chat History button
* Top-K retrieval slider
* Persistent vector store (loaded from disk on startup)
* Cross-platform temp file handling (Windows + Linux/Mac)
"""

import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Make sure project root is on the path when running `streamlit run app/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chatbot import Chatbot
from app.config import APP_DESCRIPTION, APP_TITLE, DATA_RAW_DIR, VECTOR_DB_PATH
from components.document_loader import load_document
from components.embedder import HuggingFaceEmbedder
from components.text_splitter import split_documents
from components.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "store_ready" not in st.session_state:
    st.session_state.store_ready = False

if "top_k" not in st.session_state:
    st.session_state.top_k = 4


# ── Helper: initialise chatbot (cached) ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading models …")
def get_chatbot() -> tuple:
    embedder = HuggingFaceEmbedder()
    store    = VectorStore(embedder=embedder, index_path=VECTOR_DB_PATH)
    loaded   = store.load()
    chatbot  = Chatbot(vector_store=store)
    return chatbot, loaded


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Knowledge Base")
    st.caption("Upload documents and ingest them into the vector store.")

    uploaded_files = st.file_uploader(
        "Upload PDF / TXT / DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if st.button("⚙️ Ingest Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            with st.spinner("Ingesting documents …"):
                chatbot_obj, _ = get_chatbot()
                all_chunks = []

                for uf in uploaded_files:
                    # BUG FIX: use cross-platform temp directory (works on Windows + Linux)
                    tmp_path = Path(tempfile.gettempdir()) / uf.name
                    with open(tmp_path, "wb") as f:
                        f.write(uf.getbuffer())

                    try:
                        docs   = load_document(tmp_path)
                        for d in docs:
                            d.metadata["source"] = uf.name
                        chunks = split_documents(docs)
                        all_chunks.extend(chunks)
                        st.success(f"✅ {uf.name} — {len(chunks)} chunks")
                    except Exception as exc:
                        st.error(f"❌ {uf.name}: {exc}")

                if all_chunks:
                    chatbot_obj.vector_store.build(all_chunks)
                    st.session_state.store_ready = True
                    st.success(f"Vector store built with {len(all_chunks)} total chunks.")

    st.divider()
    st.caption("Or pre-load documents by placing files in `data/raw/` and running `scripts/ingest.py`.")

    # ── Status badge ─────────────────────────────────────────────────────────
    chatbot_obj, preloaded = get_chatbot()
    ready = preloaded or st.session_state.store_ready
    st.session_state.chatbot = chatbot_obj

    if ready:
        st.success("📚 Knowledge base ready")
    else:
        st.warning("⚠️ No knowledge base loaded")

    # ── ENHANCEMENT 3: Top-K slider ───────────────────────────────────────────
    st.divider()
    top_k = st.slider(
        "Retrieved chunks (Top-K)",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
        help="How many document chunks the retriever fetches per query. Higher = more context but slower.",
    )
    st.session_state.top_k = top_k

    # ── ENHANCEMENT 1: Clear Chat button ─────────────────────────────────────
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat UI ──────────────────────────────────────────────────────────────
st.title(f"🤖 {APP_TITLE}")
st.caption(APP_DESCRIPTION)
st.divider()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

# Chat input
if prompt := st.chat_input("Ask a question about your documents …"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            chatbot_obj = st.session_state.chatbot
            if chatbot_obj is None:
                chatbot_obj, _ = get_chatbot()

            # ENHANCEMENT 3: pass top_k from slider into chat
            top_k = st.session_state.get("top_k", 4)
            response = chatbot_obj.chat(prompt, top_k=top_k)

        st.markdown(response.answer)

        if response.sources:
            with st.expander("📄 Sources"):
                for src in response.sources:
                    st.markdown(f"- `{src}`")

    st.session_state.messages.append({
        "role":    "assistant",
        "content": response.answer,
        "sources": response.sources,
    })
