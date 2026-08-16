---
title: RAG Chatbot
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# 🤖 RAG Chatbot

A fully open-source, free-to-deploy **Retrieval-Augmented Generation (RAG)** chatbot.
Upload your documents, ask questions, and get grounded answers with source citations — no paid APIs required.

**Live demo:** https://huggingface.co/spaces/Mobiworks/rag-chatbot

---

## Architecture

```
                          ┌─────────────────────────────────────────┐
                          │             RAG Pipeline                 │
                          └─────────────────────────────────────────┘
  ┌──────────┐   load    ┌──────────────┐  split   ┌──────────────┐
  │ PDF/TXT/ │──────────►│  document_   │─────────►│ text_        │
  │ DOCX/HTML│           │  loader.py   │          │ splitter.py  │
  └──────────┘           └──────────────┘          └──────┬───────┘
                                                          │ chunks
                                                          ▼
                                                   ┌──────────────┐
                                                   │  embedder.py │
                                                   │ all-MiniLM   │
                                                   └──────┬───────┘
                                                          │ vectors
                                                          ▼
                                                   ┌──────────────┐
                                                   │ vector_store │◄─── load / save
                                                   │   (FAISS)    │     data/vector_db/
                                                   └──────────────┘
                                                          ▲
  ┌──────────────┐  query vector   ┌──────────────┐      │ top-K chunks
  │  User Query  │────────────────►│  retriever   │──────┘
  └──────────────┘                 └──────────────┘
         │                                │
         │                  ┌─────────────▼────────────┐
         │                  │      prompt_template      │
         │                  │  context + question → str │
         │                  └─────────────┬────────────┘
         │                                │ formatted prompt
         │                                ▼
         │                  ┌─────────────────────────┐
         │                  │      llm_handler.py      │
         │                  │  Phi-2 Q4 / Mistral-7B  │
         │                  │  (llama-cpp-python GGUF) │
         │                  └─────────────┬───────────┘
         │                                │
         ▼                                ▼
  ┌──────────────────────────────────────────────────────┐
  │               app/main.py  (Streamlit UI)            │
  │  chat history • source citations • doc upload        │
  │  clear chat button • top-k slider                    │
  └──────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py              # Streamlit UI entry point
│   ├── chatbot.py           # Orchestrates retriever + LLM chain
│   └── config.py            # All config constants
├── components/
│   ├── __init__.py
│   ├── document_loader.py   # Load & parse PDF / TXT / DOCX / HTML
│   ├── text_splitter.py     # Chunking with overlap
│   ├── embedder.py          # HuggingFace embedding wrapper
│   ├── vector_store.py      # FAISS create / save / load
│   ├── retriever.py         # Similarity search, top-K logic
│   ├── llm_handler.py       # LLM loading & inference
│   └── prompt_template.py   # RAG prompt construction
├── data/
│   ├── raw/                 # Place your source documents here
│   └── vector_db/           # Persisted FAISS index (auto-created)
├── scripts/
│   ├── ingest.py            # One-time ingestion script
│   └── evaluate.py          # Basic eval: retrieval accuracy + latency
├── tests/
│   ├── test_loader.py
│   ├── test_retriever.py
│   └── test_chatbot.py
├── .streamlit/
│   └── config.toml          # Streamlit server config
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Features

- **Upload documents** — PDF, TXT, DOCX supported
- **Auto-ingestion** — Documents in `data/raw/` are ingested automatically on startup
- **Source citations** — Every answer shows which document chunks were used with similarity scores
- **Clear chat** — Reset the conversation with one click
- **Top-K slider** — Control how many chunks are retrieved per query (1–10)
- **Persistent vector store** — FAISS index saved to disk, no re-embedding on restart

---

## Quick Start (Local)

### 1. Clone & install

```bash
git clone https://github.com/mmubasharmug-18/rag-chatbot.git
cd rag-chatbot
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env — HF_TOKEN is only needed for gated models
```

### 3. Add your documents

Place any PDF, TXT, or DOCX files in `data/raw/`.

### 4. Ingest documents

```bash
python scripts/ingest.py
```

### 5. Launch the app

```bash
streamlit run app/main.py
```

Open http://localhost:7860 in your browser.

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=app --cov=components --cov-report=term-missing
```

---

## Deployment

### HuggingFace Spaces (Free)

```bash
git remote add space https://huggingface.co/spaces/Mobiworks/rag-chatbot
git push space main --force
```

Add secrets under **Settings → Variables and secrets** if needed.

---

## Stack

| Component       | Tool                                    |
|-----------------|-----------------------------------------|
| RAG Framework   | LangChain                               |
| Embedding Model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector Store    | FAISS (local, persisted to disk)        |
| LLM             | Phi-2 Q4_K_M (GGUF)                    |
| LLM Runtime     | llama-cpp-python                        |
| Document Loaders| PyMuPDF, docx2txt, unstructured         |
| UI              | Streamlit                               |
| Deployment      | HuggingFace Spaces / Docker             |
| Cost            | **$0 — 100% free & open-source**        |

---

## License

MIT
