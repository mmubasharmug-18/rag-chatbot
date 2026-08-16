import gradio as gr
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chatbot import Chatbot
from app.config import VECTOR_DB_PATH
from components.document_loader import load_document
from components.embedder import HuggingFaceEmbedder
from components.text_splitter import split_documents
from components.vector_store import VectorStore

embedder = HuggingFaceEmbedder()
store    = VectorStore(embedder=embedder, index_path=VECTOR_DB_PATH)
store.load()
chatbot  = Chatbot(vector_store=store)

def ingest(files):
    all_chunks = []
    for f in files:
        docs   = load_document(f.name)
        chunks = split_documents(docs)
        all_chunks.extend(chunks)
    store.build(all_chunks)
    return f"✅ Ingested {len(all_chunks)} chunks from {len(files)} file(s)!"

def chat(message, history):
    response = chatbot.chat(message)
    return response.answer

with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown("# 🤖 RAG Chatbot")
    gr.Markdown("Upload documents then ask questions!")
    with gr.Row():
        with gr.Column(scale=1):
            files  = gr.File(label="Upload Documents", file_count="multiple")
            ingest_btn = gr.Button("⚙️ Ingest Documents", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            ingest_btn.click(ingest, inputs=files, outputs=status)
        with gr.Column(scale=3):
            gr.ChatInterface(chat)

demo.launch(server_name="0.0.0.0", server_port=7860)