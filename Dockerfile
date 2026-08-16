# ── Base image ─────────────────────────────────────────────────────────────────
# Python 3.11 slim keeps the image small while being recent enough for all deps.
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
# cmake + build-essential are required to compile llama-cpp-python from source
# if a pre-built wheel is not available for the target platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY . .

# Create data directories (they may be empty in the repo)
RUN mkdir -p data/raw data/vector_db data/models

# ── Expose Streamlit port ──────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# ── Entrypoint ─────────────────────────────────────────────────────────────────
CMD ["sh", "-c", "python scripts/ingest.py 2>/dev/null || true && streamlit run app/main.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true --server.maxUploadSize=200 --server.enableXsrfProtection=false"]
