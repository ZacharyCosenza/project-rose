FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first — prevents sentence-transformers from pulling the full CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies (separate layer — Docker caches this unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake ML models into the image (eliminates cold-start download on every deploy)
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('BAAI/bge-reranker-base')"

# Copy application code
# data/ and conf/ are NOT copied — they come from bind mounts at runtime
COPY config.py serve.py app.py ./
COPY src/ src/

# Directory stubs so the app doesn't error before volumes are attached
RUN mkdir -p data/02_extracted conf
