"""
index.py — chunk, embed, and save the knowledge base to disk.

With CONTEXTUALIZE=True, calls Groq once per chunk to generate a brief
situating sentence, then embeds (context + text) together. This is
"contextual retrieval" — the embedding carries more signal than raw text alone.

Usage:
    python index.py
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

DATA_DIR = Path("data")
OUT = Path("index.json")
EXT = "txt"
EMBED_MODEL = "all-MiniLM-L6-v2"
MIN_LINE_LEN = 60
CONTEXTUALIZE = True
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
CONTEXT_MODEL = "llama-3.1-8b-instant"
RATE_LIMIT_SLEEP = 2.1  # seconds; keeps us under Groq's 30 req/min free tier

CONTEXT_SYSTEM = (
    "You are indexing a personal memoir written by Rose (born 1936, Cicero IL). "
    "Given a passage, write 1-2 sentences situating it in her life story. "
    "Include time period, people mentioned, and topic. Be concise."
)


def load_chunks(data_dir: Path, ext: str) -> list[dict]:
    chunks = []
    for path in sorted(data_dir.glob(f"*.{ext}")):
        text = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines()]
        passages = [l for l in lines if len(l) > MIN_LINE_LEN]
        for i, passage in enumerate(passages):
            chunks.append({"source": path.name, "chunk_id": i, "text": passage})
    return chunks


def generate_context(chunk_text: str, client: Groq) -> str:
    response = client.chat.completions.create(
        model=CONTEXT_MODEL,
        messages=[
            {"role": "system", "content": CONTEXT_SYSTEM},
            {"role": "user", "content": chunk_text},
        ],
        temperature=0,
        max_completion_tokens=80,
    )
    return response.choices[0].message.content.strip()


def main():
    print(f"Loading chunks from {DATA_DIR}/*.{EXT} ...")
    chunks = load_chunks(DATA_DIR, EXT)
    print(f"  {len(chunks)} chunks found")

    if CONTEXTUALIZE:
        print(f"\nGenerating context with '{CONTEXT_MODEL}' (~{len(chunks) * RATE_LIMIT_SLEEP / 60:.0f} min) ...")
        client = Groq(api_key=GROQ_API_KEY)
        for chunk in tqdm(chunks):
            chunk["context"] = generate_context(chunk["text"], client)
            time.sleep(RATE_LIMIT_SLEEP)
    else:
        for chunk in chunks:
            chunk["context"] = ""

    # Embed context + text together so the vector carries both signals
    print(f"\nEmbedding with '{EMBED_MODEL}' ...")
    model = SentenceTransformer(EMBED_MODEL)
    texts_to_embed = [
        f"{c['context']}\n{c['text']}".strip() if c["context"] else c["text"]
        for c in chunks
    ]
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    print(f"  Embedding shape: {embeddings.shape}")

    records = [
        {**chunk, "embedding": emb.tolist()}
        for chunk, emb in zip(chunks, embeddings)
    ]

    OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nIndex saved → {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
