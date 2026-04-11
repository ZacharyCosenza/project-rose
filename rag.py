"""
rag.py — hybrid search (BM25 + semantic) then answer with Groq.

Hybrid search combines:
  - Cosine similarity on embeddings (semantic match)
  - BM25 keyword scoring (exact name/term match)
Both scores are normalized to [0,1] then weighted-summed.

CLI usage:
    python rag.py "What were Rose's grandparents like?"
    python rag.py          # prompts for input
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

INDEX_PATH = Path("index.json")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
TOP_K = 20           # candidates retrieved by hybrid search
RERANK_TOP_K = 8     # kept after reranking, sent to LLM
RERANKER_MODEL = "BAAI/bge-reranker-base"
SEMANTIC_WEIGHT = 0.20  # 0 = pure BM25, 1 = pure semantic

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about Rose's life memories. "
    "Answer using only the provided context. If the answer isn't there, say so."
)


def load_index() -> tuple[list[dict], np.ndarray]:
    records = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    embeddings = np.array([r["embedding"] for r in records], dtype=np.float32)
    return records, embeddings


def hybrid_search(query: str, records: list[dict], index_embs: np.ndarray, model) -> list[int]:
    # --- Semantic: cosine similarity on embeddings ---
    query_emb = model.encode(query)
    q = query_emb / np.linalg.norm(query_emb)
    normed = index_embs / np.linalg.norm(index_embs, axis=1, keepdims=True)
    cosine_scores = normed @ q

    # --- BM25: keyword match on raw text ---
    tokenized_corpus = [r["text"].lower().split() for r in records]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = np.array(bm25.get_scores(query.lower().split()), dtype=np.float32)

    # --- Normalize both to [0, 1] then combine ---
    def normalize(scores):
        lo, hi = scores.min(), scores.max()
        return (scores - lo) / (hi - lo + 1e-8)

    combined = SEMANTIC_WEIGHT * normalize(cosine_scores) + (1 - SEMANTIC_WEIGHT) * normalize(bm25_scores)
    return np.argsort(combined)[::-1][:TOP_K].tolist()


def rerank(query: str, candidates: list[dict], reranker: CrossEncoder) -> list[dict]:
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:RERANK_TOP_K]]


def answer(query: str, records: list[dict], index_embs: np.ndarray, model, groq_client: Groq, reranker: CrossEncoder) -> dict:
    """
    Run hybrid search and return the LLM answer plus source chunks.
    Accepts pre-loaded resources so callers can initialize once and reuse.
    """
    top_indices = hybrid_search(query, records, index_embs, model)
    candidates = [records[i] for i in top_indices]
    top_chunks = rerank(query, candidates, reranker)
    context = "\n\n".join(c["text"] for c in top_chunks)

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.3,
        max_completion_tokens=150,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": list({c["source"] for c in top_chunks}),
    }


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    records, index_embs = load_index()
    model = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    client = Groq(api_key=GROQ_API_KEY)
    result = answer(query, records, index_embs, model, client, reranker)
    print(result["answer"])


if __name__ == "__main__":
    main()
