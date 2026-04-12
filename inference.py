"""
rag.py — hybrid search (BM25 + semantic) then answer with Groq.

Hybrid search combines:
  - Cosine similarity on embeddings (semantic match)
  - BM25 keyword scoring (exact name/term match)
Both scores are normalized to [0,1] then weighted-summed.

Optionally augments context with knowledge graph facts from kg.json.
KG retrieval: entity regex match → predicate semantic re-rank (intersection).
Fallback when no entities match: top-k predicate semantic search across all triples.

CLI usage:
    python inference.py "What were Rose's grandparents like?"
    python inference.py          # prompts for input
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

INDEX_PATH  = Path("index.json")
KG_PATH     = Path("kg.json")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "llama-3.1-8b-instant"
GROQ_API_KEY   = os.environ["GROQ_API_KEY"]
TOP_K          = 15          # candidates retrieved by hybrid search
RERANK_TOP_K   = 8           # kept after reranking, sent to LLM
RERANKER_MODEL = "BAAI/bge-reranker-base"
SEMANTIC_WEIGHT = 0.75       # 0 = pure BM25, 1 = pure semantic
KG_TOP_K       = 15          # max triples injected into context

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about Rose's life memories. "
    "Answer using only the provided context. If the answer isn't there, say so."
)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------

def load_kg(model: SentenceTransformer) -> dict | None:
    """
    Load kg.json and pre-compute normalized predicate embeddings.
    Returns None if kg.json doesn't exist — the pipeline degrades gracefully.
    """
    if not KG_PATH.exists():
        return None

    triples = json.loads(KG_PATH.read_text(encoding="utf-8"))

    entities = {t["canonical_subject"] for t in triples} | {t["canonical_object"] for t in triples}

    # Embed and normalize canonical predicates — small set, fast
    canon_preds = list({t["canonical_predicate"] for t in triples})
    pred_embs = model.encode(canon_preds).astype(np.float32)
    norms = np.linalg.norm(pred_embs, axis=1, keepdims=True)
    pred_embs = pred_embs / np.where(norms > 0, norms, 1.0)

    return {
        "triples":     triples,
        "entities":    entities,
        "canon_preds": canon_preds,
        "pred_embs":   pred_embs,
    }


def kg_search(query: str, kg: dict, model: SentenceTransformer) -> list[dict]:
    """
    Entity regex match → predicate semantic re-rank (intersection).
    Falls back to predicate-only search when no entities are found in the query.
    """
    triples    = kg["triples"]
    entities   = kg["entities"]
    canon_preds = kg["canon_preds"]
    pred_embs  = kg["pred_embs"]

    # 1. Entity regex: find canonical entities mentioned in the query
    matched_entities = {
        ent for ent in entities
        if re.search(r"\b" + re.escape(ent) + r"\b", query, re.IGNORECASE)
    }
    entity_triples = [
        t for t in triples
        if t["canonical_subject"] in matched_entities
        or t["canonical_object"] in matched_entities
    ] if matched_entities else []

    # 2. Predicate semantic similarity to query
    q_emb = model.encode(query).astype(np.float32)
    q_emb = q_emb / np.linalg.norm(q_emb)
    pred_sims = pred_embs @ q_emb  # cosine similarity, shape (n_preds,)
    pred_sim_map = dict(zip(canon_preds, pred_sims.tolist()))

    if entity_triples:
        # Intersection: re-rank entity-matched triples by predicate relevance
        return sorted(
            entity_triples,
            key=lambda t: pred_sim_map.get(t["canonical_predicate"], 0.0),
            reverse=True,
        )[:KG_TOP_K]
    else:
        # Fallback: triples from the top-3 most query-relevant predicates
        top_pred_idxs = np.argsort(pred_sims)[::-1][:3]
        top_preds = {canon_preds[i] for i in top_pred_idxs}
        return [t for t in triples if t["canonical_predicate"] in top_preds][:KG_TOP_K]


def format_kg_context(triples: list[dict]) -> str:
    lines = ["Knowledge graph facts:"]
    for t in triples:
        lines.append(f"- {t['canonical_subject']} | {t['canonical_predicate']} | {t['canonical_object']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

def answer(
    query: str,
    records: list[dict],
    index_embs: np.ndarray,
    model,
    groq_client: Groq,
    reranker: CrossEncoder,
    kg: dict | None = None,
) -> dict:
    """
    Run hybrid search and return the LLM answer plus source chunks.
    Accepts pre-loaded resources so callers can initialize once and reuse.
    Pass kg=load_kg(model) to enable knowledge graph augmentation.
    """
    top_indices = hybrid_search(query, records, index_embs, model)
    candidates  = [records[i] for i in top_indices]
    top_chunks  = rerank(query, candidates, reranker)
    context     = "\n\n".join(c["text"] for c in top_chunks)

    if kg:
        kg_triples = kg_search(query, kg, model)
        if kg_triples:
            context = context + "\n\n" + format_kg_context(kg_triples)

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
        "answer":  response.choices[0].message.content,
        "sources": list({c["source"] for c in top_chunks}),
    }


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    records, index_embs = load_index()
    model   = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    client  = Groq(api_key=GROQ_API_KEY)
    kg      = load_kg(model)
    result  = answer(query, records, index_embs, model, client, reranker, kg=kg)
    print(result["answer"])


if __name__ == "__main__":
    main()
