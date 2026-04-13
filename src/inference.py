"""
inference.py — hybrid search (BM25 + semantic) then answer with Groq.

Hybrid search combines:
  - Cosine similarity on embeddings (semantic match)
  - BM25 keyword scoring (exact name/term match)
Both scores are normalized to [0,1] then weighted-summed.

Optionally augments context with knowledge graph facts from kg.json.
KG retrieval: entity regex match → predicate semantic re-rank (intersection).
Fallback when no entities match: top-k predicate semantic search across all triples.

CLI usage:
    python src/inference.py "What were Rose's grandparents like?"
    python src/inference.py          # prompts for input
    python src/inference.py "..." --no-kg --no-rerank --no-bm25
    python src/inference.py "..." --top-k 30 --rerank-top-k 15 --kg-top-k 20
    python src/inference.py "..." --json   # outputs result as JSON to stdout
"""

import argparse
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

ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH  = ROOT / "data" / "02_extracted" / "index.json"
KG_PATH     = ROOT / "data" / "02_extracted" / "kg.json"
CONF_DIR    = ROOT / "conf"

EMBED_MODEL        = "all-MiniLM-L6-v2"
RERANKER_MODEL     = "BAAI/bge-reranker-base"
LLM_MODEL          = "llama-3.3-70b-versatile"
LLM_MODEL_FALLBACK = "llama-3.1-8b-instant"
GROQ_API_KEY       = os.environ["GROQ_API_KEY"]

# Defaults — all overridable via CLI flags
TOP_K            = 25
RERANK_TOP_K     = 12
SEMANTIC_WEIGHT  = 0.60   # 0 = pure BM25, 1 = pure semantic
KG_TOP_K         = 15
KG_FALLBACK_THRESHOLD = 0.5
DEFAULT_MAX_TOKENS    = 500
DEFAULT_TEMPERATURE   = 0

SYSTEM_PROMPT = (CONF_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def load_index() -> tuple[list[dict], np.ndarray]:
    records = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    embeddings = np.array([r["embedding"] for r in records], dtype=np.float32)
    return records, embeddings


def hybrid_search(
    query: str,
    records: list[dict],
    index_embs: np.ndarray,
    model,
    top_k: int = TOP_K,
    semantic_weight: float = SEMANTIC_WEIGHT,
    use_bm25: bool = True,
) -> list[int]:
    # --- Semantic: cosine similarity on embeddings ---
    query_emb = model.encode(query)
    q = query_emb / np.linalg.norm(query_emb)
    normed = index_embs / np.linalg.norm(index_embs, axis=1, keepdims=True)
    cosine_scores = normed @ q

    if use_bm25:
        # --- BM25: keyword match on raw text ---
        tokenized_corpus = [r["text"].lower().split() for r in records]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = np.array(bm25.get_scores(query.lower().split()), dtype=np.float32)

        def normalize(scores):
            lo, hi = scores.min(), scores.max()
            return (scores - lo) / (hi - lo + 1e-8)

        combined = semantic_weight * normalize(cosine_scores) + (1 - semantic_weight) * normalize(bm25_scores)
    else:
        combined = cosine_scores

    return np.argsort(combined)[::-1][:top_k].tolist()


def rerank(query: str, candidates: list[dict], reranker: CrossEncoder, top_k: int = RERANK_TOP_K) -> list[dict]:
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


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
    canon_preds = list({t["predicate"] for t in triples})
    pred_embs = model.encode(canon_preds).astype(np.float32)
    norms = np.linalg.norm(pred_embs, axis=1, keepdims=True)
    pred_embs = pred_embs / np.where(norms > 0, norms, 1.0)

    return {
        "triples":     triples,
        "entities":    entities,
        "canon_preds": canon_preds,
        "pred_embs":   pred_embs,
    }


def kg_search(
    query: str,
    kg: dict,
    model: SentenceTransformer,
    kg_top_k: int = KG_TOP_K,
) -> list[dict]:
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

    def _dedup(ts: list[dict]) -> list[dict]:
        seen: set[tuple] = set()
        out = []
        for t in ts:
            key = (t["canonical_subject"], t["predicate"], t["canonical_object"])
            if key not in seen:
                seen.add(key)
                out.append(t)
        return out

    if entity_triples:
        # Intersection: re-rank entity-matched triples by predicate relevance
        ranked = sorted(
            entity_triples,
            key=lambda t: pred_sim_map.get(t["predicate"], 0.0),
            reverse=True,
        )
        return _dedup(ranked)[:kg_top_k]
    else:
        # Fallback: triples from predicates with cosine similarity > threshold to query
        top_pred_idxs = np.argsort(pred_sims)[::-1][:3]
        top_preds = {
            canon_preds[i] for i in top_pred_idxs
            if pred_sims[i] >= KG_FALLBACK_THRESHOLD
        }
        if not top_preds:
            return []
        candidates = [t for t in triples if t["predicate"] in top_preds]
        return _dedup(candidates)[:kg_top_k]


def format_kg_context(triples: list[dict]) -> str:
    lines = ["Knowledge graph facts:"]
    for t in triples:
        lines.append(f"- {t['canonical_subject']} | {t['predicate']} | {t['canonical_object']}")
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
    reranker: CrossEncoder | None = None,
    kg: dict | None = None,
    *,
    top_k: int = TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    kg_top_k: int = KG_TOP_K,
    semantic_weight: float = SEMANTIC_WEIGHT,
    use_bm25: bool = True,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """
    Run hybrid search and return the LLM answer plus source chunks.
    Accepts pre-loaded resources so callers can initialize once and reuse.
    Pass kg=load_kg(model) to enable knowledge graph augmentation.
    Pass reranker=None to skip reranking.
    """
    top_indices = hybrid_search(query, records, index_embs, model,
                                top_k=top_k, semantic_weight=semantic_weight, use_bm25=use_bm25)
    candidates = [records[i] for i in top_indices]

    if reranker is not None:
        top_chunks = rerank(query, candidates, reranker, top_k=rerank_top_k)
    else:
        top_chunks = candidates[:rerank_top_k]

    context = "\n\n".join(c["text"] for c in top_chunks)

    if kg:
        kg_triples = kg_search(query, kg, model, kg_top_k=kg_top_k)
        if kg_triples:
            context = context + "\n\n" + format_kg_context(kg_triples)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    response = None
    for model_id in (LLM_MODEL, LLM_MODEL_FALLBACK):
        try:
            response = groq_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            break
        except Exception as exc:
            if model_id == LLM_MODEL:
                print(f"  [WARN] {model_id} failed ({exc}), retrying with {LLM_MODEL_FALLBACK}",
                      file=sys.stderr)
            else:
                raise

    return {
        "answer":  response.choices[0].message.content,
        "sources": list({c["source"] for c in top_chunks}),
        "context": context,
    }


def main():
    parser = argparse.ArgumentParser(
        prog="inference.py",
        description="project-rose RAG inference",
    )
    parser.add_argument("question", nargs="?", default=None,
                        help="Question to answer (omit for interactive prompt)")
    parser.add_argument("--no-kg",      action="store_true", help="Disable KG augmentation")
    parser.add_argument("--no-rerank",  action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--no-bm25",    action="store_true", help="Disable BM25 (pure semantic search)")
    parser.add_argument("--top-k",      type=int,   default=TOP_K,           help=f"Hybrid search candidates (default {TOP_K})")
    parser.add_argument("--rerank-top-k", type=int, default=RERANK_TOP_K,    help=f"Chunks kept after reranking (default {RERANK_TOP_K})")
    parser.add_argument("--kg-top-k",   type=int,   default=KG_TOP_K,        help=f"Max KG triples injected (default {KG_TOP_K})")
    parser.add_argument("--semantic-weight", type=float, default=SEMANTIC_WEIGHT, help=f"Semantic blend ratio 0-1 (default {SEMANTIC_WEIGHT})")
    parser.add_argument("--max-tokens", type=int,   default=DEFAULT_MAX_TOKENS, help=f"LLM max completion tokens (default {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"LLM temperature (default {DEFAULT_TEMPERATURE})")
    parser.add_argument("--json",       action="store_true", help="Output result as JSON to stdout")
    args = parser.parse_args()

    query = args.question or input("Question: ")

    records, index_embs = load_index()
    model    = SentenceTransformer(EMBED_MODEL)
    reranker = None if args.no_rerank else CrossEncoder(RERANKER_MODEL)
    client   = Groq(api_key=GROQ_API_KEY)
    kg       = None if args.no_kg else load_kg(model)

    result = answer(
        query, records, index_embs, model, client, reranker, kg=kg,
        top_k=args.top_k,
        rerank_top_k=args.rerank_top_k,
        kg_top_k=args.kg_top_k,
        semantic_weight=args.semantic_weight,
        use_bm25=not args.no_bm25,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["answer"])


if __name__ == "__main__":
    main()
