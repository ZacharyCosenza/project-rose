"""
config.py — central parameter store for project-rose.

Import from any module:
    from config import EMBED_MODEL, TOP_K, ...
"""

import os

# ── Models ────────────────────────────────────────────────────────────────────

EMBED_MODEL        = "all-MiniLM-L6-v2"
RERANKER_MODEL     = "BAAI/bge-reranker-base"
LLM_MODEL          = "llama-3.3-70b-versatile"
LLM_MODEL_FALLBACK = "llama-3.1-8b-instant"
CONTEXT_MODEL      = "llama-3.1-8b-instant"
EXTRACT_MODEL      = "llama-3.3-70b-versatile"
EVAL_MODEL         = "llama-3.3-70b-versatile"

# ── Retrieval ─────────────────────────────────────────────────────────────────

TOP_K                 = 25
RERANK_TOP_K          = 12
SEMANTIC_WEIGHT       = 0.60   # 0 = pure BM25, 1 = pure semantic
KG_TOP_K              = 15
KG_FALLBACK_THRESHOLD = 0.5

# ── LLM generation ────────────────────────────────────────────────────────────

DEFAULT_MAX_TOKENS  = 500
DEFAULT_TEMPERATURE = 0

# ── Training ──────────────────────────────────────────────────────────────────

MIN_LINE_LEN     = 60
CONTEXTUALIZE    = True
RATE_LIMIT_SLEEP = 2.1   # seconds; keeps under Groq's 30 req/min free tier
OBJ_L2_THRESH    = 0.6   # entity dedup L2 threshold (unit-normalized ≈ cosine ≥ 0.82)
MAX_EXAMPLES     = 5

# ── Serve ─────────────────────────────────────────────────────────────────────

KG_GRAPH_TOP = 200   # max nodes in the /kg visualization graph

# ── Eval ──────────────────────────────────────────────────────────────────────

EVAL_SLEEP = 4.5   # seconds between questions; each question makes 2 Groq calls

# ── App ───────────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")
