"""
serve.py — FastAPI service wrapping the Rose's Memories RAG pipeline.
"""

import os
from contextlib import asynccontextmanager

import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import EMBED_MODEL, KG_GRAPH_TOP, RERANKER_MODEL
from src.inference import answer, load_index, load_kg

state = {}


@asynccontextmanager
async def lifespan(app):
    state["records"], state["index_embs"] = load_index()
    state["model"]   = SentenceTransformer(EMBED_MODEL)
    state["reranker"] = CrossEncoder(RERANKER_MODEL)
    state["groq"]    = Groq(api_key=os.environ["GROQ_API_KEY"])
    state["kg"]      = load_kg(state["model"])
    print(f"  {len(state['records'])} chunks ready")

    state["kg_graph"] = None
    if state["kg"]:
        print(f"  {len(state['kg']['triples'])} KG triples ready")
        state["kg_graph"] = _build_kg_graph(state["kg"]["triples"])

    yield
    state.clear()


def _build_kg_graph(triples: list[dict]) -> dict:
    """Precompute graph layout for the /kg endpoint."""
    G = nx.DiGraph()
    seen: set[tuple] = set()
    for t in triples:
        s, p, o = t["canonical_subject"], t["predicate"], t["canonical_object"]
        key = (s, p, o)
        if key in seen:
            continue
        seen.add(key)
        if G.has_edge(s, o):
            existing = G[s][o]["label"]
            if p not in existing.split(" / "):
                G[s][o]["label"] = existing + " / " + p
        else:
            G.add_edge(s, o, label=p)

    if len(G.nodes()) > KG_GRAPH_TOP:
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:KG_GRAPH_TOP]
        G = G.subgraph(set(top_nodes)).copy()

    pos = nx.spring_layout(G, seed=42, k=2.0, iterations=80)
    xs = [v[0] for v in pos.values()]
    ys = [v[1] for v in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_rng = (x_max - x_min) or 1.0
    y_rng = (y_max - y_min) or 1.0
    max_deg = max((G.degree(n) for n in G.nodes()), default=1)

    return {
        "nodes": [
            {
                "id":     n,
                "x":      (pos[n][0] - x_min) / x_rng,
                "y":      (pos[n][1] - y_min) / y_rng,
                "degree": G.degree(n),
                "size":   3 + (G.degree(n) / max_deg) * 9,
            }
            for n in G.nodes()
        ],
        "edges": [
            {"source": s, "target": t, "label": d["label"]}
            for s, t, d in G.edges(data=True)
        ],
    }


app = FastAPI(title="Rose's Memories", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer:     str
    sources:    list[str]
    kg_triples: list[dict] = []
    log:        list[str]  = []
    candidates: list[dict] = []
    rate_limit: dict       = {}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(state["records"]),
        "kg":     state["kg"] is not None,
    }


@app.get("/kg")
def kg():
    if not state.get("kg_graph"):
        raise HTTPException(status_code=404, detail="KG not loaded")
    return state["kg_graph"]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = answer(
        req.question,
        state["records"],
        state["index_embs"],
        state["model"],
        state["groq"],
        state["reranker"],
        kg=state["kg"],
    )
    return QueryResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
