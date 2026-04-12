# project-rose

A Retrieval-Augmented Generation (RAG) system for question-answering over a personal memoir. Combines hybrid search (semantic + BM25), cross-encoder reranking, knowledge graph augmentation, and LLM-powered answer generation.

## Architecture

### Inference pipeline

```
User Question
    ↓
[Hybrid Search]
    ├─ Semantic: cosine similarity over pre-computed embeddings (75%)
    ├─ BM25: keyword scoring over raw text (25%)
    └─ Combined → Top 20 candidates
    ↓
[Cross-Encoder Reranking]
    └─ Re-scores top 20 → keeps top 8 chunks
    ↓
[KG Augmentation]  (if kg.json exists)
    ├─ Entity regex match: find canonical entities in the query
    ├─ If hits: re-rank matched triples by predicate cosine similarity → top 15
    └─ If no hits: top-3 predicates by cosine similarity → top 15 triples
    ↓
[LLM Answer Generation]
    └─ Groq (llama-3.1-8b-instant): chunks context + KG facts → answer
    ↓
Answer + Source References
```

### Training pipeline

```
Source documents
    ↓
python training.py index
    ├─ Chunk by line (min 60 chars)
    ├─ Contextualize each chunk via LLM (Groq, rate-limited)
    ├─ Embed (context + text) with all-MiniLM-L6-v2
    └─ Checkpoint after each chunk → index.json
    ↓
python training.py extract
    ├─ Same chunking as index
    ├─ Extract (subject, predicate, object) triples via LLM per chunk
    └─ Checkpoint after each chunk → kg_raw.json
    ↓
python training.py dedupe
    ├─ Pass A — Predicate dedup: embed predicates with examples,
    │           greedy L2 cluster on unit-normalized vectors (thresh 0.8)
    └─ Pass B — Entity dedup: embed entities with triples + chunk context,
                pooled from both subject and object positions (thresh 0.6)
    └─ → kg.json  (canonical_subject, canonical_predicate, canonical_object)
```

## Files

| File | Description |
|------|-------------|
| `training.py` | Pre-inference pipeline: indexing, KG extraction, KG deduplication |
| `inference.py` | Query pipeline: hybrid search, reranking, KG augmentation, LLM answer |
| `viz.py` | Visualizations (interactive KG graph) |
| `index.json` | Embedding index — one record per chunk with 384-dim vector |
| `kg_raw.json` | Raw extracted triples with extraction progress checkpoint |
| `kg.json` | Deduplicated triples with canonical subject/predicate/object |
| `building_the_api.md` | Guide for wrapping `inference.py` in a FastAPI server |

## Setup

**Requirements:** Python 3.12+, a [Groq API key](https://console.groq.com/)

```bash
cp .env.example .env
# Add GROQ_API_KEY to .env

source .venv/bin/activate
```

## Usage

### Inference

```bash
python inference.py "Your question here"
python inference.py          # interactive prompt
```

KG augmentation is automatic if `kg.json` exists. Remove it to run without.

### Training

Each command checkpoints after every chunk — safe to interrupt and resume:

```bash
python training.py index    # → index.json
python training.py extract  # → kg_raw.json
python training.py dedupe   # → kg.json  (no LLM calls, embeddings only)
```

`index` and `extract` take ~15 min each (429 chunks × 2.1s Groq rate limit).  
`dedupe` is fast — local embeddings only, no API calls.

### Visualization

```bash
python viz.py kg            # → kg_graph.html (full graph)
python viz.py kg --top 50   # top 50 entities by degree
```

Open the output HTML in any browser. The graph is interactive — drag nodes, zoom, hover for connection counts.

## Configuration

**`inference.py`**
```python
TOP_K           = 20     # candidates from hybrid search
RERANK_TOP_K    = 8      # kept after reranking
SEMANTIC_WEIGHT = 0.75   # blend ratio (semantic vs. BM25)
KG_TOP_K        = 15     # max triples injected into context
EMBED_MODEL     = "all-MiniLM-L6-v2"
RERANKER_MODEL  = "BAAI/bge-reranker-base"
LLM_MODEL       = "llama-3.1-8b-instant"
```

**`training.py`**
```python
CONTEXTUALIZE   = True   # generate LLM context per chunk during indexing
MIN_LINE_LEN    = 60     # minimum line length to include as a chunk
PRED_L2_THRESH  = 0.8    # predicate dedup threshold (unit-normalized L2)
OBJ_L2_THRESH   = 0.6    # entity dedup threshold (unit-normalized L2)
```

Dedup threshold reference on unit-normalized vectors:

| L2 threshold | Cosine similarity |
|---|---|
| 0.4 | ≥ 0.92 (near-duplicate) |
| 0.6 | ≥ 0.82 (same entity, surface variation) |
| 0.8 | ≥ 0.68 (same concept, different phrasing) |

## Data formats

**`index.json`** — array of records:
```json
{
  "source": "bio_1.txt",
  "chunk_id": 0,
  "text": "...",
  "context": "LLM-generated situating sentence",
  "embedding": [384 floats]
}
```

**`kg_raw.json`** — checkpointed extraction state:
```json
{
  "processed": [["bio_1.txt", 0], ["bio_1.txt", 1]],
  "triples": [
    {"source": "bio_1.txt", "chunk_id": 0, "text": "...",
     "subject": "Rose Marie", "predicate": "was born in", "object": "Cicero"}
  ]
}
```

`processed` tracks every attempted chunk so chunks that yield 0 triples are not retried on restart.

**`kg.json`** — deduplicated triples (extends `kg_raw.json` triple schema):
```json
{
  "source": "bio_1.txt", "chunk_id": 0, "text": "...",
  "subject": "Rose Marie",  "predicate": "was born in",  "object": "Cicero",
  "canonical_subject": "Rose", "canonical_predicate": "born in", "canonical_object": "Cicero, IL"
}
```

Original fields preserved alongside canonical fields for audit and re-clustering.

## Programmatic usage

```python
from inference import load_index, load_kg, answer, EMBED_MODEL, RERANKER_MODEL
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
import os

records, index_embs = load_index()
model    = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)
client   = Groq(api_key=os.environ["GROQ_API_KEY"])
kg       = load_kg(model)  # None if kg.json doesn't exist

result = answer("Your question", records, index_embs, model, client, reranker, kg=kg)
# {"answer": "...", "sources": ["bio_1.txt", ...]}
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `groq` | Groq API client |
| `sentence-transformers` | Bi-encoder embeddings + CrossEncoder reranking |
| `torch` | Deep learning backend |
| `rank-bm25` | BM25 keyword scoring |
| `numpy` | Vector math |
| `networkx` | Graph construction for KG visualization |
| `pyvis` | Interactive HTML graph rendering |
| `python-dotenv` | `.env` loading |
| `tqdm` | Progress bars |
