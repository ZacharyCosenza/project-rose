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
    └─ Combined → Top 25 candidates
    ↓
[Cross-Encoder Reranking]
    └─ Re-scores top 25 → keeps top 12 chunks
    ↓
[KG Augmentation]  (if kg.json exists)
    ├─ Entity regex match: find canonical entities in the query
    ├─ If hits: re-rank matched triples by predicate cosine similarity,
    │           deduplicate (s, p, o) → top 15 unique triples
    └─ If no hits: top-3 predicates with cosine similarity ≥ 0.5,
                  deduplicate → top 15 unique triples (skipped if no
                  predicate clears the threshold)
    ↓
[LLM Answer Generation]
    └─ Groq (llama-3.3-70b-versatile, fallback llama-3.1-8b-instant)
    ↓
Answer + Source References
```

### Training pipeline

```
Source documents (data/01_raw/)
    ↓
python src/training.py index
    ├─ Chunk by line (min 60 chars)
    ├─ Contextualize each chunk via LLM (Groq, rate-limited)
    ├─ Embed (context + text) with all-MiniLM-L6-v2
    └─ Checkpoint after each chunk → data/02_extracted/index.json
    ↓
python src/training.py extract
    ├─ Same chunking as index
    ├─ LLM extracts (subject, subject_type, predicate, object, object_type) per chunk
    ├─ Hard-filter: predicate must be in ALLOWED_PREDICATES (26, fixed allowlist)
    ├─ Hard-filter: subject/object types must match predicate type rules
    ├─ Hard-filter: subject ≠ object
    └─ Checkpoint after each chunk → data/02_extracted/kg_raw.json
    ↓
python src/training.py dedupe
    ├─ Subject entity dedup: embed as "Entity + predicate signature",
    │   greedy L2 cluster on unit-normalized vectors (thresh 0.6)
    ├─ Object entity dedup: same algorithm, separate pool from subjects
    │   (prevents person names from absorbing location/organization strings)
    └─ → data/02_extracted/kg.json  (canonical_subject, canonical_object)
```

## Project layout

```
src/
    inference.py    Query pipeline: hybrid search, reranking, KG augmentation, LLM answer
    training.py     Pre-inference pipeline: indexing, KG extraction, entity deduplication
    eval.py         Evaluation framework: 30 LLM-judged questions across easy/medium/hard tiers
    viz.py          Visualizations (static KG graph rendered to HTML)
data/
    01_raw/         Source .txt documents (gitignored)
    02_extracted/   Generated artifacts: index.json, kg_raw.json, kg.json (gitignored)
conf/               System prompts and eval questions (gitignored)
eval_results/       Timestamped eval runs (gitignored)
```

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
python src/inference.py "Your question here"
python src/inference.py          # interactive prompt
```

KG augmentation is automatic if `data/02_extracted/kg.json` exists. Use `--no-kg` to disable.

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--no-kg` | — | Disable KG augmentation |
| `--no-rerank` | — | Disable cross-encoder reranking |
| `--no-bm25` | — | Pure semantic search |
| `--top-k N` | 25 | Hybrid search candidates |
| `--rerank-top-k N` | 12 | Chunks kept after reranking |
| `--kg-top-k N` | 15 | Max KG triples injected |
| `--semantic-weight F` | 0.60 | Semantic blend ratio (0–1) |
| `--max-tokens N` | 500 | LLM max completion tokens |
| `--temperature F` | 0 | LLM temperature |
| `--json` | — | Output result as JSON |

### Training

Each command checkpoints after every chunk — safe to interrupt and resume:

```bash
python src/training.py index    # → data/02_extracted/index.json
python src/training.py extract  # → data/02_extracted/kg_raw.json
python src/training.py dedupe   # → data/02_extracted/kg.json  (no LLM calls, embeddings only)
```

`index` and `extract` take ~15 min each (chunks × 2.1s Groq rate limit).  
`dedupe` is fast — local embeddings only, no API calls.

### Evaluation

```bash
python src/eval.py                      # run all 30 questions
python src/eval.py --difficulty easy    # run one tier only
python src/eval.py --no-kg              # disable KG augmentation
```

All inference flags (`--no-kg`, `--no-rerank`, `--top-k`, etc.) are forwarded to `inference.py`.

### Visualization

```bash
python src/viz.py kg            # → kg_graph.html (full graph)
python src/viz.py kg --top 50   # top 50 entities by degree
```

Open the output HTML in any browser. The layout is static (positions fixed at render time). Hover nodes for connection counts and predicates; use navigation buttons or scroll to zoom.

## Configuration

**`src/inference.py`**
```python
TOP_K                  = 25                      # candidates from hybrid search
RERANK_TOP_K           = 12                      # kept after reranking
SEMANTIC_WEIGHT        = 0.60                    # blend ratio (semantic vs. BM25)
KG_TOP_K               = 15                      # max unique triples injected into context
KG_FALLBACK_THRESHOLD  = 0.5                     # min predicate cosine sim for fallback KG search
EMBED_MODEL            = "all-MiniLM-L6-v2"
RERANKER_MODEL         = "BAAI/bge-reranker-base"
LLM_MODEL              = "llama-3.3-70b-versatile"
LLM_MODEL_FALLBACK     = "llama-3.1-8b-instant"  # used if primary model call fails
```

**`src/training.py`**
```python
CONTEXTUALIZE   = True                    # generate LLM context per chunk during indexing
MIN_LINE_LEN    = 60                      # minimum line length to include as a chunk
OBJ_L2_THRESH   = 0.6                    # entity dedup threshold (unit-normalized L2),
                                          # applied independently to subject and object pools
EXTRACT_MODEL   = "llama-3.3-70b-versatile"
```

Dedup threshold reference on unit-normalized vectors:

| L2 threshold | Cosine similarity |
|---|---|
| 0.4 | ≥ 0.92 (near-duplicate) |
| 0.6 | ≥ 0.82 (same entity, surface variation) |
| 0.8 | ≥ 0.68 (same concept, different phrasing) |

## KG extraction quality

Triples pass three post-extraction filters before being saved:

1. **Predicate allowlist** — predicate must match one of 26 canonical forms verbatim.
2. **Type compatibility** — the LLM tags each entity with `subject_type` / `object_type` (`person`, `place`, `organization`, `thing`). Each predicate has fixed type rules (e.g. kinship predicates require `person → person`; location predicates require `person → place`). Mismatches are rejected.
3. **Self-reference** — triples where `subject == object` are discarded.

Entity deduplication uses predicate-signature embeddings (`Entity: X\nPredicates: p1, p2, ...`) rather than chunk text, preventing co-occurrence noise from conflating semantically unrelated entities.

## Data formats

**`data/02_extracted/index.json`** — array of records:
```json
{
  "source": "bio_1.txt",
  "chunk_id": 0,
  "text": "...",
  "context": "LLM-generated situating sentence",
  "embedding": [384 floats]
}
```

**`data/02_extracted/kg_raw.json`** — checkpointed extraction state:
```json
{
  "processed": [["bio_1.txt", 0], ["bio_1.txt", 1]],
  "triples": [
    {
      "source": "bio_1.txt", "chunk_id": 0, "text": "...",
      "subject": "Rose Marie", "subject_type": "person",
      "predicate": "born in",
      "object": "Cicero", "object_type": "place"
    }
  ]
}
```

`processed` tracks every attempted chunk so chunks that yield 0 triples are not retried on restart.

**`data/02_extracted/kg.json`** — deduplicated triples (extends `kg_raw.json` triple schema):
```json
{
  "source": "bio_1.txt", "chunk_id": 0, "text": "...",
  "subject": "Rose Marie", "subject_type": "person",
  "predicate": "born in",
  "object": "Cicero", "object_type": "place",
  "canonical_subject": "Rose", "canonical_object": "Cicero, IL"
}
```

Original `subject` and `object` fields are preserved alongside their canonical forms for audit and re-clustering.

## Programmatic usage

```python
from src.inference import load_index, load_kg, answer, EMBED_MODEL, RERANKER_MODEL
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
import os

records, index_embs = load_index()
model    = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)
client   = Groq(api_key=os.environ["GROQ_API_KEY"])
kg       = load_kg(model)  # None if kg.json doesn't exist

result = answer("Your question", records, index_embs, model, client, reranker, kg=kg)
# {"answer": "...", "sources": ["bio_1.txt", ...], "context": "full context sent to LLM"}
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `groq` | Groq API client |
| `sentence-transformers` | Bi-encoder embeddings + CrossEncoder reranking |
| `torch` | Deep learning backend |
| `rank-bm25` | BM25 keyword scoring |
| `numpy` | Vector math |
| `networkx` | Graph construction and layout for KG visualization |
| `pyvis` | HTML graph rendering |
| `python-dotenv` | `.env` loading |
| `tqdm` | Progress bars |
