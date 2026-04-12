"""
training.py — pre-inference pipeline for project-rose.

Subcommands:
    python training.py index    Chunk, contextualize, embed → index.json
    python training.py extract  Extract KG triples via LLM → kg_raw.json
    python training.py dedupe   Canonicalize predicates + entities → kg.json

index / extract both checkpoint after every chunk: if interrupted, rerunning
the same command resumes from where it left off.

index:
    With CONTEXTUALIZE=True, calls Groq once per chunk to generate a brief
    situating sentence, then embeds (context + text) together. This is
    "contextual retrieval" — the embedding carries more signal than raw text alone.

extract:
    Calls Groq once per chunk to extract (subject, predicate, object) triples
    as structured JSON. Rate-limited to stay under Groq's free tier.
    kg_raw.json schema: {"processed": [[source, chunk_id], ...], "triples": [...]}
    "processed" tracks every attempted chunk so chunks that yield 0 triples are
    not retried on restart.

dedupe:
    Two-pass embedding-based canonicalization — no LLM calls required.
    Pass A: deduplicate predicates (relationship types).
    Pass B: deduplicate entities pooled from both subject and object positions.
    Uses greedy L2 clustering: most-frequent form becomes the canonical label.
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

# --- index constants ---
DATA_DIR         = Path("data")
OUT              = Path("index.json")
EXT              = "txt"
EMBED_MODEL      = "all-MiniLM-L6-v2"
MIN_LINE_LEN     = 60
CONTEXTUALIZE    = True
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
CONTEXT_MODEL    = "llama-3.1-8b-instant"
RATE_LIMIT_SLEEP = 2.1  # seconds; keeps us under Groq's 30 req/min free tier

# --- KG constants ---
KG_RAW         = Path("kg_raw.json")
KG_OUT         = Path("kg.json")
EXTRACT_MODEL  = "llama-3.1-8b-instant"
PRED_L2_THRESH = 0.8   # on unit-normalized embeddings; ≈ cosine similarity ≥ 0.68
OBJ_L2_THRESH  = 0.6   # tighter for entities; ≈ cosine similarity ≥ 0.82
MAX_EXAMPLES   = 3

# --- prompts ---
CONTEXT_SYSTEM = (
    "You are indexing a personal memoir written by Rose (born 1936, Cicero IL). "
    "Given a passage, write 1-2 sentences situating it in her life story. "
    "Include time period, people mentioned, and topic. Be concise."
)

EXTRACT_SYSTEM = (
    "You are extracting structured facts from a personal memoir written by Rose "
    "(born 1936, Cicero IL). "
    "Given a passage, extract every factual relationship as a JSON array of triples. "
    "Each triple must have exactly three string fields: "
    "\"subject\", \"predicate\", \"object\". "
    "Use short, lowercase, present-tense predicates (e.g. \"is born in\", \"married\", "
    "\"worked at\", \"lived in\", \"is related to\"). "
    "Subject and object should be proper nouns or short noun phrases. "
    "Return ONLY valid JSON — no explanation, no markdown fences. "
    "If no facts can be extracted, return []."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_chunks(data_dir: Path, ext: str) -> list[dict]:
    chunks = []
    for path in sorted(data_dir.glob(f"*.{ext}")):
        text = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines()]
        passages = [l for l in lines if len(l) > MIN_LINE_LEN]
        for i, passage in enumerate(passages):
            chunks.append({"source": path.name, "chunk_id": i, "text": passage})
    return chunks


def _save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# index helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# extract helpers
# ---------------------------------------------------------------------------

def extract_triples(chunk_text: str, client: Groq) -> list[dict]:
    """Call Groq to extract SPO triples from one chunk. Returns [] on any failure."""
    response = client.chat.completions.create(
        model=EXTRACT_MODEL,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": chunk_text},
        ],
        temperature=0,
        max_completion_tokens=400,
    )
    raw_text = response.choices[0].message.content.strip()

    # Try direct parse first, then fall back to extracting the [...] substring
    for attempt in (raw_text, _extract_json_array(raw_text)):
        if attempt is None:
            continue
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue

        # Unwrap {"triples": [...]} if the model wrapped the array
        if isinstance(parsed, dict) and "triples" in parsed:
            parsed = parsed["triples"]

        if not isinstance(parsed, list):
            continue

        validated = []
        for item in parsed:
            if isinstance(item, dict) and all(k in item for k in ("subject", "predicate", "object")):
                validated.append({
                    "subject":   str(item["subject"]).strip(),
                    "predicate": str(item["predicate"]).strip(),
                    "object":    str(item["object"]).strip(),
                })
        return validated

    print(f"  [WARN] Could not parse JSON from LLM response, skipping chunk.")
    return []


def _extract_json_array(text: str) -> str | None:
    """Try to extract the outermost [...] substring from text."""
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        return text[start:end + 1]
    return None


# ---------------------------------------------------------------------------
# dedupe helpers
# ---------------------------------------------------------------------------

def build_pred_embed_input(pred: str, examples: list[dict]) -> str:
    lines = [f"Predicate: {pred}", "Examples:"]
    for ex in examples[:MAX_EXAMPLES]:
        lines.append(f"- ({ex['subject']}, {pred}, {ex['object']})")
    return "\n".join(lines)


def build_entity_embed_input(entity: str, examples: list[dict], chunk_text: str) -> str:
    lines = [f"Entity: {entity}", "Triples:"]
    for ex in examples[:MAX_EXAMPLES]:
        lines.append(f"- ({ex['subject']}, {ex['predicate']}, {ex['object']})")
    lines.append(f"Context: {chunk_text}")
    return "\n".join(lines)


def greedy_cluster(items_sorted: list[str], embeddings: np.ndarray, threshold: float) -> dict[str, str]:
    """
    Greedy L2 clustering on unit-normalized embeddings. items_sorted should be
    ordered by frequency descending so the most-common form becomes the canonical
    label for each cluster. Returns {item -> canonical_item}.

    Embeddings are L2-normalized here so the distance is equivalent to cosine
    distance, making the threshold independent of embedding magnitude.
    Threshold reference on normalized vectors:
        0.4 ≈ cosine ≥ 0.92   (near-duplicate)
        0.6 ≈ cosine ≥ 0.82   (same entity, surface variation)
        0.8 ≈ cosine ≥ 0.68   (same concept, different phrasing)
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1.0)

    canonical_map: dict[str, str] = {}
    centroid_embs: list[np.ndarray] = []
    centroid_names: list[str] = []

    for item, emb in zip(items_sorted, embeddings):
        if not centroid_embs:
            centroid_embs.append(emb)
            centroid_names.append(item)
            canonical_map[item] = item
            continue

        centers = np.array(centroid_embs)
        dists = np.linalg.norm(centers - emb, axis=1)
        best_idx = int(np.argmin(dists))

        if dists[best_idx] <= threshold:
            canonical_map[item] = centroid_names[best_idx]
        else:
            centroid_embs.append(emb)
            centroid_names.append(item)
            canonical_map[item] = item

    return canonical_map


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_index() -> None:
    print(f"Loading chunks from {DATA_DIR}/*.{EXT} ...")
    chunks = load_chunks(DATA_DIR, EXT)
    print(f"  {len(chunks)} chunks found")

    # Resume: load any previously completed records
    records: list[dict] = []
    if OUT.exists():
        records = json.loads(OUT.read_text(encoding="utf-8"))
        print(f"  Resuming: {len(records)}/{len(chunks)} chunks already indexed")

    done = {(r["source"], r["chunk_id"]) for r in records}
    remaining = [c for c in chunks if (c["source"], c["chunk_id"]) not in done]

    if not remaining:
        print("Already complete.")
        return

    model = SentenceTransformer(EMBED_MODEL)
    client = Groq(api_key=GROQ_API_KEY) if CONTEXTUALIZE else None

    eta_min = len(remaining) * RATE_LIMIT_SLEEP / 60
    print(f"\nProcessing {len(remaining)} remaining chunks"
          + (f" (~{eta_min:.0f} min)" if CONTEXTUALIZE else "") + " ...")

    for chunk in tqdm(remaining):
        if CONTEXTUALIZE:
            context = generate_context(chunk["text"], client)
            time.sleep(RATE_LIMIT_SLEEP)
        else:
            context = ""

        text_to_embed = f"{context}\n{chunk['text']}".strip() if context else chunk["text"]
        emb = model.encode(text_to_embed)

        records.append({**chunk, "context": context, "embedding": emb.tolist()})
        _save_json(OUT, records)

    print(f"\nIndex saved → {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


def cmd_extract() -> None:
    print(f"Loading chunks from {DATA_DIR}/*.{EXT} ...")
    chunks = load_chunks(DATA_DIR, EXT)
    print(f"  {len(chunks)} chunks found")

    # Resume: load any previously completed work
    # kg_raw.json schema: {"processed": [[source, chunk_id], ...], "triples": [...]}
    triples: list[dict] = []
    processed: set[tuple[str, int]] = set()
    if KG_RAW.exists():
        data = json.loads(KG_RAW.read_text(encoding="utf-8"))
        triples = data.get("triples", [])
        processed = {(p[0], p[1]) for p in data.get("processed", [])}
        print(f"  Resuming: {len(processed)}/{len(chunks)} chunks already processed "
              f"({len(triples)} triples so far)")

    remaining = [c for c in chunks if (c["source"], c["chunk_id"]) not in processed]

    if not remaining:
        print("Already complete.")
        return

    client = Groq(api_key=GROQ_API_KEY)
    eta_min = len(remaining) * RATE_LIMIT_SLEEP / 60
    print(f"\nProcessing {len(remaining)} remaining chunks (~{eta_min:.0f} min) ...")

    for chunk in tqdm(remaining):
        new_triples = extract_triples(chunk["text"], client)
        for triple in new_triples:
            triples.append({
                "source":    chunk["source"],
                "chunk_id":  chunk["chunk_id"],
                "text":      chunk["text"],
                "subject":   triple["subject"],
                "predicate": triple["predicate"],
                "object":    triple["object"],
            })
        processed.add((chunk["source"], chunk["chunk_id"]))
        _save_json(KG_RAW, {"processed": list(processed), "triples": triples})
        time.sleep(RATE_LIMIT_SLEEP)

    print(f"\nExtracted {len(triples)} triples from {len(chunks)} chunks → {KG_RAW}")


def cmd_dedupe() -> None:
    if not KG_RAW.exists():
        print(f"Error: {KG_RAW} not found. Run 'python training.py extract' first.")
        sys.exit(1)

    data = json.loads(KG_RAW.read_text(encoding="utf-8"))
    # Support both the checkpointed {"triples": [...]} format and a plain array
    triples: list[dict] = data["triples"] if isinstance(data, dict) else data
    print(f"Loaded {len(triples)} triples from {KG_RAW}")

    model = SentenceTransformer(EMBED_MODEL)

    # ------------------------------------------------------------------
    # Pass A: Predicate deduplication
    # ------------------------------------------------------------------
    print("\n--- Pass A: Predicate deduplication ---")
    pred_freq = Counter(t["predicate"] for t in triples)
    sorted_preds = sorted(pred_freq, key=lambda p: pred_freq[p], reverse=True)
    print(f"  {len(sorted_preds)} unique predicates")

    pred_to_triples: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        pred_to_triples[t["predicate"]].append(t)

    pred_inputs = [
        build_pred_embed_input(p, pred_to_triples[p])
        for p in sorted_preds
    ]
    pred_embs = model.encode(pred_inputs, show_progress_bar=True)
    pred_canon = greedy_cluster(sorted_preds, pred_embs, PRED_L2_THRESH)

    for t in triples:
        t["canonical_predicate"] = pred_canon[t["predicate"]]

    n_pred_clusters = len(set(pred_canon.values()))
    print(f"  {len(sorted_preds)} predicates → {n_pred_clusters} canonical forms")

    # ------------------------------------------------------------------
    # Pass B: Entity deduplication (subjects + objects pooled)
    # ------------------------------------------------------------------
    print("\n--- Pass B: Entity deduplication (subjects + objects) ---")

    entity_freq: Counter = Counter()
    for t in triples:
        entity_freq[t["subject"]] += 1
        entity_freq[t["object"]] += 1
    sorted_entities = sorted(entity_freq, key=lambda e: entity_freq[e], reverse=True)
    print(f"  {len(sorted_entities)} unique entities")

    entity_to_triples: dict[str, list[dict]] = defaultdict(list)
    entity_to_chunk_text: dict[str, str] = {}
    for t in triples:
        for role in ("subject", "object"):
            ent = t[role]
            entity_to_triples[ent].append(t)
            if ent not in entity_to_chunk_text:
                entity_to_chunk_text[ent] = t["text"]

    entity_inputs = [
        build_entity_embed_input(e, entity_to_triples[e], entity_to_chunk_text[e])
        for e in sorted_entities
    ]
    entity_embs = model.encode(entity_inputs, show_progress_bar=True)
    entity_canon = greedy_cluster(sorted_entities, entity_embs, OBJ_L2_THRESH)

    for t in triples:
        t["canonical_subject"] = entity_canon[t["subject"]]
        t["canonical_object"]  = entity_canon[t["object"]]

    n_entity_clusters = len(set(entity_canon.values()))
    print(f"  {len(sorted_entities)} entities → {n_entity_clusters} canonical forms")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    _save_json(KG_OUT, triples)
    print(f"\nKG saved → {KG_OUT}  ({KG_OUT.stat().st_size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="training.py",
        description="project-rose pre-inference pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("index",   help="Chunk, contextualize, embed → index.json")
    subparsers.add_parser("extract", help="Extract KG triples via LLM → kg_raw.json")
    subparsers.add_parser("dedupe",  help="Canonicalize predicates + entities → kg.json")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index()
    elif args.command == "extract":
        cmd_extract()
    elif args.command == "dedupe":
        cmd_dedupe()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
