"""
training.py — pre-inference pipeline for project-rose.

Subcommands:
    python src/training.py index    Chunk, contextualize, embed → data/02_extracted/index.json
    python src/training.py extract  Extract KG triples via LLM → data/02_extracted/kg_raw.json
    python src/training.py dedupe   Canonicalize entities → data/02_extracted/kg.json

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
    Embedding-based entity canonicalization — no LLM calls required.
    Entities are pooled from both subject and object positions and deduplicated
    with greedy L2 clustering: most-frequent form becomes the canonical label.
    Predicates come from a fixed allowed list, so no predicate dedup is needed.
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

ROOT = Path(__file__).resolve().parent.parent
CONF_DIR = ROOT / "conf"

# --- index constants ---
DATA_DIR         = ROOT / "data" / "01_raw"
OUT              = ROOT / "data" / "02_extracted" / "index.json"
EXT              = "txt"
EMBED_MODEL      = "all-MiniLM-L6-v2"
MIN_LINE_LEN     = 60
CONTEXTUALIZE    = True
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
CONTEXT_MODEL    = "llama-3.1-8b-instant"
RATE_LIMIT_SLEEP = 2.1  # seconds; keeps us under Groq's 30 req/min free tier

# --- KG constants ---
KG_RAW         = ROOT / "data" / "02_extracted" / "kg_raw.json"
KG_OUT         = ROOT / "data" / "02_extracted" / "kg.json"
EXTRACT_MODEL  = "llama-3.3-70b-versatile"
OBJ_L2_THRESH  = 0.6   # entity dedup threshold on unit-normalized embeddings; ≈ cosine ≥ 0.82
MAX_EXAMPLES   = 5

# Maps predicate → (valid subject_types, valid object_types).
# Used in extract_triples() to reject type-incompatible triples without a hardcoded word blocklist.
_P   = frozenset({"person"})
_L   = frozenset({"place"})
_O   = frozenset({"organization"})
_T   = frozenset({"thing"})
_LO  = frozenset({"place", "organization"})
_LTO = frozenset({"place", "thing", "organization"})

PREDICATE_TYPE_RULES: dict[str, tuple[frozenset, frozenset]] = {
    # Kinship & social — both sides must be people
    "is child of":           (_P, _P),
    "is parent of":          (_P, _P),
    "is sibling of":         (_P, _P),
    "is spouse of":          (_P, _P),
    "is cousin of":          (_P, _P),
    "is grandchild of":      (_P, _P),
    "is grandparent of":     (_P, _P),
    "is aunt or uncle of":   (_P, _P),
    "is niece or nephew of": (_P, _P),
    "is godparent of":       (_P, _P),
    "is godchild of":        (_P, _P),
    "is friend of":          (_P, _P),
    "married":               (_P, _P),
    "divorced":              (_P, _P),
    "is related to":         (_P, _P),
    # Location events — object must be a place
    "born in":               (_P, _L),
    "died in":               (_P, _L),
    "grew up in":            (_P, _L),
    "lived in":              (_P, _L),
    "moved to":              (_P, _L),
    "visited":               (_P, _LO),   # can visit a place or organization
    # Work & education — object is an org or place
    "worked at":             (_P, _LO),
    "retired from":          (_P, _LO),
    "served as":             (_P, _LO | _T),  # roles can be tagged org or thing
    "attended":              (_P, _LO),
    "graduated from":        (_P, _LO),
    "is member of":          (_P, _O),
    # Property — object is a thing, place, or organization
    "owned":                 (_P, _LTO),
    "purchased":             (_P, _LTO),
    "sold":                  (_P, _LTO),
    "inherited":             (_P, _LTO),
}

# Authoritative predicate allowlist — also used to hard-filter LLM output in extract_triples()
ALLOWED_PREDICATES: frozenset[str] = frozenset({
    # Family
    "is child of", "is parent of", "is sibling of", "is spouse of", "is cousin of",
    "is grandchild of", "is grandparent of", "is aunt or uncle of", "is niece or nephew of",
    "is godparent of", "is godchild of",
    # Life
    "born in", "died in", "married", "divorced",
    "grew up in", "lived in", "moved to", "visited",
    # Work
    "worked at", "retired from", "served as",
    # Education
    "attended", "graduated from",
    # Social
    "is friend of", "is member of",
    # Property
    "owned", "purchased", "sold", "inherited",
    # Other
    "is related to",
})

# --- prompts (loaded from conf/) ---
CONTEXT_SYSTEM = (CONF_DIR / "context_system.txt").read_text(encoding="utf-8").strip()

_pred_list = "\n".join(f"  {p}" for p in sorted(ALLOWED_PREDICATES))
EXTRACT_SYSTEM = (CONF_DIR / "extract_system.txt").read_text(encoding="utf-8").replace("__PRED_LIST__", _pred_list)


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
        max_completion_tokens=600,
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
            if not (isinstance(item, dict) and all(k in item for k in ("subject", "predicate", "object"))):
                continue

            subj = str(item["subject"]).strip()
            pred = str(item["predicate"]).strip().lower()
            obj  = str(item["object"]).strip()

            # Reject self-referential triples
            if subj.lower() == obj.lower():
                print(f"  [SKIP] subject == object: '{subj}'")
                continue

            # Reject disallowed predicates
            if pred not in ALLOWED_PREDICATES:
                print(f"  [SKIP] disallowed predicate: '{pred}'")
                continue

            # Type-compatibility check: LLM-provided types must match predicate rules
            subj_type = str(item.get("subject_type", "")).strip().lower()
            obj_type  = str(item.get("object_type",  "")).strip().lower()
            rules = PREDICATE_TYPE_RULES.get(pred)
            if rules:
                valid_subj, valid_obj = rules
                if subj_type and subj_type not in valid_subj:
                    print(f"  [SKIP] '{subj}' ({subj_type}) invalid subject type for '{pred}'")
                    continue
                if obj_type and obj_type not in valid_obj:
                    print(f"  [SKIP] '{obj}' ({obj_type}) invalid object type for '{pred}'")
                    continue

            validated.append({
                "subject":      subj,
                "subject_type": subj_type,
                "predicate":    pred,
                "object":       obj,
                "object_type":  obj_type,
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

def build_entity_embed_input(entity: str, examples: list[dict]) -> str:
    """
    Build the string to embed for entity deduplication.

    Uses only the entity name and the predicates it appears with — not the chunk
    text. Including the full chunk contaminates the embedding with co-occurring
    entities: e.g. 'the Apartment' would absorb 'Bob' because both appear in the
    same sentence. Predicate signatures cleanly distinguish entity types
    (person-role predicates vs. location predicates vs. property predicates).
    """
    predicates = list(dict.fromkeys(ex["predicate"] for ex in examples))  # unique, insertion-ordered
    return f"Entity: {entity}\nPredicates: {', '.join(predicates[:MAX_EXAMPLES])}"


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
                "source":       chunk["source"],
                "chunk_id":     chunk["chunk_id"],
                "text":         chunk["text"],
                "subject":      triple["subject"],
                "subject_type": triple["subject_type"],
                "predicate":    triple["predicate"],
                "object":       triple["object"],
                "object_type":  triple["object_type"],
            })
        processed.add((chunk["source"], chunk["chunk_id"]))
        _save_json(KG_RAW, {"processed": list(processed), "triples": triples})
        time.sleep(RATE_LIMIT_SLEEP)

    print(f"\nExtracted {len(triples)} triples from {len(chunks)} chunks → {KG_RAW}")


def cmd_dedupe() -> None:
    if not KG_RAW.exists():
        print(f"Error: {KG_RAW} not found. Run 'python src/training.py extract' first.")
        sys.exit(1)

    data = json.loads(KG_RAW.read_text(encoding="utf-8"))
    # Support both the checkpointed {"triples": [...]} format and a plain array
    triples: list[dict] = data["triples"] if isinstance(data, dict) else data
    print(f"Loaded {len(triples)} triples from {KG_RAW}")

    model = SentenceTransformer(EMBED_MODEL)

    # ------------------------------------------------------------------
    # Entity deduplication (subjects and objects in separate pools)
    #
    # Subjects are overwhelmingly people; objects are people, places, and
    # organizations. Pooling them together lets a high-frequency person name
    # ("Rose") absorb unrelated location or object strings that happen to be
    # near it in embedding space. Clustering each pool independently prevents
    # cross-role contamination while still collapsing surface variants within
    # each role (e.g. "Cicero" / "Cicero IL" / "Cicero, IL" in objects).
    # ------------------------------------------------------------------
    print("\n--- Entity deduplication (subjects and objects separately) ---")

    entity_to_triples: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        for role in ("subject", "object"):
            entity_to_triples[t[role]].append(t)

    def _cluster_pool(entities: list[str]) -> dict[str, str]:
        freq: Counter = Counter()
        for e in entities:
            freq[e] += 1
        sorted_ents = sorted(freq, key=lambda e: freq[e], reverse=True)
        inputs = [build_entity_embed_input(e, entity_to_triples[e]) for e in sorted_ents]
        embs = model.encode(inputs, show_progress_bar=False)
        return greedy_cluster(sorted_ents, embs, OBJ_L2_THRESH)

    subj_entities = [t["subject"] for t in triples]
    obj_entities  = [t["object"]  for t in triples]

    print("  Clustering subjects ...")
    subj_canon = _cluster_pool(subj_entities)
    print(f"  {len(set(subj_entities))} subjects → {len(set(subj_canon.values()))} canonical forms")

    print("  Clustering objects ...")
    obj_canon = _cluster_pool(obj_entities)
    print(f"  {len(set(obj_entities))} objects → {len(set(obj_canon.values()))} canonical forms")

    for t in triples:
        t["canonical_subject"] = subj_canon[t["subject"]]
        t["canonical_object"]  = obj_canon[t["object"]]

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

    subparsers.add_parser("index",   help="Chunk, contextualize, embed → data/02_extracted/index.json")
    subparsers.add_parser("extract", help="Extract KG triples via LLM → data/02_extracted/kg_raw.json")
    subparsers.add_parser("dedupe",  help="Canonicalize entities → data/02_extracted/kg.json")

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
