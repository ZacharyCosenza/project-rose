"""
eval.py — RAG pipeline evaluation framework.

Runs 30 questions (10 easy, 10 medium, 10 hard) through inference.py, judges
each answer with an LLM, and reports accuracy by difficulty tier.

Scoring: correct=1.0, partial=0.5, incorrect=0.0

Usage:
    python src/eval.py                      # run all 30 questions
    python src/eval.py --difficulty easy    # run one tier only
    python src/eval.py --no-kg              # disable KG augmentation
    python src/eval.py --out results.json   # custom output path

All inference.py flags are forwarded via their eval.py equivalents:
    python src/eval.py --no-kg --no-rerank --no-bm25
    python src/eval.py --top-k 30 --rerank-top-k 15 --kg-top-k 20
    python src/eval.py --max-tokens 800 --temperature 0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

load_dotenv()

ROOT        = Path(__file__).resolve().parent.parent
CONF_DIR    = ROOT / "conf"
RESULTS_DIR = ROOT / "eval_results"
INFERENCE   = ROOT / "src" / "inference.py"

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
EVAL_MODEL   = "llama-3.3-70b-versatile"
EVAL_SLEEP   = 4.5   # seconds between questions; each question makes 2 Groq calls

SCORE_MAP = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

JUDGE_SYSTEM = (CONF_DIR / "judge_system.txt").read_text(encoding="utf-8").strip()
QUESTIONS    = json.loads((CONF_DIR / "questions.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge(question: str, expected: str, actual: str, client: Groq) -> tuple[str, float]:
    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Expected: {expected}\n"
                    f"Actual: {actual}"
                ),
            },
        ],
        temperature=0,
        max_completion_tokens=5,
    )
    verdict = response.choices[0].message.content.strip().lower().rstrip(".")
    if verdict not in SCORE_MAP:
        verdict = "incorrect"
    return verdict, SCORE_MAP[verdict]


# ---------------------------------------------------------------------------
# Inference via subprocess
# ---------------------------------------------------------------------------

def run_inference(question: str, args: argparse.Namespace) -> dict:
    """
    Call inference.py as a subprocess with --json output.
    Returns the parsed result dict, or raises RuntimeError on failure.
    """
    cmd = [sys.executable, str(INFERENCE), question, "--json"]

    if args.no_kg:
        cmd.append("--no-kg")
    if args.no_rerank:
        cmd.append("--no-rerank")
    if args.no_bm25:
        cmd.append("--no-bm25")
    if args.top_k != 25:
        cmd += ["--top-k", str(args.top_k)]
    if args.rerank_top_k != 12:
        cmd += ["--rerank-top-k", str(args.rerank_top_k)]
    if args.kg_top_k != 15:
        cmd += ["--kg-top-k", str(args.kg_top_k)]
    if args.semantic_weight != 0.60:
        cmd += ["--semantic-weight", str(args.semantic_weight)]
    if args.max_tokens != 500:
        cmd += ["--max-tokens", str(args.max_tokens)]
    if args.temperature != 0:
        cmd += ["--temperature", str(args.temperature)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"inference.py exited {proc.returncode}:\n{proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval(questions: list[dict], args: argparse.Namespace) -> dict:
    client = Groq(api_key=GROQ_API_KEY)

    results = []
    for q in tqdm(questions, desc="Evaluating"):
        rag_result = run_inference(q["question"], args)
        actual = rag_result["answer"]
        verdict, score = judge(q["question"], q["expected"], actual, client)
        results.append({
            "id":         q["id"],
            "difficulty": q["difficulty"],
            "question":   q["question"],
            "expected":   q["expected"],
            "actual":     actual,
            "context":    rag_result["context"],
            "verdict":    verdict,
            "score":      score,
        })
        time.sleep(EVAL_SLEEP)

    # Aggregate by difficulty
    by_difficulty: dict[str, dict] = {}
    for r in results:
        d = r["difficulty"]
        by_difficulty.setdefault(d, {"scores": [], "n": 0})
        by_difficulty[d]["scores"].append(r["score"])
        by_difficulty[d]["n"] += 1

    summary = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "kg_enabled":       not args.no_kg,
        "total_questions":  len(results),
        "overall_accuracy": sum(r["score"] for r in results) / len(results),
        "by_difficulty":    {
            d: {
                "accuracy": sum(v["scores"]) / v["n"],
                "n":        v["n"],
                "correct":  sum(1 for s in v["scores"] if s == 1.0),
                "partial":  sum(1 for s in v["scores"] if s == 0.5),
                "incorrect":sum(1 for s in v["scores"] if s == 0.0),
            }
            for d, v in by_difficulty.items()
        },
        "results": results,
    }
    return summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(summary: dict) -> None:
    kg_label = "on" if summary["kg_enabled"] else "off"
    print(f"\n{'='*55}")
    print(f"  Overall accuracy: {summary['overall_accuracy']:.0%}   KG: {kg_label}")
    print(f"{'='*55}")
    for diff in ("easy", "medium", "hard"):
        if diff not in summary["by_difficulty"]:
            continue
        s = summary["by_difficulty"][diff]
        bar = ("█" * s["correct"]) + ("▒" * s["partial"]) + ("░" * s["incorrect"])
        print(f"  {diff:8s}  {s['accuracy']:.0%}  {bar}  ({s['n']} questions)")
    print()
    verdicts = {"correct": "✓", "partial": "~", "incorrect": "✗"}
    for r in summary["results"]:
        icon = verdicts.get(r["verdict"], "?")
        label = r["difficulty"][0].upper()
        print(f"  {icon} [{label}] {r['question'][:58]}")
    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eval.py",
        description="Evaluate the RAG pipeline",
    )
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"],
                        help="Run one difficulty tier only")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save results to this path (default: eval_results/TIMESTAMP.json)")
    # Forwarded to inference.py
    parser.add_argument("--no-kg",      action="store_true", help="Disable KG augmentation")
    parser.add_argument("--no-rerank",  action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--no-bm25",    action="store_true", help="Disable BM25 (pure semantic search)")
    parser.add_argument("--top-k",      type=int,   default=25,   help="Hybrid search candidates (default 25)")
    parser.add_argument("--rerank-top-k", type=int, default=12,   help="Chunks kept after reranking (default 12)")
    parser.add_argument("--kg-top-k",   type=int,   default=15,   help="Max KG triples injected (default 15)")
    parser.add_argument("--semantic-weight", type=float, default=0.60, help="Semantic blend ratio 0-1 (default 0.60)")
    parser.add_argument("--max-tokens", type=int,   default=500,  help="LLM max completion tokens (default 500)")
    parser.add_argument("--temperature", type=float, default=0,   help="LLM temperature (default 0)")
    args = parser.parse_args()

    questions = QUESTIONS
    if args.difficulty:
        questions = [q for q in questions if q["difficulty"] == args.difficulty]

    n = len(questions)
    eta = n * EVAL_SLEEP / 60
    print(f"Running {n} questions (~{eta:.0f} min) ...")

    summary = run_eval(questions, args)
    print_summary(summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = args.out or RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
