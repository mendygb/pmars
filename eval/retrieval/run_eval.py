"""
Retrieval eval — runs test_queries.json against the live hybrid retrieval pipeline.
Measures Recall@k and logs results to Braintrust (project: pmars, experiment: retrieval-recall).

Usage:
    python eval/retrieval/run_eval.py
    python eval/retrieval/run_eval.py --topk 4      # default is 4
    python eval/retrieval/run_eval.py --verbose
    python eval/retrieval/run_eval.py --filter tag_based

Requires BRAINTRUST_API_KEY, PINECONE_API_KEY, OPENAI_API_KEY, and MongoDB running.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import braintrust

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

_rag_path = Path(__file__).parent.parent.parent / "rag" / "03_rag_query.py"
_spec = importlib.util.spec_from_file_location("rag03", _rag_path)
_rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rag)

QUERIES_PATH = Path(__file__).parent / "test_queries.json"
CLEANED_POSTS_PATH = Path(__file__).parent.parent.parent / "rag" / "cleaned_posts.json"
CLEANED_CHUNKS_PATH = Path(__file__).parent.parent.parent / "rag" / "cleaned_chunks.json"
PASS_THRESHOLD = 0.8


def load_queries(category_filter=None):
    queries = json.loads(QUERIES_PATH.read_text())
    if category_filter:
        queries = [q for q in queries if q["category"] == category_filter]
    runnable = [q for q in queries if q["expected_pids"] != ["__TODO__"]]
    skipped = len(queries) - len(runnable)
    if skipped:
        print(f"Skipping {skipped} cases with __TODO__ expected_pids.")
    return runnable


def setup_retrieval():
    from pinecone import Pinecone
    from openai import OpenAI
    from pymongo import MongoClient

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(_rag.INDEX_NAME)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    mongo = MongoClient(_rag.MONGO_URI)
    posts_col = mongo[_rag.DB_NAME][_rag.COLLECTION_NAME]

    cleaned_posts = {p["pid"]: p for p in json.loads(CLEANED_POSTS_PATH.read_text())}

    chunks_by_pid = {}
    for chunk in json.loads(CLEANED_CHUNKS_PATH.read_text()):
        chunks_by_pid.setdefault(chunk["pid"], []).append(chunk)

    return index, client, posts_col, chunks_by_pid, cleaned_posts


def recall_at_k(retrieved_pids: list[str], expected_pids: list[str]) -> float:
    if not expected_pids:
        return 1.0
    return sum(1 for pid in expected_pids if pid in retrieved_pids) / len(expected_pids)


def reciprocal_rank(retrieved_pids: list[str], expected_pids: list[str]) -> float:
    expected_set = set(expected_pids)
    for rank, pid in enumerate(retrieved_pids, start=1):
        if pid in expected_set:
            return 1.0 / rank
    return 0.0


def main(top_k: int, verbose: bool, category_filter: str | None):
    if not os.environ.get("BRAINTRUST_API_KEY"):
        print("ERROR: BRAINTRUST_API_KEY is not set. Add it to .env before running.")
        sys.exit(1)

    queries = load_queries(category_filter)
    if not queries:
        print("No runnable queries. Fill in expected_pids in test_queries.json first.")
        return

    print("Setting up retrieval pipeline...")
    index, client, posts_col, chunks_by_pid, cleaned_posts = setup_retrieval()

    experiment = braintrust.init(
        project="pmars",
        experiment="retrieval-recall",
        api_key=os.environ.get("BRAINTRUST_API_KEY"),
    )

    print(f"Running {len(queries)} queries (recall@{top_k})...\n")

    category_recalls: dict[str, list[float]] = {}

    for q in queries:
        t0 = time.monotonic()
        docs, metrics = _rag.retrieve(index, client, posts_col, chunks_by_pid, cleaned_posts, q["query"])
        latency_ms = (time.monotonic() - t0) * 1000

        retrieved_pids = [d["pid"] for d in docs[:top_k]]
        recall = recall_at_k(retrieved_pids, q["expected_pids"])
        rr = reciprocal_rank(retrieved_pids, q["expected_pids"])

        category_recalls.setdefault(q["category"], []).append((recall, rr))

        if verbose or recall < 1.0:
            status = "PASS" if recall == 1.0 else f"PARTIAL({recall:.0%})"
            print(f"[{status}] {q['id']} — {q['query'][:60]}")
            if recall < 1.0:
                print(f"  expected: {q['expected_pids']}")
                print(f"  got:      {retrieved_pids}")
            print(f"  recall@{top_k}={recall:.2f}  mrr={rr:.2f}")

        experiment.log(
            input={"query": q["query"]},
            output={"retrieved_pids": retrieved_pids},
            expected={"pids": q["expected_pids"]},
            scores={"recall_at_k": recall, "mrr": rr},
            tags=[q["category"]],
            metadata={
                "id": q["id"],
                "category": q["category"],
                "top_k": top_k,
                "note": q.get("note", ""),
                "vector_hits": metrics.get("vector_hits", 0),
                "tag_hits": metrics.get("tag_hits", 0),
                "matched_tags": metrics.get("matched_tags", []),
            },
            metrics={"latency_ms": round(latency_ms)},
        )

    print("\nBy category:")
    for cat in sorted(category_recalls):
        pairs = category_recalls[cat]
        avg_recall = sum(r for r, _ in pairs) / len(pairs)
        avg_mrr = sum(m for _, m in pairs) / len(pairs)
        print(f"  {cat}: recall@{top_k}={avg_recall:.0%}  mrr={avg_mrr:.2f}")

    summary = experiment.summarize()
    print(summary)

    avg_recall = summary.scores["recall_at_k"].score
    if avg_recall < PASS_THRESHOLD:
        print(f"\nFAIL: recall@{top_k} {avg_recall:.0%} below threshold {PASS_THRESHOLD:.0%}")
        sys.exit(1)
    else:
        print(f"\nPASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--filter", dest="category", default=None)
    args = parser.parse_args()

    main(top_k=args.topk, verbose=args.verbose, category_filter=args.category)
