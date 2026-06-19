"""
Director routing eval — runs golden_dataset.json against the live Director node.
Results are logged to Braintrust (project: pmars, experiment: director-routing).

Usage:
    python eval/director/run_eval.py
    python eval/director/run_eval.py --verbose   # show per-case output
    python eval/director/run_eval.py --filter first_turn_vague  # run one category

Exits with code 1 if next_node accuracy falls below PASS_THRESHOLD.
Requires BRAINTRUST_API_KEY in .env.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import braintrust

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from agents.nodes.director import make_director_node
from agents.state import PostState

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
PASS_THRESHOLD = 0.85


def load_cases(category_filter: str | None = None) -> list[dict]:
    cases = json.loads(GOLDEN_PATH.read_text())
    if category_filter:
        cases = [c for c in cases if c["category"] == category_filter]
    cases = [c for c in cases if c["state"]["user_input"] != "__PLACEHOLDER__"]
    return cases


def build_state(case: dict) -> PostState:
    s = case["state"]
    return PostState(
        user_input=s["user_input"],
        history=s.get("history", []),
        draft_content=s.get("draft_content") or "",
        style="",
        location_info={},
        final_post="",
        suggestions=[],
        next_node="",
        needs_clarification=False,
        clarification_question="",
        safety_passed=False,
        media_id="eval_user",
        user_profile_injected=False,
    )


async def main(verbose: bool, category_filter: str | None):
    if not os.environ.get("BRAINTRUST_API_KEY"):
        print("ERROR: BRAINTRUST_API_KEY is not set. Add it to .env before running.")
        sys.exit(1)

    cases = load_cases(category_filter)
    if not cases:
        print("No cases to run.")
        return

    print(f"Running {len(cases)} cases...")
    director_node = make_director_node(debug=False)

    experiment = braintrust.init(
        project="pmars",
        experiment="director-routing",
        api_key=os.environ.get("BRAINTRUST_API_KEY"),
    )

    category_results: dict[str, list[bool]] = {}

    for case in cases:
        state = build_state(case)
        t0 = time.monotonic()
        result = await director_node(state)
        latency_ms = (time.monotonic() - t0) * 1000

        expected = case["expected"]
        actual_node = result.get("next_node", "")
        actual_clarification = result.get("needs_clarification", False)

        accepted_nodes = expected.get("accepted_next_nodes", [expected["next_node"]])
        node_match = actual_node in accepted_nodes
        clarification_match = actual_clarification == expected.get("needs_clarification", False)
        passed = node_match and clarification_match

        category_results.setdefault(case["category"], []).append(passed)

        if verbose or not passed:
            status = "PASS" if passed else "FAIL"
            print(f"\n[{status}] {case['id']} ({case['category']})")
            print(f"  input: {state['user_input'][:80]}...")
            if not node_match:
                print(f"  next_node: expected={accepted_nodes} actual={actual_node}")
            if not clarification_match:
                print(f"  needs_clarification: expected={expected.get('needs_clarification')} actual={actual_clarification}")

        experiment.log(
            input=case["state"],
            output={"next_node": actual_node, "needs_clarification": actual_clarification},
            expected={"next_node": expected["next_node"], "needs_clarification": expected.get("needs_clarification", False)},
            scores={
                "node_match": 1.0 if node_match else 0.0,
                "clarification_match": 1.0 if clarification_match else 0.0,
            },
            tags=[case["category"]],
            metadata={
                "id": case["id"],
                "category": case["category"],
                "accepted_nodes": accepted_nodes,
                "note": case.get("note", ""),
            },
            metrics={"latency_ms": round(latency_ms)},
        )

    # Per-category breakdown (terminal only — Braintrust shows this by metadata filter)
    print("\nBy category:")
    for cat in sorted(category_results):
        results = category_results[cat]
        print(f"  {cat}: {sum(results)}/{len(results)}")

    summary = experiment.summarize()
    print(summary)

    node_acc = summary.scores["node_match"].score
    if node_acc < PASS_THRESHOLD:
        print(f"\nFAIL: node_match {node_acc:.0%} below threshold {PASS_THRESHOLD:.0%}")
        sys.exit(1)
    else:
        print(f"\nPASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--filter", dest="category", default=None)
    args = parser.parse_args()

    asyncio.run(main(verbose=args.verbose, category_filter=args.category))
