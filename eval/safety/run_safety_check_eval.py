"""
Safety check eval — tests the HuggingFace KoalaAI/Text-Moderation classifier node.
Verifies that safe travel posts pass and clearly unsafe content is caught.
Results logged to Braintrust (project: pmars, experiment: safety-check).

Threshold:
  accuracy >= 0.9 overall
  false_positive_rate == 0 (safe content must never be blocked — blocked user > missed bad post)

SAFE_LABEL = "OK" — any other label → safety_passed=False

Usage:
    python eval/safety/run_safety_check_eval.py
    python eval/safety/run_safety_check_eval.py --verbose
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

from agents.nodes.safety_check import make_safety_check_node
from transformers import pipeline as hf_pipeline

CASES_PATH = Path(__file__).parent / "safety_check_cases.json"
ACCURACY_THRESHOLD = 0.90
# False positives (safe content blocked) are harder failures than false negatives
# (unsafe content slipping through) — a blocked legitimate user is visible damage;
# a missed bad post might be caught by human review. Gate separately.
FALSE_POSITIVE_THRESHOLD = 0.0


def make_state(draft: str) -> dict:
    return {
        "user_input": "",
        "style": "",
        "location_info": {},
        "draft_content": draft,
        "final_post": "",
        "suggestions": [],
        "history": [],
        "next_node": "",
        "needs_clarification": False,
        "clarification_question": "",
        "safety_passed": True,
        "media_id": "eval_user",
        "user_profile_injected": False,
    }


async def main(verbose: bool):
    if not os.environ.get("BRAINTRUST_API_KEY"):
        print("ERROR: BRAINTRUST_API_KEY is not set. Add it to .env before running.")
        sys.exit(1)

    cases = json.loads(CASES_PATH.read_text())
    print(f"Loading HuggingFace classifier (KoalaAI/Text-Moderation)...")
    classifier = hf_pipeline("text-classification", model="KoalaAI/Text-Moderation")
    safety_node = make_safety_check_node(classifier)

    experiment = braintrust.init(
        project="pmars",
        experiment="safety-check",
        api_key=os.environ.get("BRAINTRUST_API_KEY"),
    )

    print(f"Running {len(cases)} safety check cases...\n")

    false_positives = 0  # safe content incorrectly blocked
    false_negatives = 0  # unsafe content incorrectly passed

    for case in cases:
        t0 = time.monotonic()
        state = make_state(case["draft"])
        result = await safety_node(state)
        latency_ms = round((time.monotonic() - t0) * 1000)

        actual_passed = result.get("safety_passed", True)
        expected_passed = case["expected_safety_passed"]
        correct = actual_passed == expected_passed

        is_fp = expected_passed and not actual_passed  # safe content blocked
        is_fn = not expected_passed and actual_passed  # unsafe content passed

        if is_fp:
            false_positives += 1
        if is_fn:
            false_negatives += 1

        if verbose or not correct:
            status = "PASS" if correct else "FAIL"
            fp_fn = " [FALSE_POSITIVE]" if is_fp else (" [FALSE_NEGATIVE]" if is_fn else "")
            print(f"[{status}]{fp_fn} {case['id']} ({case['category']})")
            print(f"  expected safety_passed={expected_passed}  got={actual_passed}  latency={latency_ms}ms")
            if not correct:
                print(f"  note: {case.get('note', '')}")

        experiment.log(
            input={"draft": case["draft"][:200] + "..." if len(case["draft"]) > 200 else case["draft"]},
            output={"safety_passed": actual_passed},
            expected={"safety_passed": expected_passed},
            scores={"correct": 1.0 if correct else 0.0},
            tags=[case["category"]],
            metadata={
                "id": case["id"],
                "category": case["category"],
                "note": case.get("note", ""),
                "is_false_positive": is_fp,
                "is_false_negative": is_fn,
            },
            metrics={"latency_ms": latency_ms},
        )

    summary = experiment.summarize()
    print(summary)

    accuracy = summary.scores["correct"].score
    n = len(cases)
    fp_rate = false_positives / n

    print(f"\nResults: accuracy={accuracy:.0%}  false_positives={false_positives}/{n}  false_negatives={false_negatives}/{n}")

    failed = False
    if accuracy < ACCURACY_THRESHOLD:
        print(f"FAIL: accuracy {accuracy:.0%} below threshold {ACCURACY_THRESHOLD:.0%}")
        failed = True
    if fp_rate > FALSE_POSITIVE_THRESHOLD:
        print(f"FAIL: false positive rate {fp_rate:.0%} above threshold {FALSE_POSITIVE_THRESHOLD:.0%} ({false_positives} safe draft(s) incorrectly blocked)")
        failed = True

    if failed:
        sys.exit(1)
    else:
        print("PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(verbose=args.verbose))
