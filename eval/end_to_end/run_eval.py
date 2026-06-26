"""
End-to-end eval — runs the full pipeline on fixed test cases and judges output quality.
Results are logged to Braintrust (project: pmars, experiment: e2e-quality).

Scoring layers:
  pipeline_complete  — binary sanity check: did pipeline run and return a non-empty post?
  faithfulness       — claim decomposition: do post claims stay within retrieved context + request?
  relevance          — is the post about what the user asked for? (single pass, with completeness)
  completeness       — does the post cover the key details the user mentioned? (same pass)
  hook_strength      — reference only, no threshold
  style_match        — reference only, no threshold
  hashtag_quality    — reference only, no threshold
  personal_voice     — reference only, no threshold

Thresholds (CI gate):
  pipeline_complete: per-case, must be 1.0 for every case
  faithfulness / relevance / completeness: experiment average >= QUALITY_THRESHOLD (0.6)
  0.6 reflects creative writing baseline — LLM blends retrieved facts with world knowledge,
  which is expected behavior. 0.6 still catches systematic hallucination patterns.

Judge model: ChatOpenAI wrapping gpt-4o-mini at temperature=0 (override with EVAL_JUDGE_MODEL; swap ChatOpenAI for another LangChain provider class for cross-provider use)

Usage:
    python eval/end_to_end/run_eval.py
    python eval/end_to_end/run_eval.py --verbose
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import braintrust
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pinecone import Pinecone
from pymongo import MongoClient
from tavily import TavilyClient
from transformers import pipeline as hf_pipeline

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from agents.pipeline import build_graph
from agents.state import PostState

CASES_PATH = Path(__file__).parent / "test_cases.json"
RAG_DIR = Path(__file__).parent.parent.parent / "rag"

JUDGE_MODEL    = os.environ.get("EVAL_JUDGE_MODEL",    "gpt-4o-mini")  # override to switch OpenAI judge model (e.g. gpt-4o); cross-provider requires langchain-anthropic + conditional client
JUDGE_PROVIDER = os.environ.get("EVAL_JUDGE_PROVIDER", "openai")       # reserved for future cross-provider support; currently only "openai" is wired up
QUALITY_THRESHOLD = 0.6  # faithfulness, relevance, completeness — 0.6 reflects creative writing baseline (LLM blends retrieved facts with world knowledge)


# ── Judge prompts ──────────────────────────────────────────────────────────────

_CLAIM_EXTRACT_PROMPT = """Extract only concrete, verifiable factual claims from this social media post.

Include:
- Specific named places, trails, or attractions (e.g. "The Main Trail is a 2-mile loop")
- Concrete numbers: prices, distances, hours, ratings, heights (e.g. "parking costs $15")
- Named food items or products (e.g. "they serve yakitori and takoyaki")
- Specific facilities or amenities (e.g. "there are picnic areas")
- Specific opening hours or operational facts

Exclude (do NOT extract):
- Atmospheric or sensory descriptions ("the air was crisp", "fog danced between the hills")
- Subjective opinions or evaluations ("breathtaking", "unmatched", "stunning views")
- Metaphorical or creative language ("painted in shades of gold", "sea of people")
- General impressions or emotional reactions ("it was surreal", "overwhelming in the best way")
- Universally observable facts ("traffic lights change", "people walk")
- The author's personal plans or feelings about future visits

Post:
{post}

Return JSON only: {{"claims": ["claim 1", "claim 2", ...]}}"""

_CLAIM_VERIFY_PROMPT = """Is the following claim supported by the provided context? The context is the source material (retrieved information + user's original description) used to write the post.

Context:
{context}

Claim: {claim}

Return JSON only: {{"supported": true}} or {{"supported": false}}"""

_QUALITY_PROMPT = """A user asked for help writing a social media post about a place they visited. Evaluate the generated post.

User request: {user_input}
Generated post: {post}

Score each criterion from 0.0 to 1.0:
- relevance: Is the post clearly about the experience/place the user described?
- completeness: Does the post incorporate the key details and aspects the user mentioned?

Think through your evaluation first, then assign scores.

Return JSON only: {{"reasoning": "step-by-step evaluation", "relevance": float, "completeness": float}}"""

_REFERENCE_PROMPT = """Evaluate this Xiaohongshu (Chinese social media) post for quality.

Post:
{post}

Intended style: {style}

Score each criterion from 0.0 to 1.0:
- hook_strength: Opening lines are specific and attention-grabbing, not generic
- style_match: Post format and tone match the intended style ({style})
- hashtag_quality: Hashtags are relevant and specific, not generic filler
- personal_voice: Reads like an authentic personal experience, not a template

Think through your evaluation first, then assign scores.

Return JSON only: {{"reasoning": "step-by-step evaluation", "hook_strength": float, "style_match": float, "hashtag_quality": float, "personal_voice": float}}"""


# ── Judge functions ────────────────────────────────────────────────────────────

def judge_faithfulness(judge: ChatOpenAI, post: str, context: str) -> tuple[float, list[dict]]:
    """Claim decomposition: extract claims then verify each against context."""
    resp = judge.invoke(_CLAIM_EXTRACT_PROMPT.format(post=post))
    claims = json.loads(resp.content).get("claims", [])

    if not claims:
        return 1.0, []

    results = []
    for claim in claims:
        resp = judge.invoke(_CLAIM_VERIFY_PROMPT.format(context=context, claim=claim))
        supported = json.loads(resp.content).get("supported", False)
        results.append({"claim": claim, "supported": supported})

    score = sum(1 for r in results if r["supported"]) / len(results)
    return score, results


def judge_quality(judge: ChatOpenAI, user_input: str, post: str) -> dict:
    """Single pass: relevance + completeness."""
    resp = judge.invoke(_QUALITY_PROMPT.format(user_input=user_input, post=post))
    return json.loads(resp.content)


def judge_reference(judge: ChatOpenAI, post: str, style: str) -> dict:
    """Single pass: hook, style, hashtags, voice — reference only."""
    resp = judge.invoke(_REFERENCE_PROMPT.format(post=post, style=style))
    return json.loads(resp.content)


# ── Pipeline setup ─────────────────────────────────────────────────────────────

def setup_pipeline():
    from agents.pipeline import INDEX_NAME, MONGO_URI, DB_NAME, COLLECTION_NAME

    sync_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    posts_col = mongo[DB_NAME][COLLECTION_NAME]

    chunks_by_pid = {}
    for chunk in json.loads((RAG_DIR / "cleaned_chunks.json").read_text()):
        chunks_by_pid.setdefault(chunk["pid"], []).append(chunk)
    cleaned_posts = {p["pid"]: p for p in json.loads((RAG_DIR / "cleaned_posts.json").read_text())}

    print("Loading safety classifier...")
    safety_classifier = hf_pipeline("text-classification", model="KoalaAI/Text-Moderation")

    compiled = build_graph(
        sync_client, tavily_client, index, posts_col,
        chunks_by_pid, cleaned_posts,
        maps_api_key=maps_api_key,
        safety_classifier=safety_classifier,
    )
    return compiled


def make_initial_state(case: dict) -> dict:
    state = {
        "user_input": case["user_input"],
        "style": "",
        "location_info": {},
        "draft_content": "",
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
    # refinement cases can pre-seed draft_content, location_info, style, etc.
    state.update(case.get("initial_state_overrides", {}))
    return state


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(verbose: bool):
    if not os.environ.get("BRAINTRUST_API_KEY"):
        print("ERROR: BRAINTRUST_API_KEY is not set. Add it to .env before running.")
        sys.exit(1)

    cases = json.loads(CASES_PATH.read_text())
    print(f"Running {len(cases)} end-to-end cases...")

    compiled = setup_pipeline()
    judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0)

    experiment = braintrust.init(
        project="pmars",
        experiment="e2e-quality",
        api_key=os.environ.get("BRAINTRUST_API_KEY"),
    )

    any_hard_fail = False

    for case in cases:
        print(f"\n[{case['id']}] {case['category']} — running pipeline...")
        t0 = time.monotonic()

        try:
            state = await compiled.ainvoke(make_initial_state(case))
            pipeline_ms = round((time.monotonic() - t0) * 1000)
            final_post = state.get("final_post", "")
            pipeline_complete = 1.0 if final_post and not state.get("needs_clarification") else 0.0
        except Exception as e:
            pipeline_ms = round((time.monotonic() - t0) * 1000)
            final_post = ""
            pipeline_complete = 0.0
            print(f"  Pipeline error: {e}")

        if verbose:
            print(f"  pipeline_complete={pipeline_complete}  latency={pipeline_ms}ms")
            if final_post:
                print(f"  post preview: {final_post[:120]}...")

        faithfulness, faith_detail = 0.0, []
        relevance = completeness = 0.0
        hook_strength = style_match = hashtag_quality = personal_voice = 0.0
        quality_reasoning = reference_reasoning = ""
        retrieved_context = ""

        if pipeline_complete == 1.0:
            location_info = state.get("location_info", {})
            # research node splits context into style_context (RAG) and facts_context (web/URL/maps)
            retrieved_context = "\n\n".join(filter(None, [
                location_info.get("style_context", ""),
                location_info.get("facts_context", ""),
            ]))
            style = state.get("style", case.get("expected_style", "freeform"))

            faith_context = f"User's request:\n{case['user_input']}\n\nRetrieved context:\n{retrieved_context or 'None'}"
            # refinement cases: the seeded draft is also valid source material for faithfulness
            seeded_draft = case.get("initial_state_overrides", {}).get("draft_content", "")
            if seeded_draft:
                faith_context += f"\n\nOriginal draft (source material for refinement):\n{seeded_draft}"

            print("  Judging faithfulness (claim decomposition)...")
            faithfulness, faith_detail = judge_faithfulness(judge, final_post, faith_context)

            # judge_input_for_quality overrides user_input for relevance/completeness judge
            # when the raw user_input ("make it shorter") doesn't convey the original intent
            quality_user_input = case.get("judge_input_for_quality", case["user_input"])
            print("  Judging relevance + completeness...")
            quality = judge_quality(judge, quality_user_input, final_post)
            relevance = quality.get("relevance", 0.0)
            completeness = quality.get("completeness", 0.0)
            quality_reasoning = quality.get("reasoning", "")

            print("  Judging reference scores...")
            reference = judge_reference(judge, final_post, style)
            hook_strength = reference.get("hook_strength", 0.0)
            style_match = reference.get("style_match", 0.0)
            hashtag_quality = reference.get("hashtag_quality", 0.0)
            personal_voice = reference.get("personal_voice", 0.0)
            reference_reasoning = reference.get("reasoning", "")

            if verbose:
                print(f"  faithfulness={faithfulness:.2f}  relevance={relevance:.2f}  completeness={completeness:.2f}")
                print(f"  hook={hook_strength:.2f}  style={style_match:.2f}  hashtags={hashtag_quality:.2f}  voice={personal_voice:.2f}")
                if faith_detail:
                    unsupported = [r["claim"] for r in faith_detail if not r["supported"]]
                    if unsupported:
                        print(f"  unsupported claims: {unsupported}")

        if pipeline_complete < 1.0:
            any_hard_fail = True

        experiment.log(
            input={"user_input": case["user_input"]},
            output={
                "final_post": final_post,
                "style": state.get("style", "") if pipeline_complete else "",
            },
            expected={"expected_style": case.get("expected_style", "")},
            scores={
                "pipeline_complete": pipeline_complete,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "completeness": completeness,
                "hook_strength": hook_strength,
                "style_match": style_match,
                "hashtag_quality": hashtag_quality,
                "personal_voice": personal_voice,
            },
            tags=[case["category"]],
            metadata={
                "id": case["id"],
                "category": case["category"],
                "note": case.get("note", ""),
                "quality_reasoning": quality_reasoning,
                "reference_reasoning": reference_reasoning,
                "faith_detail": faith_detail,
                "retrieved_context_len": len(retrieved_context),
            },
            metrics={"pipeline_ms": pipeline_ms},
        )

    summary = experiment.summarize()
    print(summary)

    quality_scores = {k: summary.scores[k].score for k in ("faithfulness", "relevance", "completeness") if k in summary.scores}
    quality_fail = [k for k, v in quality_scores.items() if v < QUALITY_THRESHOLD]

    if any_hard_fail:
        print("\nFAIL: one or more cases had pipeline_complete=0")
        sys.exit(1)
    elif quality_fail:
        print(f"\nFAIL: experiment average below {QUALITY_THRESHOLD}: {quality_fail}")
        sys.exit(1)
    else:
        print("\nPASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(verbose=args.verbose))
