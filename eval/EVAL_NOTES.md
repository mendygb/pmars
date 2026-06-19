# Eval Framework — Design Notes

Offline evaluation for the pmars AI writing pipeline. Each eval layer is independent and can be run separately. Results are logged to Braintrust (project: `pmars`).

---

## Structure

```
eval/
  director/
    golden_dataset.json     — (input, expected routing) pairs
    run_eval.py             — Director routing accuracy
  retrieval/
    test_queries.json       — (query, expected_pids) pairs
    run_eval.py             — hybrid retrieval recall@k + MRR
  end_to_end/
    test_cases.json         — 5 cases covering all research paths + refinement turn
    run_eval.py             — full pipeline with LLM-as-judge quality scoring
  safety/
    injection_cases.json    — 12 legit inputs + 8 injection attempts
    run_injection_eval.py   — injection classifier accuracy + false positive rate
    safety_check_cases.json — 4 safe drafts + 3 unsafe drafts
    run_safety_check_eval.py — HuggingFace content classifier accuracy + false positive rate
  EVAL_NOTES.md             — this file
```

---

## Layer 1: Director Routing Eval

**What it tests**: Given a user input + conversation state, does Director route to the right next node?

**Metric**: `next_node` accuracy. Threshold: **85%** (allows intentional ambiguity; `accepted_next_nodes` captures cases with multiple valid routes).

**Run**:
```bash
python eval/director/run_eval.py
```

---

## Layer 2: Retrieval Eval

**What it tests**: Does hybrid retrieval (Pinecone dense + MongoDB tag search + RRF) return the relevant posts for a query?

**Metrics**: Recall@k (CI gate, threshold: **80%**) + MRR (reference only — useful for tracking ranking degradation).

**Run**:
```bash
python eval/retrieval/run_eval.py
python eval/retrieval/run_eval.py --topk 4 --verbose
python eval/retrieval/run_eval.py --filter tag_based
```

---

## Layer 3: End-to-End Quality Eval

**What it tests**: Given a real user request, does the full pipeline produce a faithful, relevant, complete post?

**Test cases**: 5 cases covering all research paths + one refinement turn:
- `e001` bay_area_rag — RAG path (Pinecone + MongoDB tag search)
- `e002` non_bay_area_web — Tavily web search
- `e003` url_fetch — fetch_url tool
- `e004` google_maps_place — Google Maps MCP (`get_place_details`)
- `e005` refinement_turn — pre-seeded draft, copywriter-only routing path

**Metrics and thresholds**:
- `pipeline_complete`: per-case hard fail (any case = 0 → exit 1)
- `faithfulness`, `relevance`, `completeness`: experiment average >= **0.6** (0.6 accounts for LLM world knowledge use, not a sign of poor quality)
- `hook_strength`, `style_match`, `hashtag_quality`, `personal_voice`: reference only, no threshold

**Run**:
```bash
python eval/end_to_end/run_eval.py
python eval/end_to_end/run_eval.py --verbose
```

---

## Layer 4: Safety Eval

Two separate components, two separate evals.

### Injection Classifier (GPT-based, per API call, checks user input)

**What it tests**: `_is_injection()` in `services/chat_service.py` — does it correctly pass legit travel inputs and block injection attacks?

**Thresholds**:
- Accuracy >= **90%**
- False positive rate == **0%** (legit inputs must never be blocked)

**Run**:
```bash
python eval/safety/run_injection_eval.py
python eval/safety/run_injection_eval.py --verbose
```

### Content Safety Check (HuggingFace, checks LLM draft)

**What it tests**: `make_safety_check_node()` in `agents/nodes/safety_check.py` — does the KoalaAI/Text-Moderation classifier correctly pass safe travel posts and flag harmful content?

**Thresholds**:
- Accuracy >= **90%**
- False positive rate == **0%** (safe posts must not be blocked)

**Run**:
```bash
python eval/safety/run_safety_check_eval.py
python eval/safety/run_safety_check_eval.py --verbose
```

---

## Running All Evals

All evals require `BRAINTRUST_API_KEY` set in `.env`. Results log to the `pmars` project in Braintrust — each run creates a new experiment and diffs against the previous one automatically.

```bash
# Director routing (needs OPENAI_API_KEY)
python eval/director/run_eval.py

# Retrieval (needs Pinecone + MongoDB + OPENAI_API_KEY)
python eval/retrieval/run_eval.py

# End-to-end (needs all services + HuggingFace locally)
python eval/end_to_end/run_eval.py

# Safety — injection classifier (needs OPENAI_API_KEY)
python eval/safety/run_injection_eval.py

# Safety — content check (downloads KoalaAI model on first run, ~400MB)
python eval/safety/run_safety_check_eval.py
```

All scripts exit non-zero if any threshold is missed, making them suitable as a CI gate.

---

## Known Limitations / Prerequisites Before CI

- **Retrieval eval runs against the live Pinecone index.** If corpus changes, `expected_pids` can break without any logic regression. Fix: separate test namespace with a pinned snapshot.
- **E2e eval is non-deterministic.** Pipeline agents run at non-zero temperature. Quality scores vary run-to-run. Thresholds are calibrated for a 5-case set; more cases would smooth variance.
- **Safety check eval contains minimal harmful content** in `safety_check_cases.json` (necessary to verify the classifier works — standard safety ML practice). The content is for testing only; keep the file out of public-facing docs.
