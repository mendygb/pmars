# End-to-End Eval — Placeholder

Full pipeline quality eval. Not yet implemented — design decision needed first.

## What this would test

Given a user request (place + description), does the final post meet quality criteria:
- Hook strength (first 2 lines grab attention)
- Factual accuracy (details match Google Maps / web results)
- Style match (matches detected style — checkin / recommendation / guide / diary)
- Hashtag relevance and count
- Length fits platform norms

## Approaches under consideration

**Option A: LLM-as-judge**
Run the pipeline, then send the output to GPT-4o with a structured rubric (score 1–5 on each criterion). Fast to implement, no human time required. Adds ~$0.01 per eval run. Risk: LLM judge can be inconsistent.

**Option B: Human eval with rubric**
Same rubric, graded by a human. Most reliable signal. Requires user time.

**Option C: Regression baseline**
Compare each run against a known-good baseline output (stored as fixture). Useful for catching regressions but doesn't measure absolute quality.

## Decision needed

Which approach (or combination) to implement. Once decided, update this file and implement in `run_eval.py`.
