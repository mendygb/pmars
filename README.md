# pmars

Turn your experiences into engaging social media posts — powered by RAG and a multi-agent pipeline.

---

## Overview

Two main components:

- **`rag/`** — Data pipeline: ingests, cleans, embeds, and indexes Bay Area posts for retrieval
- **`agents/`** — Multi-agent post writer: uses the RAG index to generate polished, style-aware posts

---

## Setup

1. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_key
   PINECONE_API_KEY=your_key
   TAVILY_API_KEY=your_key
   GOOGLE_MAPS_API_KEY=your_key
   REDIS_URL=redis://localhost:6379
   ```

2. Start Redis (requires Docker):
   ```bash
   docker compose up -d
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the Google Maps MCP server (requires Node.js):
   ```bash
   npm install -g @modelcontextprotocol/server-google-maps
   ```

5. The RAG pipeline expects posts to be stored in a database — this project uses a local MongoDB instance, but you can swap in any database that fits your setup.

---

## Web App

A browser UI on top of the agent pipeline — Firebase Auth, FastAPI backend (SSE streaming), React frontend.

### Firebase setup (one-time)

1. Create a project at [console.firebase.google.com](https://console.firebase.google.com)
2. Authentication → Sign-in methods → enable **Google** and **Email/Password**
3. Project Settings → General → Add web app → copy config → create `frontend/.env`:
   ```
   VITE_FIREBASE_API_KEY=
   VITE_FIREBASE_AUTH_DOMAIN=
   VITE_FIREBASE_PROJECT_ID=
   VITE_FIREBASE_APP_ID=
   ```
   (Use `frontend/.env.example` as a template)
4. Project Settings → Service Accounts → Generate new private key → save as `firebase-service-account.json` in the project root

### User mapping

Create `users.json` in the project root (gitignored — do not commit):
```json
{
  "FIREBASE_UID_HERE": {
    "local_uid": "user_001",
    "display_name": "Alice",
    "author_name": null
  }
}
```
Find your Firebase UID in the Firebase Console → Authentication → Users. Only UIDs listed here can access the app.

> **Testing only.** `users.json` is a manual whitelist for local development. In production, replace this lookup with a real user database.

### Running

```bash
# Terminal 1 — Backend (port 8000)
python -m uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend (port 5173)
cd frontend && npm install && npm run dev
```

Open [http://localhost:5173](http://localhost:5173) → sign in → fill in location/title/content → click **Write with AI** → chat with the pipeline → click **Apply** to bring the draft back to the editing page.

### Architecture

- **`main.py`** — App factory: lifespan startup (builds LangGraph, loads Safety Check model), CORS, router registration
- **`core/`** — Config (pydantic-settings reads `.env`), Firebase auth, logging setup
- **`api/`** — Route handlers: `chat.py` (SSE stream + cancel), `sessions.py` (session CRUD), `title.py` (title generation), `health.py`
- **`schemas/`** — Pydantic request/response models
- **`services/`** — Business logic: `chat_service.py` (GPT-based injection check + pipeline orchestration), `title_service.py`
- **`db/redis_store.py`** — Redis session storage (async, 30-day TTL): full pipeline state per session; cancel flags use a 5-minute TTL key
- **`db/mongo_store.py`** — MongoDB: `posts` collection (RAG tag search), `completed_sessions` (archived on Apply), `audit_logs` (per-request token usage, latency, and context — written after every pipeline run for billing and monitoring)
- **`docker-compose.yml`** — Starts a local Redis 7 instance with RDB persistence on a named volume
- **`frontend/`** — React (Vite): Firebase Auth, EditingPage (location/title/content form with AI title generation), Chat sub-view (auto-send, Apply button per draft, Stop button to cancel mid-stream), dev history page with debug panel (shows both in-progress and applied sessions)

---

## RAG Pipeline (`rag/`)

Three scripts that build the retrieval index. Run them once in order. The corpus used here (24 Bay Area lifestyle posts) is an example — swap in your own posts to tailor the pipeline to any niche.

```bash
python rag/01_ingest_and_clean.py    # fetch from MongoDB, scrub PII, chunk text
python rag/02_embed_and_index.py     # embed chunks, upsert to Pinecone
python rag/03_rag_query.py           # interactive single-LLM post writer (baseline)
```

**How it works:**
- Posts are fetched from MongoDB, PII-scrubbed (emails, phones, @mentions, URLs), and split into 500-char chunks with 50-char overlap
- Chunks are embedded with `text-embedding-3-small` and stored in Pinecone (`pmars-social-posts`, cosine, 1536-dim)
- At query time, hybrid retrieval combines Pinecone dense search + MongoDB tag search, merged with Reciprocal Rank Fusion (RRF)
- Top-4 full posts (not chunks) are passed to the LLM as context

**Output files:**
- `rag/cleaned_chunks.json` — 36 chunks used for embedding and indexing
- `rag/cleaned_posts.json` — 24 full posts used as LLM context (source of truth)
- `rag/outputs/` — saved session transcripts

---

## Multi-Agent Pipeline (`agents/`)

A LangGraph-powered pipeline that coordinates 5 specialized agents to produce higher-quality posts.

```bash
python agents/pipeline.py

# debug mode — prints per-agent timing, tool calls, and RAG metrics
python agents/pipeline.py --debug
```

### Agents

| Agent | Role |
|---|---|
| **Director** | Entry point for every turn. Classifies post style, detects if more info is needed, and routes to the right agents. On refinement turns, decides which agents to re-run. |
| **Research** | LLM-driven tool selection agent. Picks and runs the right tools in parallel, then merges results. |
| **Copywriter** | Writes the draft using style constraints, place facts, and style references. |
| **Safety Check** | Fast pass/fail content classification (local HuggingFace model). Blocks unsafe drafts before streaming starts. |
| **Critic** | Reviews the draft for hook strength, clichés, and hashtags — streams the final polished post via SSE. |

### Research Tools

The Research agent selects from four tools and runs them in parallel:

| Tool | Source | Used when |
|---|---|---|
| `retrieve_rag` | Pinecone + MongoDB (hybrid) | Bay Area place or style reference needed |
| `search_web` | Tavily | Non-Bay Area place or current details needed |
| `fetch_url` | MCP (`mcp-server-fetch`) | User provides a URL |
| `get_place_details` | MCP (`@modelcontextprotocol/server-google-maps`) | Specific named place — fetches rating, hours, reviews |

### Post Styles

The Director automatically picks the best format based on your description:

| Style | Chinese | Description |
|---|---|---|
| `checkin` | 打卡 | Short, visual, "I was here" |
| `recommendation` | 种草 | Persuasive, creates desire for a highlight |
| `guide` | 攻略 | Practical tips, saves reader effort |
| `diary` | 日记 | Narrative, emotional, personal arc |
| `freeform` | 自由发挥 | Fallback when no style clearly fits |

### Graph Flow

```
User input
    ↓
[Director] — classifies style, decides routing
    ↓
[Research] → [Copywriter] → [Safety Check] → [Critic]   ← first post
                  ↑
    Director skips agents that don't need to re-run on refinement turns:
    - tone/style change   → Copywriter → Safety Check → Critic
    - new detail/info     → Research → Copywriter → Safety Check → Critic
    - hashtag/hook fix    → Critic only
```

### Multi-turn Refinement

After a post is generated, describe any changes in plain language:
- `"make it funnier"` — rewrites tone
- `"add more tips about parking"` — re-retrieves and rewrites
- `"fix the hashtags"` — polishes tags only

Session transcripts are saved to `agents/outputs/`.

---

## Offline Eval (`eval/`)

Four-layer eval framework for offline quality checks. All evals require `BRAINTRUST_API_KEY` set in `.env` — results are logged to the `pmars` Braintrust project with automatic experiment diffs.

```bash
# Director routing accuracy (OPENAI_API_KEY)
python eval/director/run_eval.py

# Retrieval recall@4 + MRR (Pinecone + MongoDB + OPENAI_API_KEY)
python eval/retrieval/run_eval.py

# End-to-end quality — LLM-as-judge across 5 pipeline paths (all services)
python eval/end_to_end/run_eval.py

# Safety — injection classifier (OPENAI_API_KEY)
python eval/safety/run_injection_eval.py

# Safety — content safety check (downloads KoalaAI model ~400MB on first run)
python eval/safety/run_safety_check_eval.py
```

| Layer | What it tests | Threshold |
|---|---|---|
| **Director** | Routing accuracy across 19 golden cases | 85% `next_node` accuracy |
| **Retrieval** | Hybrid recall (Pinecone + MongoDB tag search + RRF) | 80% recall@4; MRR reference only |
| **End-to-end** | Full pipeline quality via LLM-as-judge (5 cases: RAG, web, URL, Google Maps, refinement turn) | `pipeline_complete`=1.0 per case; faithfulness/relevance/completeness avg ≥ 0.6 |
| **Safety** | Injection classifier + HuggingFace content check | 90% accuracy; 0% false positive rate on each |

See `eval/EVAL_NOTES.md` for design rationale and known limitations.

---

## Model Notes

All agents default to `gpt-4o-mini`. Each agent's model is independently configurable via `.env` — set any of the following to swap a specific agent without touching code:

```
DIRECTOR_MODEL=gpt-4o-mini
RESEARCH_MODEL=gpt-4o-mini
COPYWRITER_MODEL=gpt-4o
CRITIC_MODEL=gpt-4o
TITLE_MODEL=gpt-4o-mini
```

Copywriter and Critic are the quality-sensitive agents most likely to benefit from upgrading to `gpt-4o`.
