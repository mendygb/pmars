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
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the Google Maps MCP server (requires Node.js):
   ```bash
   npm install -g @modelcontextprotocol/server-google-maps
   ```

4. The RAG pipeline expects posts to be stored in a database — this project uses a local MongoDB instance, but you can swap in any database that fits your setup.

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

A LangGraph-powered pipeline that coordinates 4 specialized agents to produce higher-quality posts.

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
| **SEO & Critic** | Reviews the draft for hook strength, clichés, and hashtags — outputs the final polished post. |

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
[Research] → [Copywriter] → [SEO & Critic]   ← first post
                  ↑
    Director skips agents that don't need to re-run on refinement turns:
    - tone/style change   → Copywriter → Critic
    - new detail/info     → Research → Copywriter → Critic
    - hashtag/hook fix    → Critic only
```

### Multi-turn Refinement

After a post is generated, describe any changes in plain language:
- `"make it funnier"` — rewrites tone
- `"add more tips about parking"` — re-retrieves and rewrites
- `"fix the hashtags"` — polishes tags only

Session transcripts are saved to `agents/outputs/`.

---

## Model Notes

All agents use `gpt-4o-mini` by default. Files are commented with `# UPGRADE:` where swapping to `gpt-4o` would improve quality (Director routing, Copywriter writing, Critic evaluation).
