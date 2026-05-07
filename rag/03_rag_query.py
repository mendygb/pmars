import os
import re
import json
import time
import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from pymongo import MongoClient
from transformers import pipeline as hf_pipeline

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Retrieval constants ────────────────────────────────────────────────────────
INDEX_NAME = "pmars-social-posts"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.8
MAX_TOKENS = 600
TOP_K_VECTOR = 5    # candidates from Pinecone dense search
TOP_K_TAG = 5       # candidates from MongoDB tag search
TOP_N_FINAL = 4     # posts passed to LLM after RRF re-ranking
RRF_K = 60          # RRF constant — higher value flattens score differences between ranks

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "social_media_db"
COLLECTION_NAME = "posts"

# ── Intent classification constants ───────────────────────────────────────────
# Local NLI model used as the fast first-pass classifier.
# In production this is called a "neural router" — a lightweight local model that
# sits in front of expensive LLM calls and routes traffic cheaply (~50ms, free).
INTENT_MODEL = "cross-encoder/nli-deberta-v3-small"
INTENT_THRESHOLD = 0.75   # below this, fall back to LLM for accurate classification
INTENT_LABELS = [
    "the user wants to write a brand new social media post about a place they visited",
    "the user wants to edit, improve, or add details to the post that was just written",
]

# All unique tags in the corpus, used to detect relevant keywords in user queries
KNOWN_TAGS = [
    "Aesthetic", "BayArea", "CDM", "California", "Dining", "Dinner",
    "Family", "Filoli", "Food", "Foodie", "Fremont", "Fun", "GreatAmerica",
    "Greek", "Guide", "HappyHollow", "Hiking", "History", "LosAltos",
    "Luxury", "MissionPeak", "MountainView", "MuirWoods", "Nature",
    "PaloAlto", "Pier39", "Reflections", "SanFrancisco", "SanJose",
    "SantaClara", "Shoreline", "Taverna", "Thai", "Tips", "Tourist",
    "TravelTips", "Woodside", "Zoo",
]

# Normalize tags by stripping spaces and lowercasing so camelCase tags like
# "MuirWoods" can match user input written as "Muir Woods"
TAG_LOOKUP = {re.sub(r"\s+", "", t).lower(): t for t in KNOWN_TAGS}

SYSTEM_PROMPT = """You are a social media copywriter specializing in travel and lifestyle content.

You have two modes — use context to decide which applies:
1. Writing mode: when the user describes a place they visited, write one fresh, engaging post using the reference posts as inspiration for tone, style, and detail. Do NOT copy sentences verbatim.
2. Refinement mode: when the user asks to adjust, shorten, expand, change tone, or discuss the post, use the conversation history to refine.

Post writing guidelines:
- Length: 3–6 sentences (~80–180 words)
- Tone: warm, personal, and specific — avoid generic travel clichés
- Write in first person ("We discovered...", "I couldn't believe...")
- Highlight one specific detail (food, view, activity, hidden gem) that makes the place memorable
- Include 4–6 relevant hashtags at the end
- Use emojis sparingly (one per key idea at most)"""


def classify_intent(classifier, client, user_query, has_prior_context):
    """
    Classify user intent as 'new_query' or 'refinement'.

    Flow:
      - First turn always → 'new_query' (nothing to refine yet)
      - Local NLI model (INTENT_MODEL) for fast classification (~50ms, free)
      - If confidence < INTENT_THRESHOLD → fall back to gpt-4o-mini for accuracy

    Returns (intent, confidence, method, elapsed_ms).
    """
    if not has_prior_context:
        return "new_query", 1.0, "no_prior_context", 0

    t = time.time()
    result = classifier(user_query, candidate_labels=INTENT_LABELS)
    elapsed_ms = round((time.time() - t) * 1000)

    top_label = result["labels"][0]
    top_score = result["scores"][0]
    local_intent = "new_query" if "new place" in top_label else "refinement"

    if top_score >= INTENT_THRESHOLD:
        return local_intent, top_score, "local_model", elapsed_ms

    # Confidence too low — fall back to LLM for accurate classification of ambiguous input
    t_llm = time.time()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user message as exactly one of: 'new_query' (describing a new place "
                    "or experience to write about) or 'refinement' (adjusting, shortening, changing tone, "
                    "or otherwise editing an existing post). Reply with only the label."
                ),
            },
            {"role": "user", "content": user_query},
        ],
        max_tokens=10,
        temperature=0,
    )
    label = response.choices[0].message.content.strip().lower()
    llm_intent = "new_query" if "new" in label else "refinement"
    elapsed_ms += round((time.time() - t_llm) * 1000)

    return llm_intent, top_score, "llm_fallback", elapsed_ms


def embed_query(client, text):
    """Embed the user query into a dense vector for Pinecone similarity search."""
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def extract_tags(query):
    """
    Detect corpus tags mentioned in the user query.
    Normalizes both sides (remove spaces, lowercase) so "Muir Woods" matches tag "MuirWoods".
    """
    query_norm = re.sub(r"\s+", "", query).lower()
    return [orig for norm, orig in TAG_LOOKUP.items() if norm in query_norm]


def mongo_tag_search(posts_col, chunks_by_pid, matched_tags, top_k):
    """
    Sparse retrieval leg: query MongoDB for posts whose tags overlap with the query tags.
    Mimics an Elasticsearch keyword search — exact tag match, ranked by recency.
    Returns one chunk per post — only the pid matters here since RRF resolves
    every result back to its full document afterwards.
    """
    if not matched_tags:
        return []
    docs = list(
        posts_col.find({"tags": {"$in": matched_tags}}, {"pid": 1, "_id": 0})
        .sort("created", -1)
        .limit(top_k)
    )
    return [
        chunks_by_pid[doc["pid"]][0]
        for doc in docs
        if doc["pid"] in chunks_by_pid
    ]


def rrf_merge(vector_matches, tag_chunks, top_n, k=RRF_K):
    """
    Reciprocal Rank Fusion: combine two independently ranked lists into one.

    Formula: rrf_score(doc) = Σ  1 / (k + rank(doc, list))
                                 for each list where doc appears

    A document appearing in both lists scores higher than one appearing in only one.
    k=60 is the standard default — it dampens the impact of rank position differences.

    vector_matches: Pinecone result dicts, ranked by cosine similarity (rank 1 = best)
    tag_chunks:     chunk dicts from MongoDB, ranked by post recency (rank 1 = newest)

    Returns a ranked list of pids — callers expand each pid to its full document.
    """
    scores = {}  # pid → cumulative RRF score

    # Dense retrieval contribution (chunk-level, but scored at post level via pid)
    for rank, match in enumerate(vector_matches, 1):
        pid = match["metadata"]["pid"]
        scores[pid] = scores.get(pid, 0) + 1 / (k + rank)

    # Sparse (tag) retrieval contribution
    for rank, chunk in enumerate(tag_chunks, 1):
        pid = chunk["pid"]
        scores[pid] = scores.get(pid, 0) + 1 / (k + rank)

    return sorted(scores, key=lambda p: scores[p], reverse=True)[:top_n]


def build_context(docs):
    """Format retrieved full posts as labeled blocks to inject into the LLM prompt."""
    blocks = [f"[Post {i}: {d['title']}]\n{d['cleaned_text']}" for i, d in enumerate(docs, 1)]
    return "\n\n".join(blocks)


def format_metrics(metrics):
    """Render per-query timing, retrieval stats, and hit details as a readable block."""
    lines = [
        "── Metrics ──────────────────────────────────────────",
        f"  Intent:           {metrics['intent']:<14}"
        f"({metrics['intent_confidence']:.2f} conf, {metrics['intent_method']}, {metrics['intent_ms']} ms)",
    ]

    if metrics.get("retrieval_skipped"):
        lines.append(f"  Retrieval:          skipped — reusing {len(metrics['pinned_posts'])} pinned posts")
        for p in metrics["pinned_posts"]:
            lines.append(f"      {p['pid']:<24}  {p['title']}")
    else:
        tags_str = ", ".join(metrics["matched_tags"]) if metrics["matched_tags"] else "none"
        lines += [
            f"  Embed query:      {metrics['embed_ms']:>6} ms",
            f"  Vector search:    {metrics['vector_search_ms']:>6} ms  ({metrics['vector_hits']} hits)",
        ]
        for h in metrics["vector_hits_detail"]:
            lines.append(f"      [{h['score']:.4f}]  {h['chunk_id']:<20}  {h['title']}")

        lines.append(f"  Tag search:       {metrics['tag_search_ms']:>6} ms  ({metrics['tag_hits']} hits, tags: {tags_str})")
        for h in metrics["tag_hits_detail"]:
            lines.append(f"      {h['pid']:<24}  {h['title']}")

        lines.append(f"  RRF merge:        {metrics['rrf_ms']:>6} ms  ({metrics['final_count']} posts selected)")
        for p in metrics["final_posts"]:
            lines.append(f"      {p['pid']:<24}  {p['title']}")

    lines += [
        f"  LLM first token:  {metrics['llm_first_token_ms']:>6} ms",
        f"  LLM total:        {metrics['llm_total_ms']:>6} ms",
        "  ────────────────────────────────────────────────────",
        f"  Total:            {metrics['total_ms']:>6} ms",
        "─────────────────────────────────────────────────────",
    ]
    return "\n".join(lines)


def retrieve(index, client, posts_col, chunks_by_pid, cleaned_posts, user_query):
    """
    RAG retrieval pipeline (steps 1–4):
      1. Embed query            → dense vector
      2. Pinecone vector search → top-K semantically similar chunks
      3. MongoDB tag search     → top-K keyword/tag matched posts
      4. RRF merge              → unified ranked list, expanded to full posts
    Returns (final_docs, metrics_partial).
    """
    metrics = {}

    # ── Step 1: Embed the user query ──────────────────────────────────────────
    t = time.time()
    vector = embed_query(client, user_query)
    metrics["embed_ms"] = round((time.time() - t) * 1000)

    # ── Step 2: Dense retrieval — semantic similarity via Pinecone ────────────
    t = time.time()
    vector_results = index.query(vector=vector, top_k=TOP_K_VECTOR, include_metadata=True)
    metrics["vector_search_ms"] = round((time.time() - t) * 1000)
    vector_matches = vector_results.get("matches", [])
    metrics["vector_hits"] = len(vector_matches)
    metrics["vector_hits_detail"] = [
        {"chunk_id": m["id"], "score": round(m["score"], 4), "title": m["metadata"]["title"]}
        for m in vector_matches
    ]

    # ── Step 3: Sparse retrieval — tag/keyword match via MongoDB ──────────────
    t = time.time()
    matched_tags = extract_tags(user_query)
    tag_chunks = mongo_tag_search(posts_col, chunks_by_pid, matched_tags, TOP_K_TAG)
    metrics["tag_search_ms"] = round((time.time() - t) * 1000)
    metrics["matched_tags"] = matched_tags
    metrics["tag_hits"] = len(tag_chunks)
    metrics["tag_hits_detail"] = [{"pid": c["pid"], "title": c["title"]} for c in tag_chunks]

    # ── Step 4: RRF — re-rank and merge both result lists ─────────────────────
    t = time.time()
    ranked_pids = rrf_merge(vector_matches, tag_chunks, TOP_N_FINAL)
    # Expand each pid to its full cleaned post — no chunking overlap, original text preserved.
    # Retrieval happened at chunk granularity for precision; context is served at post
    # granularity from cleaned_posts so the LLM always sees the complete original post.
    final_docs = [cleaned_posts[pid] for pid in ranked_pids if pid in cleaned_posts]
    metrics["rrf_ms"] = round((time.time() - t) * 1000)
    metrics["final_count"] = len(final_docs)
    metrics["final_posts"] = [{"pid": d["pid"], "title": d["title"]} for d in final_docs]

    return final_docs, metrics


def chat_turn(messages, client, user_query, final_docs, metrics):
    """
    LLM generation step (step 5) for one conversation turn.
    Injects retrieved (or pinned) context into the user message, appends to the shared
    messages list to maintain conversation history, and streams the reply.
    Returns (reply_text, metrics_updated).
    """
    context = build_context(final_docs)
    user_content = (
        f"Reference posts from similar places:\n\n{context}\n\n"
        f"---\n\n{user_query}"
    )
    messages.append({"role": "user", "content": user_content})

    t = time.time()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    print("\nAssistant:\n")
    reply = ""
    first_token_recorded = False
    for chunk in response:
        token = chunk.choices[0].delta.content
        if token:
            if not first_token_recorded:
                metrics["llm_first_token_ms"] = round((time.time() - t) * 1000)
                first_token_recorded = True
            print(token, end="", flush=True)
            reply += token
    print("\n")

    # Append assistant reply to history so subsequent turns have full context
    messages.append({"role": "assistant", "content": reply})
    metrics["llm_total_ms"] = round((time.time() - t) * 1000)

    return reply, metrics


def save_transcript(transcript):
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    filename = datetime.datetime.now().strftime("rag_%Y%m%d_%H%M%S.txt")
    path = os.path.join(outputs_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        for user_query, reply, metrics in transcript:
            f.write(f"You: {user_query}\n\n")
            f.write(f"Assistant:\n{reply}\n\n")
            f.write(format_metrics(metrics) + "\n\n")
            f.write("=" * 53 + "\n\n")
    print(f"Transcript saved to {path}")


def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    # MongoDB connection for sparse tag retrieval
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    posts_col = mongo_client[DB_NAME][COLLECTION_NAME]

    # Load cleaned chunks into memory and build a pid → [chunks] index for tag search lookups
    chunks_path = os.path.join(os.path.dirname(__file__), "cleaned_chunks.json")
    with open(chunks_path, encoding="utf-8") as f:
        all_chunks = json.load(f)
    chunks_by_pid = {}
    for c in all_chunks:
        chunks_by_pid.setdefault(c["pid"], []).append(c)

    # Load full cleaned posts (no chunking, no overlap) as the source of truth for LLM context.
    # In production, replace this with a Redis-backed fetch to avoid re-reading from disk and
    # to handle corpus updates without restarting the process:
    #   redis_client = redis.Redis(host=..., port=6379)
    #   def get_post(pid):
    #       cached = redis_client.get(f"post:{pid}")
    #       return json.loads(cached) if cached else fetch_from_mongo_and_cache(pid)
    posts_path = os.path.join(os.path.dirname(__file__), "cleaned_posts.json")
    with open(posts_path, encoding="utf-8") as f:
        cleaned_posts = {p["pid"]: p for p in json.load(f)}

    # Load local NLI intent classifier once at startup (~few seconds, then ~50ms per call).
    # In production this model would be served via TorchServe or Triton behind an internal
    # endpoint so it stays warm and is shared across instances.
    print("Loading intent classifier...", end="", flush=True)
    classifier = hf_pipeline("zero-shot-classification", model=INTENT_MODEL)
    print(" ready.\n")

    # Conversation history shared across all turns — enables multi-turn refinement
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Pinned docs from the last new_query retrieval, reused on refinement turns so the
    # LLM always refines with the same reference material that inspired the original post
    pinned_docs = []

    print("RAG Post Writer — describe a place you visited, or refine the post in follow-up turns.")
    print("Type 'quit' or 'exit' to end the session.\n")

    transcript = []
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        t_start = time.time()

        # ── Intent classification ──────────────────────────────────────────────
        intent, confidence, method, intent_ms = classify_intent(
            classifier, client, user_input, has_prior_context=bool(pinned_docs)
        )

        if intent == "new_query":
            final_docs, metrics = retrieve(
                index, client, posts_col, chunks_by_pid, cleaned_posts, user_input
            )
            pinned_docs = final_docs   # pin for subsequent refinement turns
        else:
            # Reuse pinned docs — no retrieval cost, context stays grounded in same corpus slice
            final_docs = pinned_docs
            metrics = {
                "retrieval_skipped": True,
                "pinned_posts": [{"pid": d["pid"], "title": d["title"]} for d in pinned_docs],
            }

        metrics["intent"] = intent
        metrics["intent_confidence"] = round(confidence, 3)
        metrics["intent_method"] = method
        metrics["intent_ms"] = intent_ms

        reply, metrics = chat_turn(messages, client, user_input, final_docs, metrics)
        metrics["total_ms"] = round((time.time() - t_start) * 1000)

        print(format_metrics(metrics) + "\n")
        transcript.append((user_input, reply, metrics))

    if transcript:
        save_transcript(transcript)


if __name__ == "__main__":
    main()
