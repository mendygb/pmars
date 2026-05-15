import time
from openai import OpenAI
from agents.state import PostState

# UPGRADE: swap gpt-4o-mini → gpt-4o for richer, more platform-native writing
MODEL = "gpt-4o-mini"

# Style-specific writing rules injected into the system prompt per post format
STYLE_GUIDELINES = {
    # checkin (打卡): short, visual, "I was here"
    "checkin": (
        "Write a short check-in post (60–100 words). "
        "Open with a location declaration or arrival statement. "
        "Lean into the visual — describe what the camera would capture. "
        "End with a question to invite comments."
    ),
    # recommendation (种草): persuasive, creates desire
    "recommendation": (
        "Write a recommendation post (100–160 words). "
        "Open with a hook that names the standout item immediately. "
        "Use sensory language — taste, texture, smell, sight. "
        "Include one concrete comparison or superlative to create desire."
    ),
    # guide (攻略): practical, tips-first
    "guide": (
        "Write a practical guide post (120–180 words). "
        "Lead with the single most useful tip upfront. "
        "Include 2–3 numbered tips in the body. "
        "Close with a logistics line (price, hours, or booking note)."
    ),
    # diary (日记): narrative, emotional, personal arc
    "diary": (
        "Write a personal diary post (120–180 words). "
        "Open with the emotional state or time of day to set the scene. "
        "Include at least one specific sensory detail. "
        "End with a reflection or feeling, not a tip."
    ),
    # freeform (自由发挥): no rigid constraints
    "freeform": (
        "Write an engaging social media post (80–160 words). "
        "Choose the structure that best fits the content."
    ),
}

BASE_SYSTEM_PROMPT = """You are a social media copywriter specializing in travel and lifestyle content. You write engaging, authentic posts about places people have visited.

General rules (always apply):
- Write in first person ("I", "we")
- Be specific — name the item, moment, or feeling; never use filler phrases like "amazing experience" or "hidden gem"
- Include 5–8 relevant hashtags at the end (mix of location, activity, niche, and mood tags)
- Use emojis sparingly — one per key idea at most, never as decoration
- Output the post only — no preamble like "Here is your post:"
- IMPORTANT: if the user explicitly requests a specific format (e.g. "poem", "haiku", "bullet list", "rhyme"), honor that format above all other style guidelines below

Style-specific rules for this post (override with user's explicit format request if one exists):
{style_guide}"""


def make_copywriter_node(client: OpenAI, debug=False):
    def copywriter_node(state: PostState) -> dict:
        print("✍️  Writing your post...")

        style = state.get("style", "freeform")
        style_guide = STYLE_GUIDELINES.get(style, STYLE_GUIDELINES["freeform"])
        system_prompt = BASE_SYSTEM_PROMPT.format(style_guide=style_guide)

        is_refinement = bool(state.get("draft_content"))
        loc = state.get("location_info", {})
        # Style reference (RAG posts) — capped to limit input tokens
        style_context = loc.get("style_context", "")[:1500]
        # Factual data (Google Maps, web search, fetched URL) — sent in full
        facts_context = loc.get("facts_context", "")

        if is_refinement:
            # Targeted rewrite — touch only what the user asked to change
            user_content = (
                f"Current draft:\n{state['draft_content']}\n\n"
                f"User's change request: {state['user_input']}\n\n"
                "Apply ONLY the requested changes. "
                "Preserve voice, specific details, and hashtags unless asked to change them."
            )
        else:
            # First draft — factual context first, then style reference
            user_content = f"User's experience: {state['user_input']}"

            if facts_context:
                user_content = f"Place details:\n\n{facts_context}\n\n---\n\n{user_content}"

            if style_context:
                user_content = (
                    f"Reference posts from similar places:\n\n{style_context}\n\n"
                    f"---\n\n{user_content}"
                )

            # Include prior conversation turns (e.g. director clarification Q&A)
            if state.get("history"):
                recent = state["history"][-4:]
                qa_lines = "\n".join(
                    f"{'Director' if t['role'] == 'assistant' else 'User'}: {t['content']}"
                    for t in recent
                )
                user_content = f"Conversation context:\n{qa_lines}\n\n{user_content}"

        t0 = time.time()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.9,  # high — creative latitude for writing
            max_tokens=600,
        )
        llm_ms = int((time.time() - t0) * 1000)

        if debug:
            print(f"\n── Copywriter ───────────────────────────────────")
            print(f"  LLM (writing):  {llm_ms:>6} ms")
            print(f"─────────────────────────────────────────────────\n")

        draft = response.choices[0].message.content.strip()
        return {"draft_content": draft}

    return copywriter_node
