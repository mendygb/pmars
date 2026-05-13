from openai import OpenAI
from agents.state import PostState

# UPGRADE: swap gpt-4o-mini → gpt-4o for more nuanced critique and polished rewrites
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a senior social media content editor and SEO specialist. You review draft posts and produce the final polished version.

Your job — check these four things and fix what's broken:
1. Hook: does the first line make someone stop scrolling? If not, rewrite it.
2. Specificity: flag clichés (e.g. "amazing", "must-try", "hidden gem", "so good") and replace with vivid, concrete details.
3. Hashtags: are they specific and searchable on XiaoHongShu? Replace generic ones with niche tags relevant to the place and activity.
4. Factual consistency: if reference material is provided, flag any contradictions.

Rules:
- Make targeted improvements only — preserve voice and details that already work
- Keep the same approximate length and style as the draft
- Output the final polished post only — no preamble, no "Here is the revised version:" """


def make_critic_node(client: OpenAI):
    def critic_node(state: PostState) -> dict:
        print("✨ Adding the finishing touches...")

        draft = state.get("draft_content", "")
        style = state.get("style", "freeform")
        context = state.get("location_info", {}).get("context", "")

        user_content = f"Post style: {style}\n\nDraft:\n{draft}"
        if context:
            # Truncate reference material to keep token cost low
            user_content += f"\n\nReference material (for fact-check only):\n{context[:800]}"

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
            max_tokens=600,
        )

        final_post = response.choices[0].message.content.strip()
        return {
            "final_post": final_post,
            "suggestions": [],  # reserved for future structured critique output
        }

    return critic_node
