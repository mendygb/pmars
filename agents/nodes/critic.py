import time
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import PostState

# UPGRADE: swap gpt-4o-mini → gpt-4o for more nuanced critique and polished rewrites
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a senior social media content editor and SEO specialist. You review draft posts and produce the final polished version.

Your job — check these four things and fix what's broken:
1. Hook: does the first line make someone stop scrolling? If not, rewrite it.
2. Specificity: flag clichés (e.g. "amazing", "must-try", "hidden gem", "so good") and replace with vivid, concrete details.
3. Hashtags: are they specific and searchable? Replace generic ones with niche tags relevant to the place and activity.
4. Factual consistency: if reference material is provided, flag any contradictions.

Rules:
- Make targeted improvements only — preserve voice and details that already work
- Keep the same approximate length and style as the draft
- Output the final polished post only — no preamble, no "Here is the revised version:" """


def make_critic_node(client: AsyncOpenAI, debug=False):
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0.4,
        max_tokens=600,
        streaming=True,
        api_key=client.api_key,
    )

    async def critic_node(state: PostState) -> dict:
        print("✨ Adding the finishing touches...")

        draft = state.get("draft_content", "")
        style = state.get("style", "freeform")
        loc = state.get("location_info", {})
        facts_context = loc.get("facts_context", "")

        user_content = f"Post style: {style}\n\nDraft:\n{draft}"
        if facts_context:
            user_content += f"\n\nPlace details (for fact-check only):\n{facts_context[:800]}"

        t0 = time.time()
        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])
        llm_ms = int((time.time() - t0) * 1000)

        if debug:
            print(f"\n── Critic ───────────────────────────────────────")
            print(f"  LLM (review):   {llm_ms:>6} ms")
            print(f"─────────────────────────────────────────────────\n")

        final_post = response.content.strip()
        return {
            "final_post": final_post,
            "suggestions": [],
        }

    return critic_node
