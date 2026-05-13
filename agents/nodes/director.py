import json
from openai import OpenAI
from agents.state import PostState

# UPGRADE: swap gpt-4o-mini → gpt-4o for more accurate routing and style classification
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are the Director of a XiaoHongShu (小红书) post-writing pipeline. You coordinate specialized agents to help users write engaging social media posts about places they've visited.

## Post Styles
- checkin (打卡): short, visual, location-declaration — "I was here"
- recommendation (种草): persuasive, creates desire for a specific highlight (dish, view, activity)
- guide (攻略): practical tips, saves reader effort, numbered structure
- diary (日记): narrative, emotional, personal arc
- freeform (自由发挥): fallback when none of the four styles clearly fit

## Routing Rules

**First turn (no existing draft):**
- If the description gives you a place + at least one detail (a feeling, food, activity, or view) → route to research
- If the description is too vague (just a place name with no detail) → ask ONE short, targeted question to extract a memorable moment or vibe
- Pick the style that best fits the user's words
- IMPORTANT: only ask a clarifying question if there is truly not enough to write anything. If the history already contains a prior clarification exchange, do NOT ask again — proceed to writing regardless.

**Refinement turn (existing draft is present in context):**
- "change tone / style / language / mood / make it funnier / shorter / longer" → route to copywriter
- "add more info / details / tips about X / I also did Y" → route to research (will re-retrieve)
- "fix hashtags / title / hook" → route to critic
- Anything else unclear → route to copywriter (safest default)

**New post after refinement (existing draft present but user describes a brand-new experience or place):**
- Signals: "I went to X", "write about X", "not the post I want, I want one about X", "different place", describing somewhere that wasn't in the original request
- Set is_new_post to true AND next_node to "research" — this resets the draft and retrieves fresh context

## Output Format
Respond ONLY with a valid JSON object:
{
  "style": "checkin" | "recommendation" | "guide" | "diary" | "freeform",
  "next_node": "research" | "copywriter" | "critic" | "ask_user",
  "needs_clarification": true | false,
  "clarification_question": "<one short targeted question>" | null,
  "is_new_post": true | false
}"""


def make_director_node(client: OpenAI, debug=False):
    def director_node(state: PostState) -> dict:
        print("\n💭 Understanding your vibe...")

        has_draft = bool(state.get("draft_content"))

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Include conversation history so Director understands prior turns
        for turn in state.get("history", []):
            messages.append(turn)

        # On refinement turns, show the Director the current draft alongside the new request
        user_content = state["user_input"]
        if has_draft:
            user_content = (
                f"[Current draft]\n{state['draft_content']}\n\n"
                f"[User's request]\n{state['user_input']}"
            )

        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,  # low — routing should be stable and deterministic
        )

        decision = json.loads(response.choices[0].message.content)

        if debug:
            print(f"\n── Director decision ──────────────────────────────")
            print(f"  style:          {decision.get('style')}")
            print(f"  next_node:      {decision.get('next_node')}")
            print(f"  is_new_post:    {decision.get('is_new_post')}")
            print(f"  clarification:  {decision.get('needs_clarification')} → {decision.get('clarification_question')}")
            print(f"───────────────────────────────────────────────────\n")

        needs_clarification = decision.get("needs_clarification", False)

        updates = {
            "style": decision.get("style", "freeform"),
            # Force ask_user when clarification is needed — prevents running downstream agents simultaneously
            "next_node": "ask_user" if needs_clarification else decision.get("next_node", "research"),
            "needs_clarification": needs_clarification,
            "clarification_question": decision.get("clarification_question") or "",
        }

        # Reset draft and location if the user is starting a brand-new post
        if decision.get("is_new_post"):
            updates["draft_content"] = ""
            updates["location_info"] = {}

        return updates

    return director_node
