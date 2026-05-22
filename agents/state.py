from typing import TypedDict


class PostState(TypedDict):
    user_input: str
    # checkin (打卡) / recommendation (种草) / guide (攻略) / diary (日记) / freeform (自由发挥)
    style: str
    location_info: dict    # RAG retrieval results: {"docs": [...], "context": str, "metrics": {...}}
    draft_content: str     # Copywriter's output (internal — user never sees this)
    final_post: str        # SEO & Critic's output — shown to user
    suggestions: list      # Critic's internal notes (reserved for future use)
    history: list          # conversation turns: [{"role": "user"|"assistant", "content": str}]
    next_node: str         # Director's routing: "research" | "copywriter" | "critic" | "ask_user"
    needs_clarification: bool
    clarification_question: str
    safety_passed: bool
    media_id: str
    user_profile_injected: bool
