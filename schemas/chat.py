from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_input: str
    session_id: str
    debug: bool = False
    display_input: str = ""


class DebugInfo(BaseModel):
    style: str
    next_node: str
    needs_clarification: bool
    draft_content: str
    final_post: str
    web_search_used: bool
    rag_metrics: dict
    timings: dict


class GenerateTitleRequest(BaseModel):
    content: str = ""
    title: str = ""
