from pydantic import BaseModel

from schemas.chat import DebugInfo


class NewSessionResponse(BaseModel):
    session_id: str
    local_uid: str


class NewSessionRequest(BaseModel):
    draft_content: str = ""


class CompleteSessionRequest(BaseModel):
    final_content: str


class SessionSummary(BaseModel):
    session_id: str
    updated_at: str
    turn_count: int
    style: str
    preview: str
    completed: bool = False


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class SessionDetailResponse(BaseModel):
    session_id: str
    updated_at: str
    history: list[dict]
    debug: DebugInfo
