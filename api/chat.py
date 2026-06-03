from fastapi import APIRouter, Depends, HTTPException, Request

from core.security import get_current_user
from db.redis_store import get_session, set_cancel
from schemas.chat import ChatRequest
from services.chat_service import run_chat_stream

router = APIRouter()


@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request, user: dict = Depends(get_current_user)):
    return await run_chat_stream(req, user, request.app.state.graph)


@router.post("/api/sessions/{session_id}/cancel", status_code=204)
async def cancel_session(session_id: str, user: dict = Depends(get_current_user)):
    try:
        await get_session(session_id, user["local_uid"])
    except KeyError as e:
        raise HTTPException(404, str(e))
    await set_cancel(session_id)
