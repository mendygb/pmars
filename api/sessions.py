from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from core.security import get_current_user
from db import mongo_store
from db.redis_store import (create_session, delete_session, get_session,
                             get_session_detail, list_sessions)
from schemas.session import (CompleteSessionRequest, NewSessionRequest,
                              NewSessionResponse, SessionDetailResponse,
                              SessionListResponse)

router = APIRouter()


@router.post("/api/sessions/new", response_model=NewSessionResponse)
async def new_session(user: dict = Depends(get_current_user), req: NewSessionRequest = None):
    draft = req.draft_content if req else ""
    session_id = await create_session(user["local_uid"], draft_content=draft)
    return NewSessionResponse(session_id=session_id, local_uid=user["local_uid"])


@router.get("/api/sessions", response_model=SessionListResponse)
async def sessions_list(user: dict = Depends(get_current_user)):
    in_progress = [{**s, "completed": False} for s in await list_sessions(user["local_uid"])]
    completed = mongo_store.list_completed(user["local_uid"])
    all_sessions = sorted(in_progress + completed, key=lambda s: s["updated_at"], reverse=True)
    return SessionListResponse(sessions=all_sessions)


@router.delete("/api/sessions/{session_id}", status_code=204)
async def session_delete(session_id: str, user: dict = Depends(get_current_user)):
    await delete_session(session_id, user["local_uid"])


@router.delete("/api/sessions/{session_id}/completed", status_code=204)
async def session_delete_completed(session_id: str, user: dict = Depends(get_current_user)):
    count = mongo_store.delete_completed(session_id, user["local_uid"])
    if count == 0:
        raise HTTPException(404, f"Completed session {session_id!r} not found")


@router.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
async def session_detail(session_id: str, user: dict = Depends(get_current_user)):
    try:
        detail = await get_session_detail(session_id, user["local_uid"])
    except KeyError:
        doc = mongo_store.find_completed(session_id, user["local_uid"])
        if not doc:
            raise HTTPException(404, f"Session {session_id!r} not found")
        try:
            detail = {
                "session_id": session_id,
                "updated_at": doc.get("completed_at", ""),
                "history": doc.get("history", []),
                "debug": {
                    "style": doc.get("style", ""),
                    "next_node": "",
                    "needs_clarification": False,
                    "draft_content": "",
                    "final_post": doc.get("final_content", ""),
                    "web_search_used": False,
                    "rag_metrics": {},
                    "timings": {},
                },
            }
            return SessionDetailResponse(**detail)
        except Exception as e:
            raise HTTPException(500, f"Failed to build response: {e}")
    except Exception as e:
        raise HTTPException(500, f"Redis lookup failed: {e}")
    return SessionDetailResponse(**detail)


@router.post("/api/sessions/{session_id}/complete", status_code=204)
async def session_complete(session_id: str, req: CompleteSessionRequest, user: dict = Depends(get_current_user)):
    if not mongo_store.is_available():
        raise HTTPException(503, "MongoDB not available")
    try:
        state = await get_session(session_id, user["local_uid"])
    except KeyError as e:
        raise HTTPException(404, str(e))
    mongo_store.insert_completed({
        "session_id": session_id,
        "local_uid": user["local_uid"],
        "history": state.get("history", []),
        "style": state.get("style", ""),
        "final_content": req.final_content,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    await delete_session(session_id, user["local_uid"])
