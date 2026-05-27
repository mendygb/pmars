import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# Uvicorn configures its own loggers but leaves the root logger without a handler,
# so agent pipeline logs would silently drop. Attach a handler to the agents namespace.
_agent_logger = logging.getLogger("agents")
_agent_logger.setLevel(logging.INFO)
if not _agent_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)-8s %(name)s — %(message)s"))
    _agent_logger.addHandler(_h)

import firebase_admin
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from firebase_admin import auth, credentials
from openai import AsyncOpenAI, OpenAI
from pinecone import Pinecone
from pymongo import MongoClient
from pydantic import BaseModel
from tavily import TavilyClient
from transformers import pipeline as hf_pipeline

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.pipeline import build_graph
from agents.state import PostState
from db import (check_cancel, clear_cancel, complete_session, create_session,
                delete_session, get_session, get_session_detail, init_db,
                list_sessions, save_session, set_cancel)

# Keys that LangGraph knows about — strip everything else before passing state in
_POSTSTATE_KEYS = set(PostState.__annotations__.keys())

# ── Globals populated at startup ──────────────────────────────────────────────
compiled = None
USER_MAP: dict = {}
completed_sessions_col = None
injection_classifier = None


# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    user_input: str
    session_id: str
    debug: bool = False
    display_input: str = ""  # clean version saved to history; falls back to user_input


class NewSessionResponse(BaseModel):
    session_id: str
    local_uid: str


class HealthResponse(BaseModel):
    status: str


class SessionSummary(BaseModel):
    session_id: str
    updated_at: str
    turn_count: int
    style: str
    preview: str
    completed: bool = False


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class GenerateTitleRequest(BaseModel):
    content: str = ""
    title: str = ""


class NewSessionRequest(BaseModel):
    draft_content: str = ""


class CompleteSessionRequest(BaseModel):
    final_content: str


class DebugInfo(BaseModel):
    style: str
    next_node: str
    needs_clarification: bool
    draft_content: str
    final_post: str
    web_search_used: bool
    rag_metrics: dict
    timings: dict


class SessionDetailResponse(BaseModel):
    session_id: str
    updated_at: str
    history: list[dict]
    debug: DebugInfo


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled, USER_MAP, completed_sessions_col, injection_classifier

    # Firebase Admin SDK
    sa_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY_PATH", "./firebase-service-account.json")
    if os.path.exists(sa_path):
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)
    else:
        print(f"WARNING: Firebase service account not found at {sa_path!r}. Auth will be disabled.")

    # UID mapping (testing only — see README)
    if os.path.exists("users.json"):
        with open("users.json") as f:
            USER_MAP = json.load(f)
    else:
        print("WARNING: users.json not found. All authenticated requests will be rejected.")

    init_db(os.environ.get("REDIS_URL", "redis://localhost:6379"))

    # OpenAI — async client for pipeline nodes, sync client for RAG retrieve()
    async_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sync_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("pmars-social-posts")

    mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    posts_col = mongo_client["social_media_db"]["posts"]
    completed_sessions_col = mongo_client["social_media_db"]["completed_sessions"]

    rag_dir = os.path.join(os.path.dirname(__file__), "rag")
    with open(os.path.join(rag_dir, "cleaned_chunks.json")) as f:
        all_chunks = json.load(f)
    chunks_by_pid: dict = {}
    for c in all_chunks:
        chunks_by_pid.setdefault(c["pid"], []).append(c)
    with open(os.path.join(rag_dir, "cleaned_posts.json")) as f:
        cleaned_posts = {p["pid"]: p for p in json.load(f)}

    safety_classifier = hf_pipeline("text-classification", model="KoalaAI/Text-Moderation")
    injection_classifier = hf_pipeline("text-classification", model="fmops/distilbert-prompt-injection")

    compiled = build_graph(
        async_client, sync_client, tavily_client,
        index, posts_col, chunks_by_pid, cleaned_posts,
        maps_api_key=maps_api_key,
        safety_classifier=safety_classifier,
    )

    yield

    mongo_client.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth dependency ───────────────────────────────────────────────────────────
async def get_current_user(authorization: str = Header(...)) -> dict:
    if not firebase_admin._apps:
        raise HTTPException(503, "Firebase not initialized — add firebase-service-account.json")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        decoded = auth.verify_id_token(token)
    except Exception as e:
        raise HTTPException(401, f"Invalid token: {e}")
    user = USER_MAP.get(decoded["uid"])
    if not user:
        raise HTTPException(403, "Firebase UID not registered in users.json")
    return user


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/api/generate-title")
async def generate_title(req: GenerateTitleRequest, user: dict = Depends(get_current_user)):
    if not req.content and not req.title:
        raise HTTPException(400, "Provide content or title")
    if req.content:
        prompt = f"Write a short, catchy title (3-6 words) for this social media post. Output only the title, no quotes or punctuation:\n\n{req.content[:600]}"
    else:
        prompt = f"Rephrase this title to be more engaging. Keep it short (3-6 words). Output only the title, no quotes or punctuation:\n\n{req.title}"
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0.7,
    )
    return {"title": response.choices[0].message.content.strip().strip('"\'')}


@app.post("/api/sessions/new", response_model=NewSessionResponse)
async def new_session(user: dict = Depends(get_current_user), req: NewSessionRequest = None):
    draft = req.draft_content if req else ""
    session_id = await create_session(user["local_uid"], draft_content=draft)
    return NewSessionResponse(session_id=session_id, local_uid=user["local_uid"])


@app.post("/api/chat")
async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    # ── Gateway: prompt injection check ───────────────────────────────────────
    if injection_classifier is not None:
        text_to_check = req.display_input or req.user_input
        result = await asyncio.to_thread(injection_classifier, text_to_check)
        if result[0]["label"] == "LABEL_1" and result[0]["score"] > 0.85:
            async def blocked_stream():
                yield _sse({"type": "error", "payload": {"message": "⚠️ Your message was flagged as a prompt injection attempt and was not processed."}})
            return StreamingResponse(blocked_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    try:
        state = await get_session(req.session_id, user["local_uid"])
    except KeyError as e:
        raise HTTPException(404, str(e))

    state["user_input"] = req.user_input

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            is_refinement = bool(state.get("draft_content"))
            NODE_STATUS = {
                "director":   "💭 On it..."                   if is_refinement else "💭 Understanding your vibe...",
                "research":   "🔍 Digging deeper..."          if is_refinement else "🔍 Finding inspiration...",
                "copywriter": "✍️ Reworking your post..."     if is_refinement else "✍️ Writing your post...",
                "critic":     "✨ Polishing it up..."         if is_refinement else "✨ Adding the finishing touches...",
            }

            node_start_times: dict[str, float] = {}
            node_timings: dict[str, float] = {}
            pipeline_start = time.monotonic()
            # Track which nodes have had status/message emitted to avoid duplicates
            status_sent: set[str] = set()
            post_sent = False
            clarification_sent = False

            # Clear any stale cancel flag left over from a previous request on this session
            await clear_cancel(req.session_id)

            pipeline_input = {k: state[k] for k in _POSTSTATE_KEYS if k in state}
            cancelled = False
            async for ev in compiled.astream_events(pipeline_input, version="v2"):
                if await check_cancel(req.session_id):
                    await clear_cancel(req.session_id)
                    cancelled = True
                    break

                kind = ev["event"]
                name = ev.get("name", "")
                lg_node = ev.get("metadata", {}).get("langgraph_node", "")
                # Accept the registered node name from either field
                node_key = lg_node if lg_node in NODE_STATUS else (name if name in NODE_STATUS else "")

                if kind == "on_chain_start" and lg_node:
                    node_start_times[lg_node] = time.monotonic()
                    # Emit status bubble only for nodes that have a user-facing message
                    if node_key and node_key not in status_sent:
                        status_sent.add(node_key)
                        yield _sse({"type": "status", "message": NODE_STATUS[node_key]})

                # Critic uses ChatOpenAI → real streaming tokens
                elif kind == "on_chat_model_stream" and lg_node == "critic":
                    token = ev["data"]["chunk"].content
                    if token:
                        yield _sse({"type": "token", "content": token})

                # LLM done → commit the streamed bubble
                elif kind == "on_chat_model_end" and lg_node == "critic" and not post_sent:
                    post_sent = True
                    yield _sse({"type": "message_end"})

                # Process all LangGraph nodes — not gated on NODE_STATUS membership
                elif kind == "on_chain_end" and lg_node:
                    output = ev["data"].get("output", {})
                    if not isinstance(output, dict):
                        continue

                    if lg_node in node_start_times:
                        node_timings[lg_node] = round((time.monotonic() - node_start_times[lg_node]) * 1000)

                    state.update(output)

                    if req.debug:
                        yield _sse({"type": "debug", "node": lg_node, "payload": _sanitize(output)})

                    # Fallback for non-streaming nodes that produce a final_post
                    if output.get("final_post") and not post_sent:
                        post_sent = True
                        is_error = not state.get("safety_passed", True)
                        yield _sse({"type": "message", "content": output["final_post"], "is_error": is_error})

                    # Director asking for clarification
                    if output.get("needs_clarification") and not clarification_sent:
                        clarification_sent = True
                        yield _sse({"type": "clarification", "payload": {"question": output["clarification_question"]}})

            if cancelled:
                yield _sse({"type": "cancelled"})
                return  # skip save_session — DB stays at last clean state

            total_ms = round((time.monotonic() - pipeline_start) * 1000)
            state["_timings"] = {"total_ms": total_ms, **node_timings}

            history_user_content = req.display_input or req.user_input
            if state.get("needs_clarification"):
                state["history"].append({"role": "user", "content": history_user_content})
                state["history"].append({"role": "assistant", "content": state["clarification_question"]})
            elif state.get("final_post"):
                state["history"].append({"role": "user", "content": history_user_content})
                state["history"].append({"role": "assistant", "content": state["final_post"]})

            await save_session(req.session_id, user["local_uid"], state)
            yield _sse({"type": "done"})

        except Exception as e:
            yield _sse({"type": "error", "payload": {"message": str(e)}})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/sessions/{session_id}/cancel", status_code=204)
async def cancel_session(session_id: str, user: dict = Depends(get_current_user)):
    try:
        await get_session(session_id, user["local_uid"])
    except KeyError as e:
        raise HTTPException(404, str(e))
    await set_cancel(session_id)


@app.get("/api/sessions", response_model=SessionListResponse)
async def sessions_list(user: dict = Depends(get_current_user)):
    in_progress = [{**s, "completed": False} for s in await list_sessions(user["local_uid"])]

    completed = []
    if completed_sessions_col is not None:
        for doc in completed_sessions_col.find({"local_uid": user["local_uid"]}, sort=[("completed_at", -1)]):
            history = doc.get("history", [])
            user_turns = [t for t in history if t["role"] == "user"]
            if not user_turns:
                continue
            completed.append({
                "session_id": doc["session_id"],
                "updated_at": doc.get("completed_at", ""),
                "turn_count": len(user_turns),
                "style": doc.get("style", ""),
                "preview": user_turns[0]["content"][:80],
                "completed": True,
            })

    all_sessions = sorted(in_progress + completed, key=lambda s: s["updated_at"], reverse=True)
    return SessionListResponse(sessions=all_sessions)


@app.delete("/api/sessions/{session_id}", status_code=204)
async def session_delete(session_id: str, user: dict = Depends(get_current_user)):
    await delete_session(session_id, user["local_uid"])


@app.delete("/api/sessions/{session_id}/completed", status_code=204)
async def session_delete_completed(session_id: str, user: dict = Depends(get_current_user)):
    if completed_sessions_col is None:
        raise HTTPException(503, "MongoDB not available")
    result = completed_sessions_col.delete_one({"session_id": session_id, "local_uid": user["local_uid"]})
    if result.deleted_count == 0:
        raise HTTPException(404, f"Completed session {session_id!r} not found")


@app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
async def session_detail(session_id: str, user: dict = Depends(get_current_user)):
    try:
        detail = await get_session_detail(session_id, user["local_uid"])
    except KeyError:
        # Fall back to MongoDB for completed (applied) sessions
        try:
            doc = completed_sessions_col.find_one({"session_id": session_id, "local_uid": user["local_uid"]}) if completed_sessions_col is not None else None
        except Exception as e:
            raise HTTPException(500, f"MongoDB lookup failed: {e}")
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


@app.post("/api/sessions/{session_id}/complete", status_code=204)
async def session_complete(session_id: str, req: CompleteSessionRequest, user: dict = Depends(get_current_user)):
    if completed_sessions_col is None:
        raise HTTPException(503, "MongoDB not available")
    try:
        state = await get_session(session_id, user["local_uid"])
    except KeyError as e:
        raise HTTPException(404, str(e))
    await complete_session(session_id, user["local_uid"], req.final_content, state, completed_sessions_col)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sanitize(updates: dict) -> dict:
    """Strip large blobs from debug payloads to keep SSE events readable."""
    skip = {"docs", "style_context", "facts_context", "draft_content"}
    return {k: v for k, v in updates.items() if k not in skip}
