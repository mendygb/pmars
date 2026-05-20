import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

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

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.pipeline import build_graph
from agents.state import PostState
from db import create_session, delete_session, get_session, get_session_detail, init_db, list_sessions, save_session

# Keys that LangGraph knows about — strip everything else before passing state in
_POSTSTATE_KEYS = set(PostState.__annotations__.keys())

# ── Globals populated at startup ──────────────────────────────────────────────
compiled = None
USER_MAP: dict = {}


# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    user_input: str
    session_id: str
    debug: bool = False


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


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


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
    global compiled, USER_MAP

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

    init_db()

    # OpenAI — async client for pipeline nodes, sync client for RAG retrieve()
    async_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sync_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("pmars-social-posts")

    mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    posts_col = mongo_client["social_media_db"]["posts"]

    rag_dir = os.path.join(os.path.dirname(__file__), "rag")
    with open(os.path.join(rag_dir, "cleaned_chunks.json")) as f:
        all_chunks = json.load(f)
    chunks_by_pid: dict = {}
    for c in all_chunks:
        chunks_by_pid.setdefault(c["pid"], []).append(c)
    with open(os.path.join(rag_dir, "cleaned_posts.json")) as f:
        cleaned_posts = {p["pid"]: p for p in json.load(f)}

    compiled = build_graph(
        async_client, sync_client, tavily_client,
        index, posts_col, chunks_by_pid, cleaned_posts,
        maps_api_key=maps_api_key,
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


@app.post("/api/sessions/new", response_model=NewSessionResponse)
async def new_session(user: dict = Depends(get_current_user)):
    session_id = create_session(user["local_uid"])
    return NewSessionResponse(session_id=session_id, local_uid=user["local_uid"])


@app.post("/api/chat")
async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    try:
        state = get_session(req.session_id, user["local_uid"])
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

            pipeline_input = {k: state[k] for k in _POSTSTATE_KEYS if k in state}
            async for ev in compiled.astream_events(pipeline_input, version="v2"):
                kind = ev["event"]
                name = ev.get("name", "")
                lg_node = ev.get("metadata", {}).get("langgraph_node", "")
                # Accept the registered node name from either field
                node_key = lg_node if lg_node in NODE_STATUS else (name if name in NODE_STATUS else "")


                if kind == "on_chain_start" and node_key and node_key not in status_sent:
                    status_sent.add(node_key)
                    node_start_times[node_key] = time.monotonic()
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

                elif kind == "on_chain_end" and node_key:
                    output = ev["data"].get("output", {})
                    if not isinstance(output, dict):
                        continue

                    if node_key in node_start_times:
                        node_timings[node_key] = round((time.monotonic() - node_start_times[node_key]) * 1000)

                    state.update(output)

                    if req.debug:
                        yield _sse({"type": "debug", "node": node_key, "payload": _sanitize(output)})

                    # Fallback for non-streaming nodes that produce a final_post
                    if output.get("final_post") and not post_sent:
                        post_sent = True
                        yield _sse({"type": "message", "content": output["final_post"]})

                    # Director asking for clarification
                    if output.get("needs_clarification") and not clarification_sent:
                        clarification_sent = True
                        yield _sse({"type": "clarification", "payload": {"question": output["clarification_question"]}})

            total_ms = round((time.monotonic() - pipeline_start) * 1000)
            state["_timings"] = {"total_ms": total_ms, **node_timings}

            if state.get("needs_clarification"):
                state["history"].append({"role": "user", "content": req.user_input})
                state["history"].append({"role": "assistant", "content": state["clarification_question"]})
            elif state.get("final_post"):
                state["history"].append({"role": "user", "content": req.user_input})
                state["history"].append({"role": "assistant", "content": state["final_post"]})

            save_session(req.session_id, user["local_uid"], state)
            yield _sse({"type": "done"})

        except Exception as e:
            yield _sse({"type": "error", "payload": {"message": str(e)}})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sessions", response_model=SessionListResponse)
async def sessions_list(user: dict = Depends(get_current_user)):
    return SessionListResponse(sessions=list_sessions(user["local_uid"]))


@app.delete("/api/sessions/{session_id}", status_code=204)
async def session_delete(session_id: str, user: dict = Depends(get_current_user)):
    delete_session(session_id, user["local_uid"])


@app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
async def session_detail(session_id: str, user: dict = Depends(get_current_user)):
    try:
        detail = get_session_detail(session_id, user["local_uid"])
    except KeyError as e:
        raise HTTPException(404, str(e))
    return SessionDetailResponse(**detail)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sanitize(updates: dict) -> dict:
    """Strip large blobs from debug payloads to keep SSE events readable."""
    skip = {"docs", "style_context", "facts_context", "draft_content"}
    return {k: v for k, v in updates.items() if k not in skip}
