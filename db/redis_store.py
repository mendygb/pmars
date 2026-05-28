import json
import uuid
from datetime import datetime, timezone

import redis.asyncio as aioredis

_redis: aioredis.Redis = None

SESSION_TTL = 60 * 60 * 24 * 30  # 30 days
CANCEL_TTL = 300                  # 5 minutes

FRESH_STATE = {
    "user_input": "",
    "style": "",
    "location_info": {},
    "draft_content": "",
    "final_post": "",
    "suggestions": [],
    "history": [],
    "next_node": "",
    "needs_clarification": False,
    "clarification_question": "",
    "safety_passed": True,
    "media_id": "",
    "user_profile_injected": False,
}


def init_db(redis_url: str = "redis://localhost:6379"):
    global _redis
    _redis = aioredis.from_url(redis_url, decode_responses=True)


# ── Session CRUD ──────────────────────────────────────────────────────────────

async def create_session(local_uid: str, draft_content: str = "") -> str:
    session_id = str(uuid.uuid4())
    fresh = {**FRESH_STATE, "media_id": local_uid, "draft_content": draft_content}
    now = datetime.now(timezone.utc).isoformat()
    pipe = _redis.pipeline()
    pipe.hset(f"session:{session_id}", mapping={
        "local_uid": local_uid,
        "state_json": json.dumps(fresh),
        "updated_at": now,
    })
    pipe.expire(f"session:{session_id}", SESSION_TTL)
    pipe.zadd(f"sessions:uid:{local_uid}", {session_id: _unix_now()})
    await pipe.execute()
    return session_id


async def get_session(session_id: str, local_uid: str) -> dict:
    data = await _redis.hgetall(f"session:{session_id}")
    if not data or data.get("local_uid") != local_uid:
        raise KeyError(f"Session {session_id!r} not found for user {local_uid!r}")
    return json.loads(data["state_json"])


async def save_session(session_id: str, local_uid: str, state: dict):
    now = datetime.now(timezone.utc).isoformat()
    pipe = _redis.pipeline()
    pipe.hset(f"session:{session_id}", mapping={
        "local_uid": local_uid,
        "state_json": json.dumps(state),
        "updated_at": now,
    })
    pipe.expire(f"session:{session_id}", SESSION_TTL)
    pipe.zadd(f"sessions:uid:{local_uid}", {session_id: _unix_now()})
    await pipe.execute()


async def list_sessions(local_uid: str) -> list[dict]:
    session_ids = await _redis.zrevrange(f"sessions:uid:{local_uid}", 0, -1)
    if not session_ids:
        return []

    pipe = _redis.pipeline()
    for sid in session_ids:
        pipe.hgetall(f"session:{sid}")
    all_data = await pipe.execute()

    result = []
    stale_ids = []
    for sid, data in zip(session_ids, all_data):
        if not data:
            stale_ids.append(sid)
            continue
        state = json.loads(data["state_json"])
        history = state.get("history", [])
        user_turns = [t for t in history if t["role"] == "user"]
        if not user_turns:
            continue
        result.append({
            "session_id": sid,
            "updated_at": data.get("updated_at", ""),
            "turn_count": len(user_turns),
            "style": state.get("style", ""),
            "preview": user_turns[0]["content"][:80],
        })

    if stale_ids:
        await _redis.zrem(f"sessions:uid:{local_uid}", *stale_ids)

    return result


async def delete_session(session_id: str, local_uid: str):
    data = await _redis.hgetall(f"session:{session_id}")
    if not data or data.get("local_uid") != local_uid:
        raise KeyError(f"Session {session_id!r} not found for user {local_uid!r}")
    pipe = _redis.pipeline()
    pipe.delete(f"session:{session_id}")
    pipe.zrem(f"sessions:uid:{local_uid}", session_id)
    await pipe.execute()


async def get_session_detail(session_id: str, local_uid: str) -> dict:
    data = await _redis.hgetall(f"session:{session_id}")
    if not data or data.get("local_uid") != local_uid:
        raise KeyError(f"Session {session_id!r} not found for user {local_uid!r}")
    state = json.loads(data["state_json"])
    loc = state.get("location_info", {})
    return {
        "session_id": session_id,
        "updated_at": data.get("updated_at", ""),
        "history": state.get("history", []),
        "debug": {
            "style": state.get("style", ""),
            "next_node": state.get("next_node", ""),
            "needs_clarification": state.get("needs_clarification", False),
            "draft_content": state.get("draft_content", ""),
            "final_post": state.get("final_post", ""),
            "web_search_used": loc.get("web_search_used", False),
            "rag_metrics": loc.get("metrics", {}),
            "timings": state.get("_timings", {}),
        },
    }


# ── Cancel flags ──────────────────────────────────────────────────────────────

async def set_cancel(session_id: str) -> None:
    await _redis.set(f"cancel:{session_id}", "1", ex=CANCEL_TTL)


async def check_cancel(session_id: str) -> bool:
    return bool(await _redis.exists(f"cancel:{session_id}"))


async def clear_cancel(session_id: str) -> None:
    await _redis.delete(f"cancel:{session_id}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _unix_now() -> float:
    return datetime.now(timezone.utc).timestamp()
