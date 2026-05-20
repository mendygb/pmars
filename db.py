import json
import sqlite3
import uuid
from datetime import datetime

DB_PATH = "sessions.db"

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
}


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            local_uid  TEXT NOT NULL,
            state_json TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def create_session(local_uid: str) -> str:
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions (session_id, local_uid, state_json) VALUES (?, ?, ?)",
        (session_id, local_uid, json.dumps(FRESH_STATE)),
    )
    conn.commit()
    conn.close()
    return session_id


def get_session(session_id: str, local_uid: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT state_json FROM sessions WHERE session_id = ? AND local_uid = ?",
        (session_id, local_uid),
    ).fetchone()
    conn.close()
    if not row:
        raise KeyError(f"Session {session_id!r} not found for user {local_uid!r}")
    return json.loads(row[0])


def save_session(session_id: str, local_uid: str, state: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO sessions (session_id, local_uid, state_json, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
               state_json = excluded.state_json,
               updated_at = excluded.updated_at""",
        (session_id, local_uid, json.dumps(state), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def list_sessions(local_uid: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT session_id, state_json, updated_at FROM sessions WHERE local_uid = ? ORDER BY updated_at DESC",
        (local_uid,),
    ).fetchall()
    conn.close()
    result = []
    for session_id, state_json, updated_at in rows:
        state = json.loads(state_json)
        history = state.get("history", [])
        user_turns = [t for t in history if t["role"] == "user"]
        if not user_turns:
            continue  # skip sessions where no messages were sent
        result.append({
            "session_id": session_id,
            "updated_at": updated_at,
            "turn_count": len(user_turns),
            "style": state.get("style", ""),
            "preview": user_turns[0]["content"][:80],
        })
    return result


def delete_session(session_id: str, local_uid: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "DELETE FROM sessions WHERE session_id = ? AND local_uid = ?",
        (session_id, local_uid),
    )
    conn.commit()
    conn.close()


def get_session_detail(session_id: str, local_uid: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT state_json, updated_at FROM sessions WHERE session_id = ? AND local_uid = ?",
        (session_id, local_uid),
    ).fetchone()
    conn.close()
    if not row:
        raise KeyError(f"Session {session_id!r} not found for user {local_uid!r}")
    state = json.loads(row[0])
    loc = state.get("location_info", {})
    return {
        "session_id": session_id,
        "updated_at": row[1],
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
