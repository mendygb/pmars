from pymongo import MongoClient, ASCENDING, DESCENDING

_client: MongoClient = None
_completed_col = None
_posts_col = None
_audit_col = None


def init_mongo(mongo_uri: str = "mongodb://localhost:27017/") -> None:
    global _client, _completed_col, _posts_col, _audit_col
    _client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
    db = _client["social_media_db"]
    _completed_col = db["completed_sessions"]
    _posts_col = db["posts"]
    _audit_col = db["audit_logs"]
    _audit_col.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
    _audit_col.create_index([("timestamp", DESCENDING)])


def close_mongo() -> None:
    if _client:
        _client.close()


def is_available() -> bool:
    return _completed_col is not None


def get_posts_col():
    return _posts_col


def list_completed(local_uid: str) -> list[dict]:
    if _completed_col is None:
        return []
    result = []
    for doc in _completed_col.find({"local_uid": local_uid}, sort=[("completed_at", -1)]):
        history = doc.get("history", [])
        user_turns = [t for t in history if t["role"] == "user"]
        if not user_turns:
            continue
        result.append({
            "session_id": doc["session_id"],
            "updated_at": doc.get("completed_at", ""),
            "turn_count": len(user_turns),
            "style": doc.get("style", ""),
            "preview": user_turns[0]["content"][:80],
            "completed": True,
        })
    return result


def find_completed(session_id: str, local_uid: str) -> dict | None:
    if _completed_col is None:
        return None
    return _completed_col.find_one({"session_id": session_id, "local_uid": local_uid})


def delete_completed(session_id: str, local_uid: str) -> int:
    if _completed_col is None:
        return 0
    return _completed_col.delete_one({"session_id": session_id, "local_uid": local_uid}).deleted_count


def insert_completed(doc: dict) -> None:
    if _completed_col is None:
        raise RuntimeError("MongoDB not initialized")
    _completed_col.insert_one(doc)


def insert_audit_log(doc: dict) -> None:
    if _audit_col is None:
        return
    _audit_col.insert_one(doc)
