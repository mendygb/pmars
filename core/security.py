import json
import os

import firebase_admin
from fastapi import Header, HTTPException
from firebase_admin import auth, credentials

from core.config import settings

_USER_MAP: dict = {}


def init_firebase() -> None:
    if os.path.exists(settings.firebase_sa_path):
        cred = credentials.Certificate(settings.firebase_sa_path)
        firebase_admin.initialize_app(cred)
    else:
        print(f"WARNING: Firebase service account not found at {settings.firebase_sa_path!r}. Auth will be disabled.")


def load_user_map(path: str = "users.json") -> None:
    global _USER_MAP
    if os.path.exists(path):
        with open(path) as f:
            _USER_MAP = json.load(f)
    else:
        print("WARNING: users.json not found. All authenticated requests will be rejected.")


async def get_current_user(authorization: str = Header(...)) -> dict:
    if not firebase_admin._apps:
        raise HTTPException(503, "Firebase not initialized — add firebase-service-account.json")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        decoded = auth.verify_id_token(token)
    except Exception as e:
        raise HTTPException(401, f"Invalid token: {e}")
    user = _USER_MAP.get(decoded["uid"])
    if not user:
        raise HTTPException(403, "Firebase UID not registered in users.json")
    return user
