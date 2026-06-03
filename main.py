import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone
from tavily import TavilyClient
from transformers import pipeline as hf_pipeline

from agents.pipeline import build_graph
from api.chat import router as chat_router
from api.health import router as health_router
from api.sessions import router as sessions_router
from api.title import router as title_router
from core.config import settings
from core.logging import setup_agent_logger
from core.security import init_firebase, load_user_map
from db import mongo_store
from db.redis_store import init_db

setup_agent_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_firebase()
    load_user_map()
    init_db(settings.redis_url)
    mongo_store.init_mongo()

    sync_client = OpenAI(api_key=settings.openai_api_key)
    tavily_client = TavilyClient(api_key=settings.tavily_api_key)

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index("pmars-social-posts")

    rag_dir = os.path.join(os.path.dirname(__file__), "rag")
    with open(os.path.join(rag_dir, "cleaned_chunks.json")) as f:
        all_chunks = json.load(f)
    chunks_by_pid: dict = {}
    for c in all_chunks:
        chunks_by_pid.setdefault(c["pid"], []).append(c)
    with open(os.path.join(rag_dir, "cleaned_posts.json")) as f:
        cleaned_posts = {p["pid"]: p for p in json.load(f)}

    safety_classifier = hf_pipeline("text-classification", model="KoalaAI/Text-Moderation")

    app.state.graph = build_graph(
        sync_client, tavily_client,
        index, mongo_store.get_posts_col(),
        chunks_by_pid, cleaned_posts,
        maps_api_key=settings.google_maps_api_key,
        safety_classifier=safety_classifier,
    )

    yield

    mongo_store.close_mongo()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(sessions_router)
app.include_router(title_router)
