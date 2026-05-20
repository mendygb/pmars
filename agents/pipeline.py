import asyncio
import os
import sys
import json
import time
import argparse
import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pinecone import Pinecone
from pymongo import MongoClient
from tavily import TavilyClient
from langgraph.graph import StateGraph, END

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Add project root to path so `agents.*` imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import PostState
from agents.nodes.director import make_director_node
from agents.nodes.research import make_research_node
from agents.nodes.copywriter import make_copywriter_node
from agents.nodes.critic import make_critic_node

INDEX_NAME = "pmars-social-posts"
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "social_media_db"
COLLECTION_NAME = "posts"


def build_graph(async_client: AsyncOpenAI, sync_client: OpenAI, tavily_client, index, posts_col, chunks_by_pid, cleaned_posts, maps_api_key="", debug=False):
    director = make_director_node(async_client, debug=debug)
    research = make_research_node(async_client, sync_client, tavily_client, index, posts_col, chunks_by_pid, cleaned_posts, maps_api_key=maps_api_key, debug=debug)
    copywriter = make_copywriter_node(async_client, debug=debug)
    critic = make_critic_node(async_client, debug=debug)

    graph = StateGraph(PostState)
    graph.add_node("director", director)
    graph.add_node("research", research)
    graph.add_node("copywriter", copywriter)
    graph.add_node("critic", critic)

    graph.set_entry_point("director")

    # Director decides the first agent to invoke; downstream flow is fixed
    graph.add_conditional_edges(
        "director",
        lambda state: state["next_node"],
        {
            "research": "research",
            "copywriter": "copywriter",
            "critic": "critic",
            "ask_user": END,  # Director needs more info — pauses and shows clarification question
        },
    )

    graph.add_edge("research", "copywriter")
    graph.add_edge("copywriter", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


def save_transcript(turns: list, outputs_dir: str):
    os.makedirs(outputs_dir, exist_ok=True)
    filename = datetime.datetime.now().strftime("agent_%Y%m%d_%H%M%S.txt")
    path = os.path.join(outputs_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        for turn in turns:
            f.write(f"You: {turn['user_input']}\n\n")
            if turn.get("clarification_question"):
                f.write(f"Director: {turn['clarification_question']}\n\n")
            else:
                f.write(f"Style: {turn.get('style', '')}\n\n")
                f.write(f"Post:\n{turn.get('final_post', '')}\n\n")
            f.write("=" * 53 + "\n\n")
    print(f"\nTranscript saved to {path}")


async def _cli_loop(compiled, state, args, outputs_dir):
    transcript = []

    print("\n✨ Turn your experiences into posts. What's your story?")
    print("Type 'quit' or 'exit' to end.\n")

    loop = asyncio.get_running_loop()
    while True:
        user_input = (await loop.run_in_executor(None, input, "You: ")).strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        state["user_input"] = user_input
        t_start = time.time()
        state = await compiled.ainvoke(state)
        total_ms = int((time.time() - t_start) * 1000)

        turn_record = {"user_input": user_input, "style": state.get("style", "")}

        if args.debug:
            print(f"── Total ─────────────────────────────────────────")
            print(f"  {total_ms:>6} ms")
            print(f"─────────────────────────────────────────────────\n")

        if state.get("needs_clarification"):
            print(f"\nDirector: {state['clarification_question']}\n")
            turn_record["clarification_question"] = state["clarification_question"]
            # Store both the user's original message and the director's question in history
            # so the next turn has full context to continue naturally
            state["history"].append({"role": "user", "content": user_input})
            state["history"].append({"role": "assistant", "content": state["clarification_question"]})
        else:
            print(f"\n{state['final_post']}\n")
            turn_record["final_post"] = state["final_post"]
            state["history"].append({"role": "user", "content": user_input})
            state["history"].append({"role": "assistant", "content": state["final_post"]})

        transcript.append(turn_record)

    if transcript:
        save_transcript(transcript, os.path.join(os.path.dirname(__file__), "outputs"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Print RAG retrieval metrics after each research step")
    args = parser.parse_args()

    async_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sync_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    posts_col = mongo_client[DB_NAME][COLLECTION_NAME]

    rag_dir = os.path.join(os.path.dirname(__file__), "..", "rag")

    with open(os.path.join(rag_dir, "cleaned_chunks.json"), encoding="utf-8") as f:
        all_chunks = json.load(f)
    chunks_by_pid = {}
    for c in all_chunks:
        chunks_by_pid.setdefault(c["pid"], []).append(c)

    with open(os.path.join(rag_dir, "cleaned_posts.json"), encoding="utf-8") as f:
        cleaned_posts = {p["pid"]: p for p in json.load(f)}

    compiled = build_graph(async_client, sync_client, tavily_client, index, posts_col, chunks_by_pid, cleaned_posts, maps_api_key=maps_api_key, debug=args.debug)

    state: PostState = {
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

    asyncio.run(_cli_loop(compiled, state, args, os.path.join(os.path.dirname(__file__), "outputs")))


if __name__ == "__main__":
    main()
