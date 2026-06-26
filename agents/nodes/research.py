import asyncio
import importlib.util
import json
import logging
import os
import time

from mcp import ClientSession, StdioServerParameters

logger = logging.getLogger(__name__)
from mcp.client.stdio import stdio_client
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import PostState
from core.config import settings

# Import retrieve(), build_context(), format_metrics() from 03_rag_query.py.
# The filename starts with a digit so standard import won't work — use importlib.
_rag_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../rag/03_rag_query.py")
)
_spec = importlib.util.spec_from_file_location("rag03", _rag_path)
_rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rag)

retrieve = _rag.retrieve
build_context = _rag.build_context
format_metrics = _rag.format_metrics

SYSTEM_PROMPT = """You are a research agent for a social media post writing pipeline. Your job is to gather relevant reference material for writing an engaging post about a place or experience.

# NOTE (project-specific): local corpus covers 24 Bay Area lifestyle posts only.
# In a production app with a larger corpus, retrieve_rag would cover much more ground.

Tool selection guidelines:
- User provides a URL → call fetch_url to get the full page content
- Specific named place (restaurant, cafe, park, attraction) → call get_place_details for accurate facts (rating, hours, address, reviews); also call retrieve_rag if Bay Area for style reference
- Bay Area place (vague, no specific name) → call retrieve_rag; optionally search_web for current details
- Non-Bay Area place → call search_web and get_place_details if a specific name is given
- Abstract request (poem, mood, no specific place) → call retrieve_rag for style reference only"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_rag",
            "description": "Search the local corpus of Bay Area lifestyle posts for relevant style and tone references. Best for Bay Area locations and experiences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query describing the place or experience"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information about a place. Use for non-Bay Area locations, or when current details like hours, prices, or recent events would improve the post.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the full content of a specific URL as markdown. Use when the user provides a URL, or when you need detailed content from a specific webpage (restaurant site, park page, event listing).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_place_details",
            "description": "Look up a specific named place on Google Maps to get accurate details: rating, address, opening hours, phone number, website, and recent reviews. Use when the user mentions a specific restaurant, cafe, park, or attraction by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "place_name": {
                        "type": "string",
                        "description": "The name of the place to look up, include city/area for accuracy (e.g. 'Tartine Bakery San Francisco')"
                    }
                },
                "required": ["place_name"]
            }
        }
    }
]


async def _get_place_details_async(place_name: str, api_key: str) -> str:
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-google-maps"],
        env={**os.environ, "GOOGLE_MAPS_API_KEY": api_key, "NPM_CONFIG_LOGLEVEL": "silent"},
    )
    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            # Step 1: search to get place_id
            search_result = await session.call_tool(
                "maps_search_places", arguments={"query": place_name}
            )
            search_text = search_result.content[0].text if search_result.content else ""
            if not search_text:
                return ""

            try:
                places = json.loads(search_text)
                place_id = places[0].get("place_id", "") if places else ""
            except (json.JSONDecodeError, IndexError, KeyError):
                return search_text

            if not place_id:
                return search_text

            # Step 2: fetch full details using place_id
            details_result = await session.call_tool(
                "maps_get_place_details", arguments={"place_id": place_id}
            )
            return details_result.content[0].text if details_result.content else search_text


async def _fetch_url_async(url: str) -> str:
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_server_fetch"],
        env={**os.environ, "NPM_CONFIG_LOGLEVEL": "silent"},
    )
    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            result = await session.call_tool("fetch", arguments={"url": url})
            return result.content[0].text if result.content else ""


def _format_place_details(raw: str, place_name: str) -> str:
    try:
        data = json.loads(raw)
        lines = [f"[Google Maps: {data.get('name', place_name)}]"]
        if data.get("formatted_address"):
            lines.append(f"Address: {data['formatted_address']}")
        if data.get("rating"):
            total = data.get("user_ratings_total", "")
            lines.append(f"Rating: {data['rating']}/5{f' ({total} reviews)' if total else ''}")
        hours = data.get("opening_hours", {})
        if hours.get("weekday_text"):
            lines.append("Hours:\n  " + "\n  ".join(hours["weekday_text"]))
        reviews = data.get("reviews", [])[:5]
        if reviews:
            lines.append("Recent reviews:")
            for r in reviews:
                lines.append(f"  ★{r.get('rating', '?')} — {r.get('text', '')[:200]}")
        return "\n".join(lines)
    except (json.JSONDecodeError, KeyError):
        return f"[Google Maps: {place_name}]\n{raw[:1000]}"


def _format_web_results(results: list) -> str:
    blocks = [f"[Web: {r['title']}]\n{r['content']}" for r in results if r.get("content")]
    if not blocks:
        return ""
    return "Reference material from the web:\n\n" + "\n\n".join(blocks)


# HARDCODED: in production, replace with a real async HTTP call to the internal MP service:
#   GET /internal/users/{media_id}/posts?sort=likes&limit=5
# The MP service returns the user's popular published posts, which are used as a
# personalised style reference — closer to this user's voice than the community RAG corpus.
_PERSONAL_POSTS_STUB = [
    {
        "content": (
            "Lazy Sunday afternoon at Sightglass Coffee ☕\n"
            "Sunlight filtering through the industrial windows onto rough wooden tables.\n"
            "Got a pour over — floral on the nose, citrus finish.\n"
            "Just sat there, zoned out, nothing to do and no need to do anything.\n"
            "#SanFrancisco #CoffeeTime #SightglassCoffee #SlowLiving #BayAreaDaily"
        ),
        "likes": 312,
    },
    {
        "content": (
            "First time at Tartine Bakery — waited 40 minutes in line.\n"
            "Worth it? The country bread was fresh out of the oven, crust shatteringly crisp, inside chewy and moist.\n"
            "Smeared with half-melted butter. Stupidly simple, stupidly good.\n"
            "Come earlier next time — apparently the line starts at 5.\n"
            "#TartineBakery #SanFrancisco #MissionDistrict #BreadLovers #BayAreaEats"
        ),
        "likes": 487,
    },
    {
        "content": (
            "Lands End Trail at dusk.\n"
            "Fog rolling in from the Golden Gate, swallowing the towers one by one.\n"
            "Gravel underfoot, cliff to the side, wind wrecking my hair.\n"
            "Moments like this remind me why living in the Bay is worth it.\n"
            "#LandsEnd #SanFrancisco #HikingBay #Sunset #BayAreaOutdoors"
        ),
        "likes": 623,
    },
]


async def _get_personal_style_async(media_id: str) -> str:
    """Fetch user's popular posts as personalised style reference.

    HARDCODED stub — production version calls:
      GET /internal/users/{media_id}/posts?sort=likes&limit=5
    and formats the response the same way.
    """
    if not media_id:
        return ""
    posts = _PERSONAL_POSTS_STUB  # swap for real MP service response in production
    blocks = [f"[Personal post — {p['likes']} likes]\n{p['content']}" for p in posts]
    # Cap at 700 chars so community RAG always gets at least ~800 chars of the 1500 limit
    return ("User's own popular posts (personalised style reference):\n\n" + "\n\n".join(blocks))[:700]


def make_research_node(sync_client: OpenAI, tavily_client, index, posts_col, chunks_by_pid, cleaned_posts, maps_api_key="", debug=False):
    async def research_node(state: PostState) -> dict:
        logger.info("🔍 Finding inspiration...")

        try:
            return await _research_node_impl(state)
        except Exception as e:
            logger.warning(f"Research failed, continuing with empty context: {e}")
            return {"location_info": {}}

    async def _research_node_impl(state: PostState) -> dict:
        # Build retrieval query from current input + recent history for richer semantic match
        query_parts = [state["user_input"]]
        for turn in state.get("history", [])[-4:]:  # last 2 turns
            if turn["role"] == "user":
                query_parts.append(turn["content"])
        query = " ".join(query_parts)

        llm = ChatOpenAI(
            model=settings.research_model,
            temperature=0,
            api_key=settings.openai_api_key,
        ).bind_tools(TOOLS, tool_choice="required")  # must call at least one tool

        # Step 1: LLM tool selection + personal style fetch run concurrently
        t0 = time.time()
        response, personal_style = await asyncio.gather(
            llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Gather research for writing a post about: {query}"),
            ]),
            _get_personal_style_async(state.get("media_id", "")),
        )
        tool_selection_ms = int((time.time() - t0) * 1000)

        tool_calls = response.tool_calls or []

        # Step 2: Execute all tool calls concurrently — each is independent
        async def _execute_tool_call_async(tool_call):
            fn_name = tool_call["name"]
            args = tool_call["args"]
            tool_query = args.get("query", query)
            result = {
                "fn_name": fn_name,
                "context_part": None,
                "docs": [],
                "metrics": {},
                "web_search_used": False,
                "debug_lines": [],
                "elapsed_ms": 0,
            }
            t_tool = time.time()
            loop = asyncio.get_running_loop()

            if fn_name == "retrieve_rag":
                result["debug_lines"].append(f"[debug] → retrieve_rag(query='{tool_query}')")
                # retrieve() is sync (uses sync Pinecone + pymongo) — run in thread pool
                final_docs, metrics = await loop.run_in_executor(
                    None,
                    lambda: retrieve(index, sync_client, posts_col, chunks_by_pid, cleaned_posts, tool_query),
                )
                result["docs"] = final_docs
                result["metrics"] = metrics
                if final_docs:
                    result["context_part"] = build_context(final_docs)

            elif fn_name == "fetch_url":
                url = args.get("url", "")
                result["debug_lines"].append(f"[debug] → fetch_url(url='{url}')")
                try:
                    page_content = await _fetch_url_async(url)
                    if page_content:
                        result["context_part"] = f"[Fetched page: {url}]\n{page_content[:3000]}"
                    result["debug_lines"].append(f"[debug] fetched {len(page_content)} chars from {url}")
                except Exception as e:
                    result["debug_lines"].append(f"[debug] fetch_url failed: {e}")

            elif fn_name == "get_place_details":
                place_name = args.get("place_name", query)
                result["debug_lines"].append(f"[debug] → get_place_details(place='{place_name}')")
                try:
                    raw = await _get_place_details_async(place_name, maps_api_key)
                    if raw:
                        formatted = _format_place_details(raw, place_name)
                        result["context_part"] = formatted
                        chars = len(formatted)
                        words = len(formatted.split())
                        tokens_est = chars // 4
                        result["debug_lines"].append(
                            f"[debug] got place details for '{place_name}' — "
                            f"{chars} chars, {words} words, ~{tokens_est} tokens"
                        )
                    else:
                        result["debug_lines"].append(f"[debug] get_place_details returned empty")
                except Exception as e:
                    result["debug_lines"].append(f"[debug] get_place_details failed: {e}")

            elif fn_name == "search_web":
                result["web_search_used"] = True
                result["debug_lines"].append(f"[debug] → search_web(query='{tool_query}')")
                # tavily is sync — run in thread pool
                # NOTE (project-specific): Tavily used as web fallback for out-of-corpus content.
                # In a production app with a large corpus this would rarely be needed.
                search_results = await loop.run_in_executor(
                    None,
                    lambda: tavily_client.search(tool_query, max_results=3),
                )
                web_context = _format_web_results(search_results.get("results", []))
                if web_context:
                    result["context_part"] = web_context
                result["debug_lines"].append(f"[debug] web search returned {len(search_results.get('results', []))} results")

            result["elapsed_ms"] = int((time.time() - t_tool) * 1000)
            return result

        t_parallel = time.time()
        tool_results = await asyncio.gather(*[_execute_tool_call_async(tc) for tc in tool_calls])
        parallel_wall_ms = int((time.time() - t_parallel) * 1000)

        # Step 3: Merge results — keep RAG style reference separate from factual context
        combined_docs = []
        style_parts = []   # from retrieve_rag — truncated before sending to copywriter
        facts_parts = []   # from get_place_details, search_web, fetch_url — sent in full
        rag_metrics = {}
        web_search_used = False

        # Personal posts take priority — prepend so Copywriter sees them first
        if personal_style:
            style_parts.append(personal_style)

        for r in tool_results:
            if debug:
                for line in r["debug_lines"]:
                    logger.debug(line)
            combined_docs.extend(r["docs"])
            if r["context_part"]:
                if r["fn_name"] == "retrieve_rag":
                    style_parts.append(r["context_part"])
                else:
                    facts_parts.append(r["context_part"])
            if r["metrics"]:
                rag_metrics = r["metrics"]
                if debug:
                    try:
                        rag_metrics["intent"] = "retrieve_rag"
                        rag_metrics["intent_confidence"] = 1.0
                        rag_metrics["intent_method"] = "tool_call"
                        rag_metrics["intent_ms"] = 0
                        rag_metrics["llm_first_token_ms"] = 0
                        rag_metrics["llm_total_ms"] = 0
                        rag_metrics["total_ms"] = sum(
                            rag_metrics.get(k, 0)
                            for k in ("embed_ms", "vector_search_ms", "tag_search_ms", "rrf_ms")
                        )
                        logger.debug("\n" + format_metrics(rag_metrics))
                    except Exception as e:
                        logger.debug(f"RAG metrics unavailable: {e}")
                        logger.debug(f"raw metrics: {rag_metrics}")
            web_search_used = web_search_used or r["web_search_used"]

        if debug:
            tool_timing = "\n".join(f"  {r['fn_name']:<20} {r['elapsed_ms']:>6} ms" for r in tool_results)
            wall = f"\n  wall-clock:      {parallel_wall_ms:>6} ms  (parallel)" if len(tool_results) > 1 else ""
            logger.debug(
                f"\n── Research timing ──────────────────────────────\n"
                f"  tool selection:  {tool_selection_ms:>6} ms\n"
                f"{tool_timing}{wall}\n"
                "────────────────────────────────────────────────"
            )

        return {
            "location_info": {
                "docs": combined_docs,
                "style_context": "\n\n".join(style_parts),
                "facts_context": "\n\n".join(facts_parts),
                "metrics": rag_metrics,
                "web_search_used": web_search_used,
                "personal_style_chars": len(personal_style),
            }
        }

    return research_node
