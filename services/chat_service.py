import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from agents.state import PostState
from db.redis_store import check_cancel, clear_cancel, get_session, save_session
from schemas.chat import ChatRequest

_POSTSTATE_KEYS = set(PostState.__annotations__.keys())


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sanitize(updates: dict) -> dict:
    skip = {"docs", "style_context", "facts_context", "draft_content"}
    return {k: v for k, v in updates.items() if k not in skip}


async def run_chat_stream(
    req: ChatRequest,
    user: dict,
    graph,
    injection_classifier,
) -> StreamingResponse:
    if injection_classifier is not None:
        text_to_check = req.display_input or req.user_input
        result = await asyncio.to_thread(injection_classifier, text_to_check)
        if result[0]["label"] == "LABEL_1" and result[0]["score"] > 0.95:
            async def blocked_stream():
                yield _sse({"type": "error", "payload": {"message": "⚠️ We weren't able to process your message. Please try rephrasing it."}})
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
                "director":   "💭 On it..."               if is_refinement else "💭 Understanding your vibe...",
                "research":   "🔍 Digging deeper..."      if is_refinement else "🔍 Finding inspiration...",
                "copywriter": "✍️ Reworking your post..." if is_refinement else "✍️ Writing your post...",
                "critic":     "✨ Polishing it up..."     if is_refinement else "✨ Adding the finishing touches...",
            }

            node_start_times: dict[str, float] = {}
            node_timings: dict[str, float] = {}
            pipeline_start = time.monotonic()
            status_sent: set[str] = set()
            post_sent = False
            clarification_sent = False

            await clear_cancel(req.session_id)

            pipeline_input = {k: state[k] for k in _POSTSTATE_KEYS if k in state}
            cancelled = False
            async for ev in graph.astream_events(pipeline_input, version="v2"):
                if await check_cancel(req.session_id):
                    await clear_cancel(req.session_id)
                    cancelled = True
                    break

                kind = ev["event"]
                name = ev.get("name", "")
                lg_node = ev.get("metadata", {}).get("langgraph_node", "")
                node_key = lg_node if lg_node in NODE_STATUS else (name if name in NODE_STATUS else "")

                if kind == "on_chain_start" and lg_node:
                    node_start_times[lg_node] = time.monotonic()
                    if node_key and node_key not in status_sent:
                        status_sent.add(node_key)
                        yield _sse({"type": "status", "message": NODE_STATUS[node_key]})

                elif kind == "on_chat_model_stream" and lg_node == "critic":
                    token = ev["data"]["chunk"].content
                    if token:
                        yield _sse({"type": "token", "content": token})

                elif kind == "on_chat_model_end" and lg_node == "critic" and not post_sent:
                    post_sent = True
                    yield _sse({"type": "message_end"})

                elif kind == "on_chain_end" and lg_node:
                    output = ev["data"].get("output", {})
                    if not isinstance(output, dict):
                        continue

                    if lg_node in node_start_times:
                        node_timings[lg_node] = round((time.monotonic() - node_start_times[lg_node]) * 1000)

                    state.update(output)

                    if req.debug:
                        yield _sse({"type": "debug", "node": lg_node, "payload": _sanitize(output)})

                    if output.get("final_post") and not post_sent:
                        post_sent = True
                        is_error = not state.get("safety_passed", True)
                        yield _sse({"type": "message", "content": output["final_post"], "is_error": is_error})

                    if output.get("needs_clarification") and not clarification_sent:
                        clarification_sent = True
                        yield _sse({"type": "clarification", "payload": {"question": output["clarification_question"]}})

            if cancelled:
                yield _sse({"type": "cancelled"})
                return

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
