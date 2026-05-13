import os
import importlib.util

from agents.state import PostState

# Import retrieve() and build_context() from 03_rag_query.py.
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


def make_research_node(client, index, posts_col, chunks_by_pid, cleaned_posts, debug=False):
    def research_node(state: PostState) -> dict:
        print("🔍 Finding inspiration...")

        # Build retrieval query from current input + history context for richer semantic match
        query_parts = [state["user_input"]]
        for turn in state.get("history", [])[-4:]:  # last 2 turns
            if turn["role"] == "user":
                query_parts.append(turn["content"])
        query = " ".join(query_parts)

        final_docs, metrics = retrieve(
            index, client, posts_col, chunks_by_pid, cleaned_posts, query
        )

        if debug:
            try:
                # format_metrics expects intent + LLM fields from the old pipeline — inject stubs
                metrics["intent"] = state.get("next_node", "research")
                metrics["intent_confidence"] = 1.0
                metrics["intent_method"] = "director"
                metrics["intent_ms"] = 0
                metrics["llm_first_token_ms"] = 0
                metrics["llm_total_ms"] = 0
                metrics["total_ms"] = sum(
                    metrics.get(k, 0)
                    for k in ("embed_ms", "vector_search_ms", "tag_search_ms", "rrf_ms")
                )
                print("\n" + format_metrics(metrics) + "\n")
            except Exception as e:
                print(f"\n[debug] RAG metrics unavailable: {e}")
                print(f"[debug] raw metrics: {metrics}\n")

        return {
            "location_info": {
                "docs": final_docs,
                "context": build_context(final_docs),
                "metrics": metrics,
            }
        }

    return research_node
