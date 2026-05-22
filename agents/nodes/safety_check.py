import asyncio
import logging
import time
from agents.state import PostState

logger = logging.getLogger(__name__)

SAFE_LABEL = "OK"
FRIENDLY_ERROR = (
    "I wasn't able to generate a post for this content. "
    "Please try rephrasing your experience."
)


def make_safety_check_node(classifier, debug=False):
    async def safety_check_node(state: PostState) -> dict:
        draft = state.get("draft_content", "")

        loop = asyncio.get_running_loop()
        t0 = time.time()
        result = await loop.run_in_executor(None, classifier, draft)
        check_ms = int((time.time() - t0) * 1000)

        # transformers pipeline returns a dict or list[dict] depending on version
        top = result[0] if isinstance(result, list) else result
        safe = top["label"] == SAFE_LABEL

        if debug:
            logger.debug(
                f"\n── Safety Check ─────────────────────────────────\n"
                f"  label:    {top['label']}  score: {top['score']:.3f}\n"
                f"  passed:   {safe}\n"
                f"  time:     {check_ms:>6} ms\n"
                "─────────────────────────────────────────────────"
            )

        if safe:
            return {"safety_passed": True}
        return {
            "safety_passed": False,
            "final_post": FRIENDLY_ERROR,
        }

    return safety_check_node
