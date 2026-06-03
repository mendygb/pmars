from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from core.config import settings
from schemas.chat import GenerateTitleRequest


async def generate_title(req: GenerateTitleRequest) -> dict:
    if not req.content and not req.title:
        raise HTTPException(400, "Provide content or title")
    if req.content and req.title:
        prompt = f"The current title is: \"{req.title}\". Write a fresh alternative title (3-6 words) for this post — do not reuse the same words. Output only the title, no quotes or punctuation:\n\n{req.content[:600]}"
    elif req.content:
        prompt = f"Write a short, catchy title (3-6 words) for this social media post. Output only the title, no quotes or punctuation:\n\n{req.content[:600]}"
    else:
        prompt = f"Rephrase this title to be more engaging. Keep it short (3-6 words). Output only the title, no quotes or punctuation:\n\n{req.title}"

    llm = ChatOpenAI(
        model=settings.title_model,
        max_tokens=20,
        temperature=1.0,
        api_key=settings.openai_api_key,
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"title": response.content.strip().strip('"\'')}
