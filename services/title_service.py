from fastapi import HTTPException
from openai import AsyncOpenAI

from core.config import settings
from schemas.chat import GenerateTitleRequest


async def generate_title(req: GenerateTitleRequest) -> dict:
    if not req.content and not req.title:
        raise HTTPException(400, "Provide content or title")
    if req.content:
        prompt = f"Write a short, catchy title (3-6 words) for this social media post. Output only the title, no quotes or punctuation:\n\n{req.content[:600]}"
    else:
        prompt = f"Rephrase this title to be more engaging. Keep it short (3-6 words). Output only the title, no quotes or punctuation:\n\n{req.title}"
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0.7,
    )
    return {"title": response.choices[0].message.content.strip().strip('"\'')}
