from fastapi import APIRouter, Depends

from core.security import get_current_user
from schemas.chat import GenerateTitleRequest
from services.title_service import generate_title

router = APIRouter()


@router.post("/api/generate-title")
async def generate_title_endpoint(req: GenerateTitleRequest, user: dict = Depends(get_current_user)):
    return await generate_title(req)
