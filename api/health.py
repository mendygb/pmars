from fastapi import APIRouter

from schemas.common import HealthResponse

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")
