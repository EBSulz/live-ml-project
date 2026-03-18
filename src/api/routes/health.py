"""Health and readiness endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_model
from src.api.schemas import HealthResponse
from src.models.online_model import ChampionModel

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    model: ChampionModel = Depends(get_model),
) -> HealthResponse:
    uptime = time.time() - request.app.state.start_time
    return HealthResponse(
        status="healthy",
        model_loaded=model.model is not None,
        model_version=model.run_id,
        uptime_seconds=round(uptime, 2),
    )


@router.get("/ready")
async def readiness(request: Request) -> dict:
    """Returns 200 only when the model is loaded and the app is ready."""
    model: ChampionModel = request.app.state.model
    if model is None or model.model is None:
        return {"ready": False}
    return {"ready": True}
