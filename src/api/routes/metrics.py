"""Model performance metrics endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_evaluator
from src.api.schemas import MetricsResponse
from src.models.evaluation import LiveEvaluator

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
async def model_metrics(
    evaluator: LiveEvaluator = Depends(get_evaluator),
) -> MetricsResponse:
    m = evaluator.metrics
    return MetricsResponse(
        accuracy=m.get("accuracy", 0.0),
        weighted_f1=m.get("weighted_f1", 0.0),
        macro_f1=m.get("macro_f1", 0.0),
        cohen_kappa=m.get("cohen_kappa", 0.0),
        predictions_count=evaluator.step,
    )
