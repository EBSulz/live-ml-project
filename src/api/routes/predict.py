"""Prediction endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from src.api.dependencies import (
    get_data_source,
    get_evaluator,
    get_feature_engine,
    get_model,
)
from src.api.schemas import PredictionRequest, PredictionResponse
from src.data_ingestion.base import DataSource
from src.features.engineering import FeatureEngine
from src.models.evaluation import LiveEvaluator
from src.models.online_model import ChampionModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    req: PredictionRequest,
    model: ChampionModel = Depends(get_model),
    source: DataSource = Depends(get_data_source),
    evaluator: LiveEvaluator = Depends(get_evaluator),
    feature_engine: FeatureEngine = Depends(get_feature_engine),
) -> PredictionResponse:
    """Fetch latest market data, compute features, and return a signal."""
    latest = await source.fetch_latest(req.symbol, req.interval)
    features = feature_engine.update(latest)

    signal = model.predict_one(features)
    proba = model.predict_proba_one(features)

    return PredictionResponse(
        symbol=req.symbol,
        signal=signal,
        confidence=proba,
        timestamp=datetime.now(timezone.utc),
        model_version=model.run_id,
    )
