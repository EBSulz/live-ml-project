"""Pydantic v2 request/response models for the prediction API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    symbol: str = Field(
        ...,
        pattern=r"^[A-Z]{2,10}(/[A-Z]{2,10})?$",
        examples=["BTC/USD", "AAPL"],
    )
    interval: str = Field(
        default="1h",
        pattern=r"^\d+[mhd]$",
        examples=["1h", "15min", "1d"],
    )


class PredictionResponse(BaseModel):
    symbol: str
    signal: Literal["buy", "hold", "sell"]
    confidence: dict[str, float]
    timestamp: datetime
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    accuracy: float
    weighted_f1: float
    macro_f1: float
    cohen_kappa: float
    predictions_count: int
