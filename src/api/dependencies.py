"""FastAPI dependency injection for shared resources."""

from __future__ import annotations

from fastapi import Request

from src.data_ingestion.base import DataSource
from src.features.engineering import FeatureEngine
from src.models.evaluation import LiveEvaluator
from src.models.online_model import ChampionModel


def get_model(request: Request) -> ChampionModel:
    return request.app.state.model


def get_data_source(request: Request) -> DataSource:
    return request.app.state.data_source


def get_evaluator(request: Request) -> LiveEvaluator:
    return request.app.state.evaluator


def get_feature_engine(request: Request) -> FeatureEngine:
    return request.app.state.feature_engine
