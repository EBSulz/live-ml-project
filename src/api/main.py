"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import health, metrics, predict
from src.config.settings import get_settings
from src.data_ingestion.factory import get_data_source
from src.features.engineering import FeatureEngine
from src.models.evaluation import LiveEvaluator
from src.models.online_model import ChampionModel
from src.observability.logging import RequestLoggingMiddleware, setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Market Predictor (%s)", settings.app_env)

    app.state.start_time = time.time()
    app.state.model = ChampionModel.from_mlflow(settings.mlflow_experiment_name)
    app.state.data_source = get_data_source(settings.data_source)
    app.state.evaluator = LiveEvaluator(
        log_interval=settings.mlflow_log_interval,
        experiment_name=settings.mlflow_live_experiment_name,
    )
    app.state.feature_engine = FeatureEngine()

    logger.info(
        "Model loaded: %s (run=%s)", app.state.model.model_name, app.state.model.run_id
    )
    yield

    app.state.evaluator.flush()
    logger.info("Shutting down Market Predictor")


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title="Market Predictor",
        description="Live financial market prediction with online learning",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(predict.router)
    app.include_router(health.router)
    app.include_router(metrics.router)
    return app


app = create_app()
