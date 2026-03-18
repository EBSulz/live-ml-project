"""Shared test fixtures."""

from __future__ import annotations

import os
from typing import AsyncGenerator

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")
os.environ.setdefault("DATA_SOURCE", "twelve_data")
os.environ.setdefault("TWELVE_DATA_API_KEY", "test_key")
os.environ.setdefault("ALPHA_VANTAGE_API_KEY", "test_key")


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """A small OHLCV DataFrame for testing."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
            "open": [100 + i * 0.1 for i in range(100)],
            "high": [101 + i * 0.1 for i in range(100)],
            "low": [99 + i * 0.1 for i in range(100)],
            "close": [100.5 + i * 0.1 for i in range(100)],
            "volume": [1000 + i * 10 for i in range(100)],
        }
    )


@pytest.fixture
def sample_bar() -> dict:
    return {
        "timestamp": pd.Timestamp("2024-01-01 12:00:00"),
        "open": 100.0,
        "high": 101.5,
        "low": 99.5,
        "close": 101.0,
        "volume": 5000.0,
    }


@pytest.fixture
def sample_bars() -> list[dict]:
    """50 sequential bars with a simple trend."""
    bars = []
    for i in range(50):
        close = 100 + i * 0.3 + ((-1) ** i) * 0.5
        bars.append(
            {
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1000 + i * 50,
            }
        )
    return bars


@pytest.fixture
def app():
    """Create a fresh FastAPI app for testing (no real MLflow/data source)."""
    import time

    from fastapi import FastAPI

    from src.api.routes import health, metrics, predict
    from src.features.engineering import FeatureEngine
    from src.models.evaluation import LiveEvaluator
    from src.models.online_model import ChampionModel

    test_app = FastAPI()
    test_app.include_router(predict.router)
    test_app.include_router(health.router)
    test_app.include_router(metrics.router)

    test_app.state.model = ChampionModel._fallback()
    test_app.state.data_source = _MockDataSource()
    test_app.state.evaluator = LiveEvaluator(log_interval=999999)
    test_app.state.feature_engine = FeatureEngine()
    test_app.state.start_time = time.time()

    return test_app


@pytest.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class _MockDataSource:
    """Fake data source that returns deterministic data."""

    async def fetch_ohlcv(self, symbol, interval, outputsize=500):
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.5] * 10,
                "volume": [1000.0] * 10,
            }
        )

    async def fetch_latest(self, symbol, interval):
        return {
            "timestamp": pd.Timestamp("2024-01-01 12:00:00"),
            "open": 100.0,
            "high": 101.5,
            "low": 99.5,
            "close": 101.0,
            "volume": 5000.0,
        }
