"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


@pytest.mark.asyncio
async def test_readiness_endpoint(client: AsyncClient):
    resp = await client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


@pytest.mark.asyncio
async def test_metrics_endpoint(client: AsyncClient):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "accuracy" in data
    assert "weighted_f1" in data
    assert "predictions_count" in data


@pytest.mark.asyncio
async def test_predict_endpoint(client: AsyncClient):
    resp = await client.post("/predict", json={"symbol": "BTC/USD", "interval": "1h"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "BTC/USD"
    assert data["signal"] in ("buy", "hold", "sell")
    assert "confidence" in data
    assert "timestamp" in data
    assert "model_version" in data


@pytest.mark.asyncio
async def test_predict_invalid_symbol(client: AsyncClient):
    resp = await client.post("/predict", json={"symbol": "invalid!", "interval": "1h"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_predict_invalid_interval(client: AsyncClient):
    resp = await client.post("/predict", json={"symbol": "BTC/USD", "interval": "bad"})
    assert resp.status_code == 422
