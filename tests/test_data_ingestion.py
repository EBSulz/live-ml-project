"""Tests for the data ingestion module."""

from __future__ import annotations

import pytest

from src.data_ingestion.base import DataSource
from src.data_ingestion.factory import get_data_source
from src.data_ingestion.twelve_data import TwelveDataSource
from src.data_ingestion.alpha_vantage import AlphaVantageSource


class TestFactory:
    def test_get_twelve_data(self):
        source = get_data_source("twelve_data")
        assert isinstance(source, TwelveDataSource)

    def test_get_alpha_vantage(self):
        source = get_data_source("alpha_vantage")
        assert isinstance(source, AlphaVantageSource)

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown data source"):
            get_data_source("nonexistent")

    def test_all_sources_implement_abc(self):
        for name in ("twelve_data", "alpha_vantage"):
            source = get_data_source(name)
            assert isinstance(source, DataSource)


class TestTwelveDataNormalize:
    def test_normalize_valid(self):
        raw = [
            {
                "datetime": "2024-01-01 12:00:00",
                "open": "100.0",
                "high": "101.0",
                "low": "99.0",
                "close": "100.5",
                "volume": "1000",
            },
            {
                "datetime": "2024-01-01 11:00:00",
                "open": "99.5",
                "high": "100.5",
                "low": "98.5",
                "close": "100.0",
                "volume": "900",
            },
        ]
        df = TwelveDataSource._normalize(raw)
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert len(df) == 2
        assert df.iloc[0]["timestamp"] < df.iloc[1]["timestamp"]


class TestAlphaVantageExtract:
    def test_extract_intraday_series(self):
        data = {
            "Meta Data": {},
            "Time Series (60min)": {
                "2024-01-01 12:00:00": {
                    "1. open": "100.0",
                    "2. high": "101.0",
                    "3. low": "99.0",
                    "4. close": "100.5",
                    "5. volume": "1000",
                },
            },
        }
        rows = AlphaVantageSource._extract_series(data)
        assert len(rows) == 1
        assert rows[0]["open"] == "100.0"

    def test_extract_raises_on_missing_series(self):
        with pytest.raises(ValueError, match="no time series found"):
            AlphaVantageSource._extract_series({"Meta Data": {}})
