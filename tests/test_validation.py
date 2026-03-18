"""Tests for the data validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.data_quality import validate_ohlcv
from src.validation.data_drift import check_feature_drift, _psi


class TestDataQuality:
    def test_valid_df(self, sample_ohlcv_df):
        result = validate_ohlcv(sample_ohlcv_df)
        assert result["valid"] is True
        assert result["failures"] == []

    def test_missing_column(self):
        df = pd.DataFrame({"timestamp": [1], "open": [1], "high": [1]})
        result = validate_ohlcv(df)
        assert result["valid"] is False
        assert any("Missing columns" in f for f in result["failures"])

    def test_null_values(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.loc[0, "close"] = None
        result = validate_ohlcv(df)
        assert result["valid"] is False
        assert any("null" in f for f in result["failures"])

    def test_negative_price(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.loc[0, "close"] = -1.0
        result = validate_ohlcv(df)
        assert result["valid"] is False
        assert any("non-positive" in f for f in result["failures"])

    def test_negative_volume(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.loc[0, "volume"] = -100
        result = validate_ohlcv(df)
        assert result["valid"] is False

    def test_raise_on_fail(self):
        df = pd.DataFrame({"timestamp": [1], "open": [1]})
        with pytest.raises(ValueError, match="Missing columns"):
            validate_ohlcv(df, raise_on_fail=True)


class TestDataDrift:
    def test_no_drift_identical(self):
        df = pd.DataFrame({"feat_a": np.random.randn(100), "feat_b": np.random.randn(100)})
        result = check_feature_drift(df, df)
        assert result["drifted"] is False

    def test_drift_detected(self):
        ref = pd.DataFrame({"feat_a": np.random.randn(200)})
        cur = pd.DataFrame({"feat_a": np.random.randn(200) + 5})
        result = check_feature_drift(ref, cur, psi_threshold=0.1)
        assert result["drifted"] is True
        assert "feat_a" in result["drifted_features"]

    def test_psi_identical_distributions(self):
        a = np.random.randn(1000)
        psi = _psi(a, a)
        assert psi < 0.05

    def test_psi_different_distributions(self):
        a = np.random.randn(1000)
        b = np.random.randn(1000) + 10
        psi = _psi(a, b)
        assert psi > 0.2
