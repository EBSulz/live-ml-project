"""Tests for the feature engineering module."""

from __future__ import annotations

from src.features.engineering import FeatureEngine, compute_features_batch

EXPECTED_FEATURES = {
    "price_change_pct_1",
    "price_change_pct_5",
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_width",
    "bb_position",
    "ema_cross",
    "volume_ratio",
    "hl_spread",
}


class TestFeatureEngine:
    def test_single_bar_returns_all_features(self, sample_bar):
        engine = FeatureEngine()
        features = engine.update(sample_bar)
        assert EXPECTED_FEATURES.issubset(features.keys())

    def test_all_features_are_float(self, sample_bars):
        engine = FeatureEngine()
        for bar in sample_bars:
            features = engine.update(bar)
            for key, val in features.items():
                assert isinstance(val, float), f"{key} is not float: {type(val)}"

    def test_rsi_bounded(self, sample_bars):
        engine = FeatureEngine()
        for bar in sample_bars:
            features = engine.update(bar)
        assert 0 <= features["rsi"] <= 100

    def test_volume_ratio_positive(self, sample_bars):
        engine = FeatureEngine()
        for bar in sample_bars:
            features = engine.update(bar)
        assert features["volume_ratio"] > 0

    def test_price_change_first_bar_is_zero(self, sample_bar):
        engine = FeatureEngine()
        features = engine.update(sample_bar)
        assert features["price_change_pct_1"] == 0.0


class TestBatchFeatures:
    def test_batch_returns_correct_shape(self, sample_ohlcv_df):
        feat_df = compute_features_batch(sample_ohlcv_df)
        assert len(feat_df) == len(sample_ohlcv_df)
        assert "timestamp" in feat_df.columns
        assert EXPECTED_FEATURES.issubset(set(feat_df.columns) - {"timestamp"})

    def test_batch_no_nans_after_warmup(self, sample_ohlcv_df):
        feat_df = compute_features_batch(sample_ohlcv_df)
        tail = feat_df.tail(50)
        feature_cols = [c for c in tail.columns if c != "timestamp"]
        assert not tail[feature_cols].isna().any().any()
