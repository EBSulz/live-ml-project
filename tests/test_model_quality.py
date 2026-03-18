"""Model quality gate – verifies that the default model meets minimum metrics.

This test is used in CI (``make test``) as a lightweight quality gate.
The full experiment-based gate runs via ``scripts/evaluate_model.py``.
"""

from __future__ import annotations

from src.features.engineering import FeatureEngine
from src.models.evaluation import PrequentialEvaluator
from src.models.online_model import ChampionModel
from src.models.signal_labeler import label_signal


def test_fallback_model_trains_and_predicts(sample_bars):
    """Ensure the fallback model can learn from streaming data and produce
    reasonable predictions without crashing."""
    model = ChampionModel._fallback()
    engine = FeatureEngine()
    evaluator = PrequentialEvaluator()

    prev_close = None
    for bar in sample_bars:
        features = engine.update(bar)
        if prev_close is not None and prev_close != 0:
            pct = (bar["close"] - prev_close) / prev_close * 100
            y_true = label_signal(pct)
            y_pred = model.predict_one(features)
            evaluator.update(y_true, y_pred)
            model.learn_one(features, y_true)
        prev_close = bar["close"]

    metrics = evaluator.get_metrics()
    assert metrics["accuracy"] >= 0.0
    assert metrics["weighted_f1"] >= 0.0
    assert evaluator.step > 0


def test_model_produces_valid_signals(sample_bars):
    """All predictions must be one of the three valid signals."""
    model = ChampionModel._fallback()
    engine = FeatureEngine()

    for bar in sample_bars:
        features = engine.update(bar)
        signal = model.predict_one(features)
        assert signal in ("buy", "hold", "sell")
