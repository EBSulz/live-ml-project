"""Tests for the models module."""

from __future__ import annotations

from src.models.model_zoo import MODEL_ZOO, expand_grid
from src.models.online_model import ChampionModel
from src.models.signal_labeler import label_signal, label_series


class TestSignalLabeler:
    def test_buy_signal(self):
        assert label_signal(1.0, threshold=0.5) == "buy"

    def test_sell_signal(self):
        assert label_signal(-1.0, threshold=0.5) == "sell"

    def test_hold_signal(self):
        assert label_signal(0.2, threshold=0.5) == "hold"
        assert label_signal(-0.2, threshold=0.5) == "hold"

    def test_boundary(self):
        assert label_signal(0.5, threshold=0.5) == "hold"
        assert label_signal(-0.5, threshold=0.5) == "hold"
        assert label_signal(0.51, threshold=0.5) == "buy"

    def test_label_series(self):
        closes = [100.0, 101.0, 99.0, 100.0, 102.0]
        labels = label_series(closes, threshold=0.5)
        assert len(labels) == 5
        assert all(l in ("buy", "hold", "sell") for l in labels)
        assert labels[-1] == "hold"  # last item has no lookahead


class TestModelZoo:
    def test_all_zoo_entries_have_required_keys(self):
        for name, config in MODEL_ZOO.items():
            assert "factory" in config, f"{name} missing 'factory'"
            assert "param_grid" in config, f"{name} missing 'param_grid'"
            assert callable(config["factory"]), f"{name} factory not callable"

    def test_all_factories_produce_models(self):
        for name, config in MODEL_ZOO.items():
            combos = expand_grid(config["param_grid"], max_combos=1)
            model = config["factory"](**combos[0])
            assert hasattr(model, "predict_one")
            assert hasattr(model, "learn_one")

    def test_expand_grid_empty(self):
        assert expand_grid({}) == [{}]

    def test_expand_grid_single_param(self):
        result = expand_grid({"a": [1, 2, 3]})
        assert len(result) == 3

    def test_expand_grid_multiple_params(self):
        result = expand_grid({"a": [1, 2], "b": [3, 4]})
        assert len(result) == 4

    def test_expand_grid_max_combos(self):
        result = expand_grid({"a": list(range(100))}, max_combos=5)
        assert len(result) == 5


class TestChampionModel:
    def test_fallback_model(self):
        model = ChampionModel._fallback()
        assert model.run_id == "fallback"
        assert model.model is not None

    def test_predict_one_returns_string(self):
        model = ChampionModel._fallback()
        x = {"rsi": 50.0, "macd": 0.1, "bb_width": 2.0}
        result = model.predict_one(x)
        assert isinstance(result, str)
        assert result in ("buy", "hold", "sell")

    def test_predict_proba_one_returns_dict(self):
        model = ChampionModel._fallback()
        x = {"rsi": 50.0, "macd": 0.1, "bb_width": 2.0}
        proba = model.predict_proba_one(x)
        assert isinstance(proba, dict)

    def test_learn_one_does_not_raise(self):
        model = ChampionModel._fallback()
        x = {"rsi": 50.0, "macd": 0.1, "bb_width": 2.0}
        model.learn_one(x, "buy")
