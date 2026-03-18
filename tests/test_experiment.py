"""Tests for the MLflow experiment runner."""

from __future__ import annotations

import mlflow
import pytest

from src.config.settings import get_settings
from src.models.experiment import run_experiment
from src.models.model_zoo import expand_grid


def _make_data_stream(n: int = 200) -> list[tuple[dict, str]]:
    """Generate synthetic feature/label pairs."""
    import random

    rng = random.Random(42)
    labels = ["buy", "hold", "sell"]
    stream = []
    for _ in range(n):
        x = {
            "rsi": rng.uniform(20, 80),
            "macd": rng.uniform(-2, 2),
            "macd_signal": rng.uniform(-1, 1),
            "macd_histogram": rng.uniform(-1, 1),
            "bb_width": rng.uniform(0, 5),
            "bb_position": rng.uniform(0, 1),
            "ema_cross": rng.uniform(-2, 2),
            "volume_ratio": rng.uniform(0.5, 2),
            "price_change_pct_1": rng.uniform(-3, 3),
            "price_change_pct_5": rng.uniform(-5, 5),
            "hl_spread": rng.uniform(0, 3),
        }
        y = rng.choice(labels)
        stream.append((x, y))
    return stream


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Clear the lru_cache on get_settings so each test gets fresh config."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestRunExperiment:
    def test_run_single_model(self, tmp_path, monkeypatch):
        """Run experiment with just GaussianNB (fast, no hyperparams)."""
        db_path = str(tmp_path / "mlflow.db")
        tracking_uri = f"sqlite:///{db_path}"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

        mini_zoo = {
            "gaussian_nb": {
                "factory": lambda **kw: __import__(
                    "river.compose", fromlist=["Pipeline"]
                ).Pipeline(
                    __import__(
                        "river.preprocessing",
                        fromlist=["StandardScaler"],
                    ).StandardScaler(),
                    __import__(
                        "river.naive_bayes", fromlist=["GaussianNB"]
                    ).GaussianNB(),
                ),
                "param_grid": {},
            },
        }

        mlflow.set_tracking_uri(tracking_uri)
        data = _make_data_stream(100)
        champion_id = run_experiment(
            experiment_name="test-experiment",
            data_stream=data,
            zoo=mini_zoo,
            metric_window=50,
        )
        assert isinstance(champion_id, str)
        assert len(champion_id) > 0

        client = mlflow.tracking.MlflowClient()
        run = client.get_run(champion_id)
        assert run.data.tags.get("champion") == "true"
        assert "weighted_f1" in run.data.metrics


class TestExpandGrid:
    def test_empty_grid(self):
        assert expand_grid({}) == [{}]

    def test_single_value_per_param(self):
        result = expand_grid({"a": [1], "b": [2]})
        assert result == [{"a": 1, "b": 2}]
