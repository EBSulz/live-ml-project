"""Prequential (test-then-train) evaluation and live metric tracking."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
from river import metrics as river_metrics

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class PrequentialEvaluator:
    """Evaluate an online model using prequential (interleaved test-then-train).

    For each data point the caller should:
      1. ``y_pred = model.predict_one(x)``
      2. ``evaluator.update(y_true, y_pred)``
      3. ``model.learn_one(x, y_true)``
    """

    def __init__(self) -> None:
        self.accuracy = river_metrics.Accuracy()
        self.f1_weighted = river_metrics.WeightedF1()
        self.f1_macro = river_metrics.MacroF1()
        self.kappa = river_metrics.CohenKappa()
        self._step = 0

    def update(self, y_true: str, y_pred: str) -> None:
        self.accuracy.update(y_true, y_pred)
        self.f1_weighted.update(y_true, y_pred)
        self.f1_macro.update(y_true, y_pred)
        self.kappa.update(y_true, y_pred)
        self._step += 1

    @property
    def step(self) -> int:
        return self._step

    def get_metrics(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy.get(),
            "weighted_f1": self.f1_weighted.get(),
            "macro_f1": self.f1_macro.get(),
            "cohen_kappa": self.kappa.get(),
        }

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]


class LiveEvaluator:
    """Wraps ``PrequentialEvaluator`` and periodically logs to MLflow."""

    def __init__(
        self,
        log_interval: int | None = None,
        experiment_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self.log_interval = log_interval or settings.mlflow_log_interval
        self._experiment_name = (
            experiment_name or settings.mlflow_live_experiment_name
        )
        self._evaluator = PrequentialEvaluator()
        self._run_id: str | None = None

    def _ensure_run(self) -> None:
        if self._run_id is not None:
            return
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        run = mlflow.start_run(run_name="live-monitoring")
        self._run_id = run.info.run_id

    def update(self, y_true: str, y_pred: str) -> dict[str, float]:
        """Record one prediction and return current metrics."""
        self._evaluator.update(y_true, y_pred)
        current = self._evaluator.get_metrics()

        if self._evaluator.step % self.log_interval == 0:
            self._log_metrics(current)

        return current

    def _log_metrics(self, current: dict[str, float]) -> None:
        try:
            self._ensure_run()
            for name, value in current.items():
                mlflow.log_metric(
                    name, value, step=self._evaluator.step
                )
        except Exception:
            logger.exception("Failed to log metrics to MLflow")

    def check_drift(self) -> bool:
        """Return ``True`` if model quality has dropped below threshold."""
        settings = get_settings()
        if self._evaluator.step < settings.drift_window:
            return False
        return self._evaluator.get_metrics()["weighted_f1"] < settings.min_f1_threshold

    def flush(self) -> None:
        """Log final metrics and end the MLflow run."""
        if self._run_id is None:
            return
        self._log_metrics(self._evaluator.get_metrics())
        mlflow.end_run()
        self._run_id = None

    @property
    def metrics(self) -> dict[str, float]:
        return self._evaluator.get_metrics()

    @property
    def step(self) -> int:
        return self._evaluator.step
