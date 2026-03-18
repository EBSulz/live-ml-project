"""Champion model wrapper – loads the best model from MLflow and exposes
a unified predict/learn interface for the FastAPI serving layer."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class ChampionModel:
    """Thin wrapper around whatever river pipeline won the experiment."""

    def __init__(
        self,
        model: Any,
        run_id: str = "unknown",
        model_name: str = "unknown",
        metrics: dict | None = None,
    ):
        self.model = model
        self.run_id = run_id
        self.model_name = model_name
        self.metrics = metrics or {}

    # --- Factory loaders ---

    @classmethod
    def from_mlflow(
        cls, experiment_name: str | None = None
    ) -> "ChampionModel":
        """Load the champion run from the MLflow tracking server."""
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        experiment_name = experiment_name or settings.mlflow_experiment_name

        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(
                "Experiment '%s' not found – returning fresh default model",
                experiment_name,
            )
            return cls._fallback()

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.champion = 'true'",
            order_by=["metrics.weighted_f1 DESC"],
            max_results=1,
        )
        if not runs:
            logger.warning("No champion run found – returning fresh default model")
            return cls._fallback()

        run = runs[0]
        artifact_path = client.download_artifacts(run.info.run_id, "model.pkl")
        model = joblib.load(artifact_path)
        model_name = run.data.tags.get("model_family", "unknown")

        logger.info(
            "Loaded champion '%s' (run %s, F1=%.4f)",
            model_name,
            run.info.run_id,
            run.data.metrics.get("weighted_f1", 0),
        )
        return cls(
            model=model,
            run_id=run.info.run_id,
            model_name=model_name,
            metrics=run.data.metrics,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ChampionModel":
        """Load a model directly from a local file."""
        model = joblib.load(path)
        return cls(model=model, run_id="local", model_name="local_file")

    @classmethod
    def _fallback(cls) -> "ChampionModel":
        """Return a fresh default model when no champion exists yet."""
        from river import compose, ensemble, preprocessing, tree

        model = compose.Pipeline(
            preprocessing.StandardScaler(),
            ensemble.ADWINBaggingClassifier(
                model=tree.HoeffdingTreeClassifier(), n_models=10, seed=42
            ),
        )
        return cls(model=model, run_id="fallback", model_name="adwin_bagging_default")

    # --- Inference / Learning ---

    def predict_one(self, x: dict) -> str:
        pred = self.model.predict_one(x)
        return pred if pred is not None else "hold"

    def predict_proba_one(self, x: dict) -> dict[str, float]:
        proba = self.model.predict_proba_one(x)
        if not proba:
            return {"buy": 0.0, "hold": 1.0, "sell": 0.0}
        return {str(k): float(v) for k, v in proba.items()}

    def learn_one(self, x: dict, y: str) -> None:
        self.model.learn_one(x, y)

    # --- Serialization ---

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)
        logger.info("Model saved to %s", path)

    def to_bytes(self) -> bytes:
        return pickle.dumps(self.model)
