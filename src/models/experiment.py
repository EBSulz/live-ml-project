"""MLflow experiment runner – multi-model comparison with prequential evaluation."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from river import metrics as river_metrics  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

from src.config.settings import get_settings  # noqa: E402
from src.models.model_zoo import MODEL_ZOO, expand_grid  # noqa: E402

logger = logging.getLogger(__name__)


def _param_to_str(v: Any) -> str:
    """Convert a param value to a loggable string."""
    if hasattr(v, "__class__"):
        return repr(v)
    return str(v)


def _run_single_model(
    model: Any,
    model_name: str,
    params: dict,
    data_stream: list[tuple[dict, str]],
    metric_window: int,
) -> dict[str, Any]:
    """Evaluate one model via prequential evaluation and log to MLflow."""

    accuracy = river_metrics.Accuracy()
    f1_weighted = river_metrics.WeightedF1()
    f1_macro = river_metrics.MacroF1()
    kappa = river_metrics.CohenKappa()

    window_acc = river_metrics.Accuracy()
    window_f1 = river_metrics.WeightedF1()

    y_trues: list[str] = []
    y_preds: list[str] = []
    labels = ["buy", "hold", "sell"]

    with mlflow.start_run(run_name=f"{model_name}__{_params_tag(params)}") as run:
        mlflow.set_tag("model_family", model_name)

        # Log params
        mlflow.log_param("model_name", model_name)
        for k, v in params.items():
            mlflow.log_param(k, _param_to_str(v))

        for step, (x, y_true) in enumerate(data_stream, 1):
            y_pred = model.predict_one(x)
            if y_pred is None:
                y_pred = "hold"

            accuracy.update(y_true, y_pred)
            f1_weighted.update(y_true, y_pred)
            f1_macro.update(y_true, y_pred)
            kappa.update(y_true, y_pred)
            window_acc.update(y_true, y_pred)
            window_f1.update(y_true, y_pred)

            y_trues.append(y_true)
            y_preds.append(y_pred)

            model.learn_one(x, y_true)

            if step % metric_window == 0:
                mlflow.log_metric("rolling_accuracy", window_acc.get(), step=step)
                mlflow.log_metric("rolling_weighted_f1", window_f1.get(), step=step)
                window_acc = river_metrics.Accuracy()
                window_f1 = river_metrics.WeightedF1()

        # Final metrics
        final = {
            "accuracy": accuracy.get(),
            "weighted_f1": f1_weighted.get(),
            "macro_f1": f1_macro.get(),
            "cohen_kappa": kappa.get(),
        }
        mlflow.log_metrics(final)

        # Artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))

            report_path = Path(tmpdir) / "classification_report.json"
            report_path.write_text(json.dumps(final, indent=2))
            mlflow.log_artifact(str(report_path))

            try:
                fig, ax = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay.from_predictions(
                    y_trues, y_preds, labels=labels, ax=ax, cmap="Blues"
                )
                ax.set_title(f"{model_name} Confusion Matrix")
                cm_path = Path(tmpdir) / "confusion_matrix.png"
                fig.savefig(cm_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                mlflow.log_artifact(str(cm_path))
            except Exception:
                logger.warning("Could not generate confusion matrix", exc_info=True)

        return {"run_id": run.info.run_id, "metrics": final}


def _params_tag(params: dict) -> str:
    parts = [f"{k}={_param_to_str(v)}" for k, v in sorted(params.items())]
    tag = "__".join(parts) if parts else "default"
    return tag[:250]


def run_experiment(
    experiment_name: str,
    data_stream: list[tuple[dict, str]],
    zoo: dict | None = None,
    metric_window: int = 500,
    max_combos: int | None = None,
) -> str:
    """Run a full multi-model experiment and return the champion run_id.

    Parameters
    ----------
    experiment_name:
        MLflow experiment name (created if it doesn't exist).
    data_stream:
        List of ``(features_dict, label)`` tuples.
    zoo:
        Model zoo dict (defaults to ``MODEL_ZOO``).
    metric_window:
        How often to log rolling metrics.
    max_combos:
        Cap the number of hyperparam combos per model family.
    """
    if zoo is None:
        zoo = MODEL_ZOO

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    results: list[dict[str, Any]] = []

    for model_name, config in zoo.items():
        factory = config["factory"]
        combos = expand_grid(config["param_grid"], max_combos=max_combos)

        for params in combos:
            logger.info(
                "Running %s with %s (%d data points)",
                model_name,
                params,
                len(data_stream),
            )
            model = factory(**params)
            result = _run_single_model(
                model, model_name, params, data_stream, metric_window
            )
            result["model_name"] = model_name
            results.append(result)
            logger.info(
                "  -> weighted_f1=%.4f, accuracy=%.4f",
                result["metrics"]["weighted_f1"],
                result["metrics"]["accuracy"],
            )

    # Select champion
    best = max(results, key=lambda r: r["metrics"]["weighted_f1"])
    champion_run_id = best["run_id"]

    client = MlflowClient()
    client.set_tag(champion_run_id, "champion", "true")
    logger.info(
        "Champion: %s (run=%s, F1=%.4f)",
        best["model_name"],
        champion_run_id,
        best["metrics"]["weighted_f1"],
    )
    return champion_run_id
