"""Model registry integration – MLflow Model Registry + Vertex AI."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLflow Model Registry helpers
# ---------------------------------------------------------------------------


def register_champion_in_mlflow(
    run_id: str,
    model_name: str | None = None,
) -> int:
    """Register or update the champion model in MLflow Model Registry.

    Returns the new version number.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_name = model_name or settings.mlflow_model_name
    client = MlflowClient()

    artifact_uri = f"runs:/{run_id}/model.pkl"
    mv = mlflow.register_model(artifact_uri, model_name)
    version = int(mv.version)

    client.transition_model_version_stage(
        name=model_name, version=version, stage="Staging"
    )
    logger.info("Registered %s v%d in Staging", model_name, version)
    return version


def promote_to_production(
    model_name: str | None = None,
    version: int | None = None,
) -> None:
    """Promote a Staging model version to Production in MLflow."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_name = model_name or settings.mlflow_model_name
    client = MlflowClient()

    if version is None:
        staging = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging:
            raise RuntimeError(f"No Staging version found for '{model_name}'")
        version = int(staging[0].version)

    # Archive current production
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for pv in prod_versions:
        client.transition_model_version_stage(
            name=model_name, version=pv.version, stage="Archived"
        )

    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production"
    )
    logger.info("Promoted %s v%d to Production", model_name, version)


# ---------------------------------------------------------------------------
# Vertex AI Model Registry helpers
# ---------------------------------------------------------------------------


def upload_to_gcs(model_bytes: bytes, destination_blob: str) -> str:
    """Upload serialized model to GCS and return the gs:// URI."""
    from google.cloud import storage

    settings = get_settings()
    client = storage.Client(project=settings.gcp_project_id)
    bucket = client.bucket(settings.gcs_model_bucket)
    blob = bucket.blob(destination_blob)
    blob.upload_from_string(model_bytes)
    uri = f"gs://{settings.gcs_model_bucket}/{destination_blob}"
    logger.info("Uploaded model to %s", uri)
    return uri


def register_in_vertex_ai(
    artifact_uri: str,
    display_name: str = "market-predictor",
    description: str = "",
) -> str:
    """Register a model in Vertex AI Model Registry and return the resource name."""
    from google.cloud import aiplatform

    settings = get_settings()
    aiplatform.init(
        project=settings.gcp_project_id,
        location=settings.gcp_region,
        staging_bucket=settings.vertex_ai_staging_bucket,
    )

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=(
            f"{settings.gcp_region}-docker.pkg.dev/"
            f"{settings.gcp_project_id}/market-predictor/app:latest"
        ),
        description=description,
    )
    logger.info("Registered Vertex AI model: %s", model.resource_name)
    return model.resource_name


def promote_champion_to_vertex(
    experiment_name: str | None = None,
) -> str:
    """End-to-end: load MLflow champion -> upload to GCS -> register in Vertex AI."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_name = experiment_name or settings.mlflow_experiment_name

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.champion = 'true'",
        order_by=["metrics.weighted_f1 DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No champion run found")

    run = runs[0]
    artifact_path = client.download_artifacts(run.info.run_id, "model.pkl")
    model = joblib.load(artifact_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "model.pkl"
        joblib.dump(model, model_file)
        model_bytes = model_file.read_bytes()

    f1 = run.data.metrics.get("weighted_f1", 0)
    blob_name = f"models/{experiment_name}/{run.info.run_id}/model.pkl"
    upload_to_gcs(model_bytes, blob_name)

    base = f"gs://{settings.gcs_model_bucket}/models"
    artifact_uri = f"{base}/{experiment_name}/{run.info.run_id}"
    resource = register_in_vertex_ai(
        artifact_uri=artifact_uri,
        display_name="market-predictor",
        description=f"Champion from {experiment_name}, F1={f1:.4f}",
    )
    return resource
