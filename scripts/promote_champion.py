#!/usr/bin/env python
"""Promote the MLflow champion model to Production and optionally to Vertex AI.

Usage:
    python scripts/promote_champion.py --experiment-name signal-classification-v1
    python scripts/promote_champion.py --experiment-name signal-classification-v1 --vertex
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote champion model")
    parser.add_argument("--experiment-name", default="signal-classification-v1")
    parser.add_argument(
        "--vertex", action="store_true", help="Also register in Vertex AI"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.models.registry import (
        promote_to_production,
        register_champion_in_mlflow,
        promote_champion_to_vertex,
    )

    import mlflow
    from mlflow.tracking import MlflowClient
    from src.config.settings import get_settings

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        logger.error("Experiment '%s' not found", args.experiment_name)
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.champion = 'true'",
        order_by=["metrics.weighted_f1 DESC"],
        max_results=1,
    )
    if not runs:
        logger.error("No champion run found")
        return

    run = runs[0]
    logger.info(
        "Champion: run=%s, F1=%.4f",
        run.info.run_id,
        run.data.metrics.get("weighted_f1", 0),
    )

    version = register_champion_in_mlflow(run.info.run_id)
    logger.info("Registered as v%d in Staging", version)

    promote_to_production(version=version)
    logger.info("Promoted v%d to Production", version)

    if args.vertex:
        resource = promote_champion_to_vertex(args.experiment_name)
        logger.info("Registered in Vertex AI: %s", resource)

    logger.info("Promotion complete.")


if __name__ == "__main__":
    main()
