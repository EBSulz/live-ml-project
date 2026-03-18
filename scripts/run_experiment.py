#!/usr/bin/env python
"""Run a full multi-model experiment comparing all zoo classifiers.

Usage:
    python scripts/run_experiment.py \
        --experiment-name signal-classification-v1 \
        --symbol BTC/USD \
        --interval 1h \
        --outputsize 5000
"""

from __future__ import annotations

import argparse
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLflow multi-model experiment")
    parser.add_argument("--experiment-name", default="signal-classification-v1")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--outputsize", type=int, default=5000)
    parser.add_argument("--metric-window", type=int, default=500)
    parser.add_argument("--max-combos", type=int, default=None)
    parser.add_argument(
        "--register", action="store_true", help="Register champion in MLflow Registry"
    )
    return parser.parse_args()


async def fetch_data(symbol: str, interval: str, outputsize: int):
    from src.data_ingestion.factory import get_data_source

    source = get_data_source()
    return await source.fetch_ohlcv(symbol, interval, outputsize)


def main() -> None:
    args = parse_args()

    logger.info("Fetching data for %s (%s)...", args.symbol, args.interval)
    df = asyncio.run(fetch_data(args.symbol, args.interval, args.outputsize))
    logger.info("Received %d bars", len(df))

    from src.features.engineering import compute_features_batch
    from src.models.signal_labeler import label_series

    feat_df = compute_features_batch(df)
    labels = label_series(df["close"].tolist())

    feature_cols = [c for c in feat_df.columns if c != "timestamp"]
    data_stream: list[tuple[dict, str]] = []
    for i in range(len(feat_df)):
        x = {col: float(feat_df.iloc[i][col]) for col in feature_cols}
        y = labels[i]
        data_stream.append((x, y))

    # Drop the first ~30 rows (insufficient look-back for some indicators)
    data_stream = data_stream[30:]

    logger.info(
        "Running experiment '%s' with %d data points...",
        args.experiment_name,
        len(data_stream),
    )

    from src.models.experiment import run_experiment

    champion_id = run_experiment(
        experiment_name=args.experiment_name,
        data_stream=data_stream,
        metric_window=args.metric_window,
        max_combos=args.max_combos,
    )
    logger.info("Champion run ID: %s", champion_id)

    if args.register:
        from src.models.registry import register_champion_in_mlflow

        version = register_champion_in_mlflow(champion_id)
        logger.info("Registered as v%d in MLflow Model Registry", version)

    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
