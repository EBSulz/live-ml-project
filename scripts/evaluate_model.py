#!/usr/bin/env python
"""CI quality gate – evaluate the champion model and fail if below threshold.

Exit code 0 = pass, 1 = fail.

Usage:
    python scripts/evaluate_model.py --experiment-name signal-classification-v1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI model quality gate")
    parser.add_argument("--experiment-name", default="signal-classification-v1")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--outputsize", type=int, default=1000)
    parser.add_argument("--min-f1", type=float, default=None)
    parser.add_argument("--min-accuracy", type=float, default=None)
    return parser.parse_args()


async def fetch_data(symbol: str, interval: str, outputsize: int):
    from src.data_ingestion.factory import get_data_source

    source = get_data_source()
    return await source.fetch_ohlcv(symbol, interval, outputsize)


def main() -> None:
    args = parse_args()

    from src.config.settings import get_settings
    from src.features.engineering import FeatureEngine
    from src.models.evaluation import PrequentialEvaluator
    from src.models.online_model import ChampionModel
    from src.models.signal_labeler import label_signal

    settings = get_settings()
    min_f1 = args.min_f1 or settings.min_f1_threshold
    min_accuracy = args.min_accuracy or settings.min_accuracy_threshold

    logger.info("Loading champion model from experiment '%s'...", args.experiment_name)
    model = ChampionModel.from_mlflow(args.experiment_name)
    logger.info("Model: %s (run=%s)", model.model_name, model.run_id)

    logger.info("Fetching evaluation data (%d bars)...", args.outputsize)
    df = asyncio.run(fetch_data(args.symbol, args.interval, args.outputsize))

    engine = FeatureEngine()
    evaluator = PrequentialEvaluator()
    prev_close = None

    for _, row in df.iterrows():
        bar = row.to_dict()
        features = engine.update(bar)

        if prev_close is not None and prev_close != 0:
            pct = (bar["close"] - prev_close) / prev_close * 100
            y_true = label_signal(pct)
            y_pred = model.predict_one(features)
            evaluator.update(y_true, y_pred)
            model.learn_one(features, y_true)

        prev_close = bar["close"]

    metrics = evaluator.get_metrics()
    logger.info("Evaluation results:")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    passed = True
    if metrics["weighted_f1"] < min_f1:
        logger.error(
            "FAIL: weighted_f1 (%.4f) < threshold (%.4f)",
            metrics["weighted_f1"],
            min_f1,
        )
        passed = False

    if metrics["accuracy"] < min_accuracy:
        logger.error(
            "FAIL: accuracy (%.4f) < threshold (%.4f)",
            metrics["accuracy"],
            min_accuracy,
        )
        passed = False

    if passed:
        logger.info("PASS: Model quality gate passed")
        sys.exit(0)
    else:
        logger.error("FAIL: Model quality gate failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
