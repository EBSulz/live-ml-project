#!/usr/bin/env python
"""Bootstrap the champion model by warm-starting it on historical data.

This is useful when deploying a fresh model that needs some initial training
before going live.  It loads the champion from MLflow, streams historical
data through it (learn_one for each bar), and saves the updated model back.

Usage:
    python scripts/bootstrap_model.py \
        --experiment-name signal-classification-v1 \
        --symbol BTC/USD \
        --interval 1h
"""

from __future__ import annotations

import argparse
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap champion model")
    parser.add_argument("--experiment-name", default="signal-classification-v1")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--outputsize", type=int, default=2000)
    parser.add_argument("--output", default="model_bootstrap.pkl")
    return parser.parse_args()


async def fetch_data(symbol: str, interval: str, outputsize: int):
    from src.data_ingestion.factory import get_data_source

    source = get_data_source()
    return await source.fetch_ohlcv(symbol, interval, outputsize)


def main() -> None:
    args = parse_args()

    from src.features.engineering import FeatureEngine
    from src.models.online_model import ChampionModel
    from src.models.signal_labeler import label_signal

    logger.info("Loading champion model...")
    model = ChampionModel.from_mlflow(args.experiment_name)
    logger.info("Model: %s (run=%s)", model.model_name, model.run_id)

    logger.info("Fetching %d bars for %s...", args.outputsize, args.symbol)
    df = asyncio.run(fetch_data(args.symbol, args.interval, args.outputsize))
    logger.info("Received %d bars", len(df))

    engine = FeatureEngine()
    prev_close = None
    learned = 0

    for _, row in df.iterrows():
        bar = row.to_dict()
        features = engine.update(bar)

        if prev_close is not None and prev_close != 0:
            pct = (bar["close"] - prev_close) / prev_close * 100
            y = label_signal(pct)
            model.learn_one(features, y)
            learned += 1

        prev_close = bar["close"]

    logger.info("Bootstrapped model with %d data points", learned)
    model.save(args.output)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
