"""Incremental and batch feature engineering for OHLCV data.

Provides both:
- ``FeatureEngine``: stateful, processes one bar at a time (river-compatible).
- ``compute_features_batch``: pandas-based bulk computation for bootstrap.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Streaming (one-bar-at-a-time) feature engine
# ---------------------------------------------------------------------------


@dataclass
class FeatureEngine:
    """Incrementally computes technical indicators from OHLCV bars.

    Call ``update(bar)`` for each new bar; it returns a flat ``dict`` of
    features suitable for ``model.predict_one`` / ``model.learn_one``.
    """

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    ema_short: int = 9
    ema_long: int = 21
    vol_period: int = 20

    _closes: deque = field(default_factory=lambda: deque(maxlen=200))
    _volumes: deque = field(default_factory=lambda: deque(maxlen=200))
    _gains: deque = field(default_factory=lambda: deque(maxlen=200))
    _losses: deque = field(default_factory=lambda: deque(maxlen=200))
    _prev_close: float | None = field(default=None, repr=False)
    _ema_fast_val: float | None = field(default=None, repr=False)
    _ema_slow_val: float | None = field(default=None, repr=False)
    _ema_signal_val: float | None = field(default=None, repr=False)
    _ema_short_val: float | None = field(default=None, repr=False)
    _ema_long_val: float | None = field(default=None, repr=False)

    # --- helpers ---

    @staticmethod
    def _ema_step(prev: float | None, value: float, period: int) -> float:
        if prev is None:
            return value
        k = 2.0 / (period + 1)
        return value * k + prev * (1 - k)

    # --- public API ---

    def update(self, bar: dict) -> dict:
        """Ingest one OHLCV bar and return the feature vector."""
        close = float(bar["close"])
        volume = float(bar["volume"])
        high = float(bar["high"])
        low = float(bar["low"])

        self._closes.append(close)
        self._volumes.append(volume)

        features: dict = {}

        # Price change %
        if self._prev_close is not None and self._prev_close != 0:
            pct = (close - self._prev_close) / self._prev_close * 100
        else:
            pct = 0.0
        features["price_change_pct_1"] = pct

        if len(self._closes) >= 6:
            c5 = self._closes[-6]
            features["price_change_pct_5"] = (close - c5) / c5 * 100 if c5 != 0 else 0.0
        else:
            features["price_change_pct_5"] = 0.0

        # RSI
        if self._prev_close is not None:
            delta = close - self._prev_close
            self._gains.append(max(delta, 0.0))
            self._losses.append(max(-delta, 0.0))

        if len(self._gains) >= self.rsi_period:
            avg_gain = np.mean(list(self._gains)[-self.rsi_period :])
            avg_loss = np.mean(list(self._losses)[-self.rsi_period :])
            if avg_loss == 0:
                features["rsi"] = 100.0
            else:
                rs = avg_gain / avg_loss
                features["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        else:
            features["rsi"] = 50.0

        # MACD
        self._ema_fast_val = self._ema_step(self._ema_fast_val, close, self.macd_fast)
        self._ema_slow_val = self._ema_step(self._ema_slow_val, close, self.macd_slow)
        macd_line = (self._ema_fast_val or 0) - (self._ema_slow_val or 0)
        self._ema_signal_val = self._ema_step(
            self._ema_signal_val, macd_line, self.macd_signal
        )
        features["macd"] = macd_line
        features["macd_signal"] = self._ema_signal_val or 0.0
        features["macd_histogram"] = macd_line - (self._ema_signal_val or 0.0)

        # Bollinger Band width
        if len(self._closes) >= self.bb_period:
            window = list(self._closes)[-self.bb_period :]
            sma = np.mean(window)
            std = np.std(window, ddof=1)
            features["bb_width"] = (4 * std / sma * 100) if sma != 0 else 0.0
            features["bb_position"] = (
                ((close - (sma - 2 * std)) / (4 * std)) if std != 0 else 0.5
            )
        else:
            features["bb_width"] = 0.0
            features["bb_position"] = 0.5

        # EMA crossover
        self._ema_short_val = self._ema_step(self._ema_short_val, close, self.ema_short)
        self._ema_long_val = self._ema_step(self._ema_long_val, close, self.ema_long)
        features["ema_cross"] = (self._ema_short_val or 0) - (self._ema_long_val or 0)

        # Volume ratio
        if len(self._volumes) >= self.vol_period:
            avg_vol = np.mean(list(self._volumes)[-self.vol_period :])
            features["volume_ratio"] = volume / avg_vol if avg_vol != 0 else 1.0
        else:
            features["volume_ratio"] = 1.0

        # High-low spread (volatility proxy)
        features["hl_spread"] = ((high - low) / close * 100) if close != 0 else 0.0

        self._prev_close = close
        return features


# ---------------------------------------------------------------------------
# Batch (pandas) feature computation – used for bootstrap / experiments
# ---------------------------------------------------------------------------


def compute_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features over a full OHLCV DataFrame at once.

    Expects columns: [timestamp, open, high, low, close, volume].
    Returns a new DataFrame with one row per input row and feature columns.
    Rows that lack sufficient look-back are filled with neutral defaults.
    """
    engine = FeatureEngine()
    records = []
    for _, row in df.iterrows():
        bar = row.to_dict()
        features = engine.update(bar)
        features["timestamp"] = bar["timestamp"]
        records.append(features)

    feat_df = pd.DataFrame(records)
    feature_cols = [c for c in feat_df.columns if c != "timestamp"]
    feat_df[feature_cols] = feat_df[feature_cols].astype(float)
    return feat_df
