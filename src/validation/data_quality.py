"""Data quality validation using Great Expectations."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def _build_suite() -> list[dict[str, Any]]:
    """Return a list of expectation-like checks as plain dicts.

    Uses a lightweight approach so we don't require a full GE DataContext
    for every API call, while still following the GE mental model.
    """
    return [
        {"check": "columns_exist", "columns": list(EXPECTED_COLUMNS)},
        {"check": "no_nulls", "columns": ["open", "high", "low", "close"]},
        {"check": "positive", "columns": ["close", "open", "high", "low"]},
        {"check": "non_negative", "columns": ["volume"]},
        {"check": "timestamp_parseable"},
    ]


def validate_ohlcv(df: pd.DataFrame, raise_on_fail: bool = False) -> dict[str, Any]:
    """Validate an OHLCV DataFrame and return a results dict.

    Parameters
    ----------
    df:
        DataFrame with columns [timestamp, open, high, low, close, volume].
    raise_on_fail:
        If ``True``, raise ``ValueError`` on the first failure.

    Returns
    -------
    dict with keys ``valid`` (bool), ``failures`` (list[str]).
    """
    failures: list[str] = []

    # Column existence
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        failures.append(f"Missing columns: {missing}")

    if failures and raise_on_fail:
        raise ValueError(failures[0])

    # Null checks
    for col in ["open", "high", "low", "close"]:
        if col in df.columns and df[col].isna().any():
            failures.append(f"Column '{col}' contains null values")

    # Positive price checks
    for col in ["open", "high", "low", "close"]:
        if col in df.columns and (df[col] <= 0).any():
            failures.append(f"Column '{col}' contains non-positive values")

    # Non-negative volume
    if "volume" in df.columns and (df["volume"] < 0).any():
        failures.append("Column 'volume' contains negative values")

    # Timestamp
    if "timestamp" in df.columns:
        try:
            pd.to_datetime(df["timestamp"])
        except Exception:
            failures.append("Column 'timestamp' is not parseable as datetime")

    is_valid = len(failures) == 0
    if not is_valid:
        logger.warning("Data quality issues: %s", failures)
    if not is_valid and raise_on_fail:
        raise ValueError(f"Data quality validation failed: {failures}")

    return {"valid": is_valid, "failures": failures}
