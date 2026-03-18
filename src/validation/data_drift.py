"""Data drift detection using Deepchecks."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions."""
    eps = 1e-8
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        bins + 1,
    )
    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

    ref_counts = np.clip(ref_counts, eps, None)
    cur_counts = np.clip(cur_counts, eps, None)

    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def check_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    psi_threshold: float = 0.2,
) -> dict[str, Any]:
    """Compare feature distributions between a reference and current window.

    Parameters
    ----------
    reference_df:
        Historical baseline features.
    current_df:
        Recent feature window.
    feature_columns:
        Which columns to check (defaults to all numeric).
    psi_threshold:
        PSI above this value is flagged as significant drift.

    Returns
    -------
    dict with ``drifted`` (bool), ``details`` (per-feature PSI), ``drifted_features``.
    """
    if feature_columns is None:
        feature_columns = list(
            reference_df.select_dtypes(include=[np.number]).columns
        )

    details: dict[str, float] = {}
    drifted_features: list[str] = []

    for col in feature_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref = reference_df[col].dropna().values
        cur = current_df[col].dropna().values
        if len(ref) < 10 or len(cur) < 10:
            continue

        psi_value = _psi(ref, cur)
        details[col] = psi_value
        if psi_value > psi_threshold:
            drifted_features.append(col)

    has_drift = len(drifted_features) > 0
    if has_drift:
        logger.warning(
            "Feature drift detected in: %s (PSI values: %s)",
            drifted_features,
            {f: f"{details[f]:.4f}" for f in drifted_features},
        )

    return {
        "drifted": has_drift,
        "details": details,
        "drifted_features": drifted_features,
    }
