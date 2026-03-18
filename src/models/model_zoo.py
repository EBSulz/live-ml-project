"""Registry of candidate online classifiers with hyperparameter grids.

Each entry maps a human-readable name to a dict containing:
- ``factory``: callable(**kwargs) -> river pipeline
- ``param_grid``: dict of param_name -> list[values]

The experiment runner expands the grid into all combinations and evaluates
each one as a separate MLflow Run.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

from river import (
    compose,
    ensemble,
    forest,
    linear_model,
    multiclass,
    naive_bayes,
    optim,
    preprocessing,
    tree,
)


def _make_hoeffding_tree(**kw: Any) -> compose.Pipeline:
    return compose.Pipeline(
        preprocessing.StandardScaler(), tree.HoeffdingTreeClassifier(**kw)
    )


def _make_arf(**kw: Any) -> compose.Pipeline:
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        forest.ARFClassifier(**kw),
    )


def _make_adwin_bagging(**kw: Any) -> compose.Pipeline:
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        ensemble.ADWINBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(), **kw
        ),
    )


def _make_logistic_regression(**kw: Any) -> compose.Pipeline:
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        multiclass.OneVsRestClassifier(linear_model.LogisticRegression(**kw)),
    )


def _make_gaussian_nb(**kw: Any) -> compose.Pipeline:
    return compose.Pipeline(
        preprocessing.StandardScaler(), naive_bayes.GaussianNB()
    )


MODEL_ZOO: dict[str, dict[str, Any]] = {
    "hoeffding_tree": {
        "factory": _make_hoeffding_tree,
        "param_grid": {
            "grace_period": [50, 100, 200],
            "delta": [1e-5, 1e-7],
            "leaf_prediction": ["mc", "nb"],
        },
    },
    "adaptive_random_forest": {
        "factory": _make_arf,
        "param_grid": {
            "n_models": [5, 10, 20],
            "grace_period": [50, 100],
            "seed": [42],
        },
    },
    "adwin_bagging": {
        "factory": _make_adwin_bagging,
        "param_grid": {
            "n_models": [5, 10, 15],
            "seed": [42],
        },
    },
    "logistic_regression": {
        "factory": _make_logistic_regression,
        "param_grid": {
            "optimizer": [optim.SGD(0.01), optim.SGD(0.001), optim.Adam()],
            "l2": [0.0, 0.001, 0.01],
        },
    },
    "gaussian_nb": {
        "factory": _make_gaussian_nb,
        "param_grid": {},
    },
}


def expand_grid(
    param_grid: dict[str, list],
    max_combos: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Expand a hyperparameter grid into a list of flat dicts.

    If *max_combos* is set and the full grid is larger, randomly sample.
    """
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    if max_combos is not None and len(combos) > max_combos:
        rng = random.Random(seed)
        combos = rng.sample(combos, max_combos)

    return combos
