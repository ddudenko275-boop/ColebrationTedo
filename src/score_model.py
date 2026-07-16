"""Baseline score model utilities for PD calibration experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold


EPS = 1e-6


def make_boosting_score_model(random_state: int = 42) -> HistGradientBoostingClassifier:
    """Return the baseline boosting model used to produce raw PD scores."""

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        l2_regularization=0.01,
        random_state=random_state,
    )


def clipped_positive_class_proba(model, x: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Predict positive-class probability and clip away from exact 0/1."""

    return np.clip(model.predict_proba(x)[:, 1], EPS, 1.0 - EPS)


def fit_oof_score_model(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    x_test: pd.DataFrame,
    *,
    random_state: int = 42,
    n_splits: int = 5,
) -> tuple[HistGradientBoostingClassifier, np.ndarray, np.ndarray, np.ndarray]:
    """Fit cross-fitted scores for calibration and a final model for OOT scoring.

    The historical period remains one train/calibration period.  Calibration
    scores are out-of-fold: each borrower is scored by a boosting model that
    did not train on that borrower's row.  A final boosting model is then fitted
    on the full historical period and applied to OOT rows.
    """

    y_arr = np.asarray(y_train, dtype=int)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if np.bincount(y_arr).min() < n_splits:
        raise ValueError("Each class must have at least n_splits observations")

    oof_scores = np.empty(len(y_arr), dtype=float)
    fold_ids = np.empty(len(y_arr), dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_model = make_boosting_score_model(random_state=random_state)

    for fold_id, (fit_idx, score_idx) in enumerate(splitter.split(x_train, y_arr), start=1):
        fold_model = clone(base_model)
        fold_model.set_params(random_state=random_state + fold_id)
        fold_model.fit(x_train.iloc[fit_idx], y_arr[fit_idx])
        oof_scores[score_idx] = clipped_positive_class_proba(fold_model, x_train.iloc[score_idx])
        fold_ids[score_idx] = fold_id

    final_model = make_boosting_score_model(random_state=random_state)
    final_model.fit(x_train, y_arr)
    test_scores = clipped_positive_class_proba(final_model, x_test)
    return final_model, oof_scores, test_scores, fold_ids
