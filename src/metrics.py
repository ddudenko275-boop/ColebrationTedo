"""Metrics for PD calibration, discrimination and stability."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


EPS = 1e-7


def _as_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values)


def _clip_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(_as_array(y_true), _clip_prob(y_prob)))


def log_loss_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(log_loss(_as_array(y_true), _clip_prob(y_prob)))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Expected calibration error with uniform or quantile bins."""

    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    y_true = _as_array(y_true)
    y_prob = _clip_prob(y_prob)
    n = len(y_true)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            return 0.0
        edges[0] = 0.0
        edges[-1] = 1.0

    ece = 0.0
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Hosmer-Lemeshow calibration test using quantile bins."""

    y_true = _as_array(y_true)
    y_prob = _clip_prob(y_prob)
    quantiles = np.unique(np.percentile(y_prob, np.linspace(0, 100, n_bins + 1)))
    chi2_stat = 0.0
    used_bins = 0

    for i in range(len(quantiles) - 1):
        if i == len(quantiles) - 2:
            mask = (y_prob >= quantiles[i]) & (y_prob <= quantiles[i + 1])
        else:
            mask = (y_prob >= quantiles[i]) & (y_prob < quantiles[i + 1])
        if not mask.any():
            continue
        n_i = int(mask.sum())
        observed = float(y_true[mask].sum())
        expected = float(y_prob[mask].sum())
        if expected > 0 and n_i - expected > 0:
            chi2_stat += (observed - expected) ** 2 / (expected * (1.0 - expected / n_i))
            used_bins += 1

    df = max(used_bins - 2, 1)
    p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=df)
    return {
        "chi2": round(float(chi2_stat), 4),
        "p_value": round(float(p_value), 4),
        "df": int(df),
    }


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Estimate calibration slope and intercept with unregularized logistic regression."""

    y_true = _as_array(y_true)
    p = _clip_prob(y_prob)
    logit_prob = np.log(p / (1.0 - p))
    model = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e12)
    model.fit(logit_prob.reshape(-1, 1), y_true)
    return {
        "slope": round(float(model.coef_[0][0]), 4),
        "intercept": round(float(model.intercept_[0]), 4),
    }


def get_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true,
        _clip_prob(y_prob),
        n_bins=n_bins,
        strategy=strategy,
    )
    return mean_predicted, fraction_of_positives


def summary_metrics(y_true: np.ndarray, y_prob: np.ndarray, name: str = "") -> dict:
    hl = hosmer_lemeshow_test(y_true, y_prob)
    si = calibration_slope_intercept(y_true, y_prob)
    return {
        "method": name,
        "brier_score": round(brier_score(y_true, y_prob), 5),
        "log_loss": round(log_loss_score(y_true, y_prob), 5),
        "ece_uniform": round(expected_calibration_error(y_true, y_prob, strategy="uniform"), 5),
        "ece_quantile": round(expected_calibration_error(y_true, y_prob, strategy="quantile"), 5),
        "hl_chi2": hl["chi2"],
        "hl_p_value": hl["p_value"],
        "cal_slope": si["slope"],
        "cal_intercept": si["intercept"],
    }


def discrimination_metrics(y_true: np.ndarray, y_prob: np.ndarray, name: str = "") -> dict:
    auc = float(roc_auc_score(y_true, y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "method": name,
        "auc_roc": round(auc, 4),
        "gini": round(2.0 * auc - 1.0, 4),
        "ks_stat": round(float(np.max(np.abs(tpr - fpr))), 4),
    }


def fixed_bin_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    tmp = pd.DataFrame({"y": _as_array(y_true), "pd": _clip_prob(y_prob)})
    tmp["bin"] = pd.cut(tmp["pd"], bins=edges, include_lowest=True, right=True)
    out = (
        tmp.groupby("bin", observed=False)
        .agg(
            n=("y", "size"),
            share=("y", lambda s: len(s) / len(tmp)),
            avg_pd=("pd", "mean"),
            default_rate=("y", "mean"),
            defaults=("y", "sum"),
        )
        .reset_index()
    )
    return out.fillna({"avg_pd": 0.0, "default_rate": 0.0, "defaults": 0})


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> dict:
    """Population Stability Index based on expected-sample quantile bins."""

    expected = _as_array(expected)
    actual = _as_array(actual)
    breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, n_bins + 1)))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    rows = []
    value = 0.0
    for i in range(len(breakpoints) - 1):
        exp_pct = np.mean((expected >= breakpoints[i]) & (expected < breakpoints[i + 1]))
        act_pct = np.mean((actual >= breakpoints[i]) & (actual < breakpoints[i + 1]))
        exp_pct = max(float(exp_pct), EPS)
        act_pct = max(float(act_pct), EPS)
        bucket = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        value += bucket
        rows.append(
            {
                "bin": i + 1,
                "expected_pct": exp_pct,
                "actual_pct": act_pct,
                "psi_bin": bucket,
            }
        )

    return {
        "psi_value": round(float(value), 5),
        "bin_details": pd.DataFrame(rows),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iter: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:
    rng = np.random.default_rng(random_state)
    y_true = _as_array(y_true)
    y_prob = _clip_prob(y_prob)
    scores = []

    for _ in range(n_iter):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if len(np.unique(y_b)) < 2:
            continue
        scores.append(float(metric_fn(y_b, p_b)))

    scores_arr = np.asarray(scores)
    alpha = (1.0 - ci) / 2.0
    return {
        "point_estimate": round(float(metric_fn(y_true, y_prob)), 5),
        "ci_lower": round(float(np.percentile(scores_arr, alpha * 100)), 5),
        "ci_upper": round(float(np.percentile(scores_arr, (1.0 - alpha) * 100)), 5),
        "std": round(float(scores_arr.std()), 5),
        "n_iter": int(len(scores_arr)),
    }
