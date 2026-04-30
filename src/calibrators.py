"""Probability calibration methods for PD models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


EPS = 1e-6


def _as_1d(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _clip_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(_as_1d(values), EPS, 1.0 - EPS)


class LogitCalibrator:
    """Platt-style logistic calibration on logit-transformed probability scores."""

    def __init__(self):
        self._model = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e12)

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        s = _clip_prob(scores)
        return np.log(s / (1.0 - s)).reshape(-1, 1)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "LogitCalibrator":
        self._model.fit(self._transform(scores), np.asarray(y))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return _clip_prob(self._model.predict_proba(self._transform(scores))[:, 1])


class IsotonicCalibrator:
    """Non-parametric monotone calibration with piecewise-constant output."""

    def __init__(self):
        self._model = IsotonicRegression(increasing=True, out_of_bounds="clip")

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._model.fit(_as_1d(scores), np.asarray(y))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return _clip_prob(self._model.predict(_as_1d(scores)))


class BetaCalibrator:
    """Beta calibration following the Kull-Silva Filho-Flach probability map."""

    def __init__(self):
        self.a_: float | None = None
        self.b_: float | None = None
        self.c_: float | None = None
        self.success_: bool | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "BetaCalibrator":
        s = _clip_prob(scores)
        y = np.asarray(y, dtype=float)
        log_s = np.log(s)
        log_1_minus_s = np.log(1.0 - s)

        def neg_log_likelihood(params: np.ndarray) -> float:
            a, b, c = params
            p = _clip_prob(expit(a * log_s + b * log_1_minus_s + c))
            return -float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        result = minimize(
            neg_log_likelihood,
            x0=np.array([1.0, -1.0, 0.0]),
            method="L-BFGS-B",
        )
        self.a_, self.b_, self.c_ = [float(v) for v in result.x]
        self.success_ = bool(result.success)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.a_ is None or self.b_ is None or self.c_ is None:
            raise RuntimeError("BetaCalibrator must be fitted before predict().")
        s = _clip_prob(scores)
        return _clip_prob(expit(self.a_ * np.log(s) + self.b_ * np.log(1.0 - s) + self.c_))


def _bin_stats(scores: np.ndarray, y: np.ndarray, n_bins: int = 30) -> pd.DataFrame:
    tmp = pd.DataFrame({"score": _as_1d(scores), "y": np.asarray(y, dtype=float)})
    q = min(n_bins, tmp["score"].nunique())
    tmp["bin"] = pd.qcut(tmp["score"], q=q, duplicates="drop")
    return (
        tmp.groupby("bin", observed=True)
        .agg(score_mean=("score", "mean"), default_rate=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
        .sort_values("score_mean")
    )


class MonotoneSplineCalibrator:
    """Smoothed monotone calibration curve built from binned default rates.

    The method first smooths binned default rates with isotonic regression and
    then interpolates the monotone curve with PCHIP. It is deliberately simple:
    the goal is a governance-friendly smooth PD curve, not maximum flexibility.
    """

    def __init__(self, n_bins: int = 30):
        self.n_bins = n_bins
        self._interp: PchipInterpolator | None = None
        self.bin_stats_: pd.DataFrame | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "MonotoneSplineCalibrator":
        stat = _bin_stats(scores, y, n_bins=self.n_bins)
        x = stat["score_mean"].to_numpy()
        r = stat["default_rate"].to_numpy()
        w = stat["n"].to_numpy()

        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        r_iso = iso.fit_transform(x, r, sample_weight=w)
        x_u, idx = np.unique(x, return_index=True)
        y_u = r_iso[idx]

        if len(x_u) < 2:
            raise ValueError("At least two unique score bins are required for spline calibration.")

        self._interp = PchipInterpolator(x_u, y_u, extrapolate=True)
        self.bin_stats_ = stat.assign(default_rate_iso=r_iso)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._interp is None:
            raise RuntimeError("MonotoneSplineCalibrator must be fitted before predict().")
        return _clip_prob(self._interp(_as_1d(scores)))


class FrenchSplineCalibrator:
    """Two-stage logit plus monotone spline PD calibration.

    This is an experimental, ICAS-style inspired calibration recipe: a stable
    parametric logit step is followed by a smooth monotone correction curve.
    """

    def __init__(self, n_bins: int = 30):
        self.logit_stage = LogitCalibrator()
        self.spline_stage = MonotoneSplineCalibrator(n_bins=n_bins)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "FrenchSplineCalibrator":
        y = np.asarray(y)
        self.logit_stage.fit(scores, y)
        stage1 = self.logit_stage.predict(scores)
        self.spline_stage.fit(stage1, y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        stage1 = self.logit_stage.predict(scores)
        return self.spline_stage.predict(stage1)


def get_all_calibrators() -> dict:
    """Return the default set of PD calibrators used in the project."""

    return {
        "Логит-калибровка": LogitCalibrator(),
        "Изотоническая регрессия": IsotonicCalibrator(),
        "Бета-калибровка": BetaCalibrator(),
        "Монотонный сплайн": MonotoneSplineCalibrator(),
        "Французский сплайн": FrenchSplineCalibrator(),
    }
