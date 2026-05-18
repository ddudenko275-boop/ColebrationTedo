"""Probability calibration methods for PD models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


EPS = 1e-6


def _as_1d(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _clip_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(_as_1d(values), EPS, 1.0 - EPS)


def _safe_logit(values: np.ndarray) -> np.ndarray:
    return logit(_clip_prob(values))


def _interp_with_boundaries(interp, x: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
    """
    PCHIP behaves well inside the observed calibration range.
    Outside that range, boundary values are held constant.
    """
    x = _as_1d(x)
    y = interp(x)

    y = np.where(x < x_min, y_min, y)
    y = np.where(x > x_max, y_max, y)

    return y


class LogitCalibrator:
    """
    Platt-style logistic calibration.

    The model learns the transformation:
    raw RF-score -> logit(raw RF-score) -> calibrated PD
    """

    def __init__(self, C: float = 1e6):
        self._model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=C
        )

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        return _safe_logit(scores).reshape(-1, 1)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "LogitCalibrator":
        y = np.asarray(y, dtype=int)
        self._model.fit(self._transform(scores), y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return _clip_prob(self._model.predict_proba(self._transform(scores))[:, 1])


class IsotonicCalibrator:
    """
    Non-parametric monotone calibration.

    It captures nonlinearities well, but can produce a stepwise function.
    """

    def __init__(self):
        self._model = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip"
        )

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._model.fit(_as_1d(scores), np.asarray(y, dtype=float))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return _clip_prob(self._model.predict(_as_1d(scores)))


class BetaCalibrator:
    """
    Beta calibration.

    A more flexible parametric calibration:
    PD = sigmoid(a * log(s) + b * log(1 - s) + c)
    """

    def __init__(self, l2: float = 1e-4):
        self.l2 = l2
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

            z = a * log_s + b * log_1_minus_s + c
            p = _clip_prob(expit(z))

            nll = -np.sum(
                y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
            )

            penalty = self.l2 * (a ** 2 + b ** 2 + c ** 2)

            return float(nll + penalty)

        result = minimize(
            neg_log_likelihood,
            x0=np.array([1.0, -1.0, 0.0]),
            method="L-BFGS-B",
            bounds=[(-20, 20), (-20, 20), (-20, 20)]
        )

        if not result.success:
            raise RuntimeError(f"Beta calibration did not converge: {result.message}")

        self.a_, self.b_, self.c_ = [float(v) for v in result.x]
        self.success_ = True

        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.a_ is None or self.b_ is None or self.c_ is None:
            raise RuntimeError("BetaCalibrator must be fitted before predict().")

        s = _clip_prob(scores)

        z = self.a_ * np.log(s) + self.b_ * np.log(1.0 - s) + self.c_

        return _clip_prob(expit(z))


def _bin_stats(
    scores: np.ndarray,
    y: np.ndarray,
    n_bins: int = 30,
    alpha: float = 20.0
) -> pd.DataFrame:
    """
    Build statistics by quantile bins.

    alpha controls default-rate smoothing and protects small bins from overly sharp values.
    """

    scores = _as_1d(scores)
    y = np.asarray(y, dtype=float)

    tmp = pd.DataFrame({
        "score": scores,
        "y": y
    })

    n_unique = tmp["score"].nunique()

    if n_unique < 2:
        raise ValueError("At least two unique score values are required.")

    q = min(n_bins, n_unique)

    tmp["bin"] = pd.qcut(
        tmp["score"],
        q=q,
        duplicates="drop"
    )

    base_rate = float(tmp["y"].mean())

    stat = (
        tmp
        .groupby("bin", observed=True)
        .agg(
            score_mean=("score", "mean"),
            defaults=("y", "sum"),
            n=("y", "size")
        )
        .reset_index(drop=True)
        .sort_values("score_mean")
    )

    stat["default_rate_raw"] = stat["defaults"] / stat["n"]

    stat["default_rate_smooth"] = (
        stat["defaults"] + alpha * base_rate
    ) / (
        stat["n"] + alpha
    )

    return stat


class MonotoneSplineCalibrator:
    """
    Monotone spline.

    Pipeline:
    RF score -> bins -> smoothed default rate -> isotonic -> PCHIP.
    """

    def __init__(self, n_bins: int = 30, alpha: float = 20.0):
        self.n_bins = n_bins
        self.alpha = alpha

        self._interp: PchipInterpolator | None = None
        self.bin_stats_: pd.DataFrame | None = None

        self.x_min_: float | None = None
        self.x_max_: float | None = None
        self.y_min_: float | None = None
        self.y_max_: float | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "MonotoneSplineCalibrator":
        stat = _bin_stats(
            scores=scores,
            y=y,
            n_bins=self.n_bins,
            alpha=self.alpha
        )

        x = stat["score_mean"].to_numpy()
        r = stat["default_rate_smooth"].to_numpy()
        w = stat["n"].to_numpy()

        iso = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip"
        )

        r_iso = iso.fit_transform(x, r, sample_weight=w)

        x_u, idx = np.unique(x, return_index=True)
        y_u = r_iso[idx]

        if len(x_u) < 2:
            raise ValueError("At least two unique score bins are required.")

        self._interp = PchipInterpolator(
            x_u,
            y_u,
            extrapolate=False
        )

        self.x_min_ = float(x_u[0])
        self.x_max_ = float(x_u[-1])
        self.y_min_ = float(y_u[0])
        self.y_max_ = float(y_u[-1])

        self.bin_stats_ = stat.assign(default_rate_iso=r_iso)

        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._interp is None:
            raise RuntimeError("MonotoneSplineCalibrator must be fitted before predict().")

        y = _interp_with_boundaries(
            interp=self._interp,
            x=_as_1d(scores),
            x_min=self.x_min_,
            x_max=self.x_max_,
            y_min=self.y_min_,
            y_max=self.y_max_
        )

        return _clip_prob(y)


class FrenchSplineCalibrator:
    """
    Two-stage calibration:

    1. LogitCalibrator sets a stable overall PD level.
    2. A spline adjusts the residual error in logit space.

    Unlike the plain monotone spline, this version works as a correction
    to logit calibration rather than repeating the same object.
    """

    def __init__(
        self,
        n_bins: int = 30,
        alpha: float = 20.0,
        shrinkage: float = 0.6
    ):
        self.n_bins = n_bins
        self.alpha = alpha
        self.shrinkage = shrinkage

        self.logit_stage = LogitCalibrator()

        self._interp: PchipInterpolator | None = None
        self.bin_stats_: pd.DataFrame | None = None

        self.x_min_: float | None = None
        self.x_max_: float | None = None
        self.y_min_: float | None = None
        self.y_max_: float | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "FrenchSplineCalibrator":
        y = np.asarray(y, dtype=float)

        # Step 1: base logit calibration.
        self.logit_stage.fit(scores, y)
        p_logit = self.logit_stage.predict(scores)

        # Step 2: build bins using the logit-calibrated PD.
        stat = _bin_stats(
            scores=p_logit,
            y=y,
            n_bins=self.n_bins,
            alpha=self.alpha
        )

        p_bin = _clip_prob(stat["score_mean"].to_numpy())
        r = _clip_prob(stat["default_rate_smooth"].to_numpy())
        w = stat["n"].to_numpy()

        # First make the empirical default rate monotone.
        iso = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip"
        )

        r_iso = _clip_prob(iso.fit_transform(p_bin, r, sample_weight=w))

        # Key distinction:
        # the plain spline learns p -> empirical default rate,
        # while the French spline learns a correction in logit space.
        x = _safe_logit(p_bin)
        z_logit = _safe_logit(p_bin)
        z_empirical = _safe_logit(r_iso)

        # shrinkage prevents the spline from fully overriding the stable logit calibration.
        y_target = (1.0 - self.shrinkage) * z_logit + self.shrinkage * z_empirical

        x_u, idx = np.unique(x, return_index=True)
        y_u = y_target[idx]

        if len(x_u) < 2:
            raise ValueError("At least two unique bins are required.")

        self._interp = PchipInterpolator(
            x_u,
            y_u,
            extrapolate=False
        )

        self.x_min_ = float(x_u[0])
        self.x_max_ = float(x_u[-1])
        self.y_min_ = float(y_u[0])
        self.y_max_ = float(y_u[-1])

        self.bin_stats_ = stat.assign(
            default_rate_iso=r_iso,
            logit_target=y_target
        )

        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._interp is None:
            raise RuntimeError("FrenchSplineCalibrator must be fitted before predict().")

        p_logit = self.logit_stage.predict(scores)
        z = _safe_logit(p_logit)

        z_adj = _interp_with_boundaries(
            interp=self._interp,
            x=z,
            x_min=self.x_min_,
            x_max=self.x_max_,
            y_min=self.y_min_,
            y_max=self.y_max_
        )

        return _clip_prob(expit(z_adj))


def spline_smoothing_analysis(
    scores_calib: np.ndarray,
    y_calib: np.ndarray,
    scores_test: np.ndarray,
    y_test: np.ndarray,
    lam_grid: np.ndarray | None = None,
    n_bins: int = 30,
) -> pd.DataFrame:
    """Compare monotone spline smoothing values for legacy notebook runs."""

    if lam_grid is None:
        lam_grid = np.logspace(-2, 2, 9)

    y_calib = np.asarray(y_calib, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    rows = []
    for lam in lam_grid:
        cal = MonotoneSplineCalibrator(n_bins=n_bins, alpha=float(lam))
        cal.fit(scores_calib, y_calib)

        pred_calib = cal.predict(scores_calib)
        pred_test = cal.predict(scores_test)

        rows.append(
            {
                "lam": float(lam),
                "brier_calib": float(np.mean((y_calib - pred_calib) ** 2)),
                "brier_test": float(np.mean((y_test - pred_test) ** 2)),
            }
        )

    return pd.DataFrame(rows)


def get_all_calibrators() -> dict:
    """
    Calibration methods used in the comparison.
    """

    return {
        "Logit calibration": LogitCalibrator(),
        "Isotonic regression": IsotonicCalibrator(),
        "Beta calibration": BetaCalibrator(),
        "Monotone spline": MonotoneSplineCalibrator(),
        "French spline": FrenchSplineCalibrator(),
    }
