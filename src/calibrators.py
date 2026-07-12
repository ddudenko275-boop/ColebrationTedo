"""Probability calibration methods for PD models."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize
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


def _fit_logit_shift_to_mean(values: np.ndarray, target_mean: float) -> float:
    """Find a logit-scale intercept shift that makes mean PD equal target_mean."""

    target = float(np.clip(target_mean, EPS, 1.0 - EPS))
    base = _clip_prob(values)
    if np.isclose(float(base.mean()), target, atol=1e-12):
        return 0.0

    base_logit = _safe_logit(base)

    def objective(shift: float) -> float:
        return float(expit(base_logit + shift).mean() - target)

    return float(brentq(objective, -50.0, 50.0))


def _extend_monotone_support(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    y_is_probability: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend monotone spline knots to the observed score bounds.

    The added boundary point is projected linearly using the slope between
    the last two real knots. When y is a raw probability
    (y_is_probability=True, MonotoneSplineCalibrator's case), the projection
    is done in logit space and mapped back with expit: this keeps the curve
    rising past the last real knot -- unlike a flat clip, which can freeze a
    wide plateau when the last quantile bin is far from the true max score
    because the tail is sparse -- while naturally saturating toward 0/1
    instead of overshooting the way a linear projection directly in
    probability space can (see docs/spline_methodology.md, §3.3, for why both
    a flat clip and a plain probability-space linear extrapolation were
    unsatisfactory here).

    FrenchSplineCalibrator's knots are already on the logit scale
    (y_is_probability=False): the projection there is a plain linear one, and
    expit() is applied once downstream when the calibrator converts back to
    probability.
    """

    x = _as_1d(x)
    y = _as_1d(y)

    def project(x_edge: float, x_a: float, y_a: float, x_b: float, y_b: float) -> float:
        if y_is_probability:
            z_a, z_b = _safe_logit(np.array([y_a, y_b]))
        else:
            z_a, z_b = y_a, y_b
        slope = (z_b - z_a) / (x_b - x_a)
        z_edge = z_b + slope * (x_edge - x_b)
        return float(expit(z_edge)) if y_is_probability else float(z_edge)

    if x_min < x[0]:
        y_left = project(x_min, x[1], y[1], x[0], y[0])
        x = np.r_[float(x_min), x]
        y = np.r_[y_left, y]

    if x_max > x[-1]:
        y_right = project(x_max, x[-2], y[-2], x[-1], y[-1])
        x = np.r_[x, float(x_max)]
        y = np.r_[y, y_right]

    return x, y


def _spline_knot_diagnostics(
    stat: pd.DataFrame,
    target_col: str,
    score_col: str = "score_mean",
) -> pd.DataFrame:
    """Return knot-level slope diagnostics for fitted bin-based splines."""

    if target_col not in stat.columns:
        raise ValueError(f"Column {target_col!r} is not available in bin_stats_.")

    out = stat.copy().reset_index(drop=True)
    x = out[score_col].to_numpy(dtype=float)
    y = out[target_col].to_numpy(dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)
    slope = np.r_[np.nan, np.divide(dy, dx, out=np.full_like(dy, np.nan), where=dx != 0.0)]
    slope_change = np.r_[np.nan, np.diff(slope)]

    out["spline_target"] = y
    out["local_slope"] = slope
    out["local_slope_change"] = slope_change
    out["abs_local_slope_change"] = np.abs(slope_change)

    finite_change = out["abs_local_slope_change"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_change) > 0:
        threshold = float(finite_change.quantile(0.75))
        out["candidate_switch_knot"] = out["abs_local_slope_change"] >= threshold
    else:
        out["candidate_switch_knot"] = False

    return out


def _interp_with_boundaries(interp, x: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
    """
    PCHIP аккуратно работает внутри диапазона.
    За пределами calibration-диапазона фиксируем крайние значения.
    """
    x = _as_1d(x)
    y = interp(x)

    y = np.where(x < x_min, y_min, y)
    y = np.where(x > x_max, y_max, y)

    return y


class LogitCalibrator:
    """
    Platt-style logistic calibration.

    Модель строит преобразование:
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
    Непараметрическая монотонная калибровка.

    Хорошо ловит нелинейности, но может давать ступенчатую функцию.
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

    Более гибкая параметрическая калибровка:
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
) -> pd.DataFrame:
    """
    Строим статистику по квантильным бинам.

    default_rate_raw идет в isotonic regression напрямую, взвешенно по n —
    сглаживание делает сама isotonic regression (PAVA), без дополнительного
    сдвига к базовой ставке портфеля перед ней. Двойное сглаживание (сначала
    shrinkage к base_rate, потом isotonic поверх) может схлопывать соседние
    бины в плоский участок, который не отражает реальной формы данных
    (см. docs/spline_methodology.md, §3.2).
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

    return stat


class MonotoneSplineCalibrator:
    """
    Монотонный сплайн.

    Логика:
    RF-score -> бины -> isotonic (взвешенная по n) -> PCHIP.

    This is a calibration spline with quantile-bin knots, not a regression
    spline that optimizes structural breakpoints. Smoothing is done once, by
    isotonic regression itself (weighted by bin size); there is no separate
    shrinkage-to-base-rate step before it (see docs/spline_methodology.md,
    §3.2, for why stacking two smoothing steps produced artificial flat
    segments). After the spline shape is fitted, a logit-scale intercept
    shift preserves the fit-sample central tendency exactly.

    n_bins default (75) comes from scripts/spline_parameter_search.py's data
    floor -- the largest bin count that still keeps ~20 expected defaults per
    bin on this portfolio -- not from a target economic outcome.
    """

    def __init__(self, n_bins: int = 75):
        self.n_bins = n_bins

        self._interp: PchipInterpolator | None = None
        self.bin_stats_: pd.DataFrame | None = None

        self.x_min_: float | None = None
        self.x_max_: float | None = None
        self.y_min_: float | None = None
        self.y_max_: float | None = None
        self.target_pd_: float | None = None
        self.ct_shift_: float = 0.0
        self.x_knots_: np.ndarray | None = None
        self.y_knots_: np.ndarray | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "MonotoneSplineCalibrator":
        y = np.asarray(y, dtype=float)
        stat = _bin_stats(
            scores=scores,
            y=y,
            n_bins=self.n_bins,
        )

        x = stat["score_mean"].to_numpy()
        r = stat["default_rate_raw"].to_numpy()
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

        score_values = _as_1d(scores)
        x_u, y_u = _extend_monotone_support(
            x_u,
            y_u,
            float(score_values.min()),
            float(score_values.max()),
        )

        self._interp = PchipInterpolator(
            x_u,
            y_u,
            extrapolate=False
        )

        self.x_min_ = float(x_u[0])
        self.x_max_ = float(x_u[-1])
        self.y_min_ = float(y_u[0])
        self.y_max_ = float(y_u[-1])
        self.x_knots_ = x_u.copy()
        self.y_knots_ = y_u.copy()

        self.bin_stats_ = stat.assign(default_rate_iso=r_iso)
        self.target_pd_ = float(y.mean())
        self.ct_shift_ = _fit_logit_shift_to_mean(
            self._predict_without_ct_shift(scores),
            self.target_pd_,
        )

        return self

    def _predict_without_ct_shift(self, scores: np.ndarray) -> np.ndarray:
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

    def predict(self, scores: np.ndarray) -> np.ndarray:
        y = self._predict_without_ct_shift(scores)
        return _clip_prob(expit(_safe_logit(y) + self.ct_shift_))

    def knot_diagnostics(self) -> pd.DataFrame:
        """
        Return bin/knot diagnostics with a first-derivative proxy.

        candidate_switch_knot marks the largest local slope changes. It is a
        diagnostic for review, not an automatic breakpoint test. Using the
        spline's first derivative to locate regression switching points
        follows Ilyasov (2018), DOI 10.18721/JE.11412 (see
        docs/spline_methodology.md, §3.7).
        """

        if self.bin_stats_ is None:
            raise RuntimeError("MonotoneSplineCalibrator must be fitted before diagnostics.")
        return _spline_knot_diagnostics(self.bin_stats_, target_col="default_rate_iso")


class FrenchSplineCalibrator:
    """
    Двухэтапная калибровка (logit + spline "polish", по методике,
    комбинирующей логит-калибровку со сплайн-сглаживанием поверх):

    1. LogitCalibrator задает стабильный общий уровень PD.
    2. Сплайн корректирует остаточную ошибку в logit-пространстве.

    В отличие от простой версии, здесь сплайн не просто повторяет обычный
    MonotoneSplineCalibrator, а работает как поправка к логит-калибровке.

    As with MonotoneSplineCalibrator, knots are derived from quantile bins and
    smoothed once by isotonic regression (weighted by bin size), without a
    separate pre-isotonic shrinkage step. The final logit-scale intercept
    shift is part of the fitted calibrator and preserves the fit-sample
    central tendency exactly.

    n_bins default (75) is the same data-floor pick as MonotoneSplineCalibrator.
    shrinkage default (0.9) minimizes OOT Brier score on this portfolio in
    scripts/spline_parameter_search.py -- note the fit is quite flat for
    shrinkage above ~0.6, so this choice is not highly sensitive.
    """

    def __init__(
        self,
        n_bins: int = 75,
        shrinkage: float = 0.9
    ):
        self.n_bins = n_bins
        self.shrinkage = shrinkage

        self.logit_stage = LogitCalibrator()

        self._interp: PchipInterpolator | None = None
        self.bin_stats_: pd.DataFrame | None = None

        self.x_min_: float | None = None
        self.x_max_: float | None = None
        self.y_min_: float | None = None
        self.y_max_: float | None = None
        self.target_pd_: float | None = None
        self.ct_shift_: float = 0.0
        self.x_knots_: np.ndarray | None = None
        self.y_knots_: np.ndarray | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "FrenchSplineCalibrator":
        y = np.asarray(y, dtype=float)

        # Шаг 1: базовая логит-калибровка.
        self.logit_stage.fit(scores, y)
        p_logit = self.logit_stage.predict(scores)

        # Шаг 2: строим бины уже по логит-калиброванной PD.
        stat = _bin_stats(
            scores=p_logit,
            y=y,
            n_bins=self.n_bins,
        )

        p_bin = _clip_prob(stat["score_mean"].to_numpy())
        r = _clip_prob(stat["default_rate_raw"].to_numpy())
        w = stat["n"].to_numpy()

        # Сначала делаем монотонную эмпирическую default rate.
        iso = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip"
        )

        r_iso = _clip_prob(iso.fit_transform(p_bin, r, sample_weight=w))

        # Важное отличие:
        # обычный сплайн строит p -> empirical default rate.
        # французский сплайн строит поправку в logit-пространстве.
        x = _safe_logit(p_bin)
        z_logit = _safe_logit(p_bin)
        z_empirical = _safe_logit(r_iso)

        # shrinkage не дает сплайну полностью "сломать" стабильную логит-калибровку.
        y_target = (1.0 - self.shrinkage) * z_logit + self.shrinkage * z_empirical

        x_u, idx = np.unique(x, return_index=True)
        y_u = y_target[idx]

        if len(x_u) < 2:
            raise ValueError("At least two unique bins are required.")

        z_values = _safe_logit(p_logit)
        x_u, y_u = _extend_monotone_support(
            x_u,
            y_u,
            float(z_values.min()),
            float(z_values.max()),
            y_is_probability=False,
        )

        self._interp = PchipInterpolator(
            x_u,
            y_u,
            extrapolate=False
        )

        self.x_min_ = float(x_u[0])
        self.x_max_ = float(x_u[-1])
        self.y_min_ = float(y_u[0])
        self.y_max_ = float(y_u[-1])
        self.x_knots_ = x_u.copy()
        self.y_knots_ = y_u.copy()

        self.bin_stats_ = stat.assign(
            default_rate_iso=r_iso,
            logit_target=y_target
        )
        self.target_pd_ = float(y.mean())
        self.ct_shift_ = _fit_logit_shift_to_mean(
            self._predict_without_ct_shift(scores),
            self.target_pd_,
        )

        return self

    def _predict_without_ct_shift(self, scores: np.ndarray) -> np.ndarray:
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

    def predict(self, scores: np.ndarray) -> np.ndarray:
        y = self._predict_without_ct_shift(scores)
        return _clip_prob(expit(_safe_logit(y) + self.ct_shift_))

    def knot_diagnostics(self) -> pd.DataFrame:
        """
        Return bin/knot diagnostics with a first-derivative proxy.

        For the French spline the target is shown on the PD scale even though
        the fitted spline correction is built in logit space.
        """

        if self.bin_stats_ is None:
            raise RuntimeError("FrenchSplineCalibrator must be fitted before diagnostics.")
        stat = self.bin_stats_.copy()
        stat["logit_target_pd"] = _clip_prob(expit(stat["logit_target"].to_numpy(dtype=float)))
        return _spline_knot_diagnostics(stat, target_col="logit_target_pd")


def spline_smoothing_analysis(
    scores_calib: np.ndarray,
    y_calib: np.ndarray,
    scores_test: np.ndarray,
    y_test: np.ndarray,
    n_bins_grid: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Compare monotone spline smoothing (by n_bins) for legacy notebook runs.

    n_bins is now the only smoothing knob MonotoneSplineCalibrator exposes:
    isotonic regression (weighted by bin size) does the smoothing directly,
    there is no separate alpha shrinkage parameter to sweep anymore (see
    docs/spline_methodology.md, §3.2 and §4).
    """

    if n_bins_grid is None:
        n_bins_grid = (10, 15, 20, 25, 30, 40, 50, 75, 100)

    y_calib = np.asarray(y_calib, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    rows = []
    for n_bins in n_bins_grid:
        cal = MonotoneSplineCalibrator(n_bins=int(n_bins))
        cal.fit(scores_calib, y_calib)

        pred_calib = cal.predict(scores_calib)
        pred_test = cal.predict(scores_test)

        rows.append(
            {
                "n_bins": int(n_bins),
                "brier_calib": float(np.mean((y_calib - pred_calib) ** 2)),
                "brier_test": float(np.mean((y_test - pred_test) ** 2)),
            }
        )

    return pd.DataFrame(rows)


def get_all_calibrators() -> dict:
    """
    Набор калибраторов для сравнения.
    """

    return {
        "Логит-калибровка": LogitCalibrator(),
        "Изотоническая регрессия": IsotonicCalibrator(),
        "Бета-калибровка": BetaCalibrator(),
        "Монотонный сплайн": MonotoneSplineCalibrator(),
        "Французский сплайн": FrenchSplineCalibrator(),
    }
