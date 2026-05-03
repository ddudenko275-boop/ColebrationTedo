"""Historical portfolio reconstruction and rating-level PD diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from src.capital import IRBAssumptions, calculate_irb_capital

DEFAULT_RATING_ORDER = ("A", "B", "C", "D")
MASTER_SCALE_RATINGS = (
    "A1",
    "A2",
    "A3",
    "B1",
    "B2",
    "B3",
    "C1",
    "C2",
    "C3",
    "D1",
    "D2",
    "D3",
    "E",
)
DEFAULT_ASSET_EAD = 1_000_000.0
EPS = 1e-6


def _as_1d(values: np.ndarray | pd.Series, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1:
        out = out.reshape(-1)
    if len(out) == 0:
        raise ValueError(f"{name} must not be empty")
    return out


def _resolve_ead(df: pd.DataFrame, ead_col: str | None) -> np.ndarray:
    if ead_col is None:
        return np.ones(len(df), dtype=float)
    if ead_col not in df.columns:
        raise KeyError(f"'{ead_col}' is not present in df")
    ead = _as_1d(df[ead_col], ead_col)
    if np.any(ead < 0.0):
        raise ValueError("ead values must be non-negative")
    return ead


def _resolve_rating_ead(
    df: pd.DataFrame,
    ead_col: str | None,
    default_asset_ead: float,
) -> np.ndarray:
    if ead_col is not None:
        return _resolve_ead(df, ead_col)
    if default_asset_ead < 0.0:
        raise ValueError("default_asset_ead must be non-negative")
    return np.full(len(df), float(default_asset_ead), dtype=float)


def _clip_prob(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(_as_1d(values, "pd_values"), EPS, 1.0 - EPS)


def _validate_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")


def portfolio_average_pd(
    df: pd.DataFrame,
    pd_col: str,
    ead_col: str | None = None,
) -> float:
    """Return one aggregate PD for the whole portfolio.

    This is useful as a top-line summary only. Rating-level diagnostics should
    use :func:`historical_portfolio_panel` to preserve the risk distribution.
    """

    _validate_columns(df, (pd_col,))
    pd_values = _as_1d(df[pd_col], pd_col)
    ead = _resolve_ead(df, ead_col)
    total_ead = ead.sum()
    if total_ead <= 0.0:
        raise ValueError("total ead must be positive")
    return float(np.average(pd_values, weights=ead))


def historical_portfolio_panel(
    df: pd.DataFrame,
    pd_col: str,
    period_col: str = "origination_year",
    rating_col: str = "rating",
    default_col: str = "default",
    ead_col: str | None = None,
    rating_order: tuple[str, ...] | None = DEFAULT_RATING_ORDER,
) -> pd.DataFrame:
    """Recreate portfolio structure by period and rating with PD diagnostics."""

    _validate_columns(df, (period_col, rating_col, default_col, pd_col))
    work = df[[period_col, rating_col, default_col, pd_col]].copy()
    work["_ead"] = _resolve_ead(df, ead_col)
    work["_expected_defaults"] = work[pd_col].astype(float)
    work["_defaulted_ead"] = work[default_col].astype(float) * work["_ead"]
    work["_expected_default_ead"] = work[pd_col].astype(float) * work["_ead"]
    work["_weighted_pd"] = work[pd_col].astype(float) * work["_ead"]

    panel = (
        work.groupby([period_col, rating_col], observed=False)
        .agg(
            n_obs=(default_col, "size"),
            total_ead=("_ead", "sum"),
            defaults=(default_col, "sum"),
            defaulted_ead=("_defaulted_ead", "sum"),
            avg_pd=(pd_col, "mean"),
            ead_weighted_pd=("_weighted_pd", "sum"),
            expected_defaults=("_expected_defaults", "sum"),
            expected_default_ead=("_expected_default_ead", "sum"),
        )
        .reset_index()
    )
    panel["observed_default_rate"] = panel["defaults"] / panel["n_obs"]
    panel["observed_default_ead_rate"] = panel["defaulted_ead"] / panel["total_ead"]
    panel["ead_weighted_pd"] = panel["ead_weighted_pd"] / panel["total_ead"]
    panel["default_gap"] = panel["defaults"] - panel["expected_defaults"]
    panel["default_gap_rate"] = panel["observed_default_rate"] - panel["avg_pd"]
    panel["default_ead_gap"] = panel["defaulted_ead"] - panel["expected_default_ead"]
    panel["default_ead_gap_rate"] = (
        panel["observed_default_ead_rate"] - panel["ead_weighted_pd"]
    )
    panel["calibration_ratio"] = np.where(
        panel["expected_defaults"] > 0.0,
        panel["defaults"] / panel["expected_defaults"],
        np.nan,
    )

    total_by_period = panel.groupby(period_col)["n_obs"].transform("sum")
    ead_by_period = panel.groupby(period_col)["total_ead"].transform("sum")
    panel["portfolio_count_share"] = panel["n_obs"] / total_by_period
    panel["portfolio_ead_share"] = panel["total_ead"] / ead_by_period

    if rating_order is not None:
        order = {rating: i for i, rating in enumerate(rating_order)}
        panel["_rating_order"] = panel[rating_col].map(order).fillna(len(order))
        panel = panel.sort_values([period_col, "_rating_order", rating_col]).drop(
            columns="_rating_order"
        )
    else:
        panel = panel.sort_values([period_col, rating_col])

    return panel.reset_index(drop=True)


def compare_methods_by_historical_panel(
    df: pd.DataFrame,
    predictions: Mapping[str, np.ndarray | pd.Series],
    period_col: str = "origination_year",
    rating_col: str = "rating",
    default_col: str = "default",
    ead_col: str | None = None,
    rating_order: tuple[str, ...] | None = DEFAULT_RATING_ORDER,
) -> pd.DataFrame:
    """Compare calibrated PD methods on the same period-rating portfolio panel."""

    if not predictions:
        raise ValueError("predictions must contain at least one method")

    rows = []
    for method, values in predictions.items():
        pd_values = _as_1d(values, method)
        if len(pd_values) != len(df):
            raise ValueError(
                f"Prediction length for '{method}' must match df length: "
                f"{len(pd_values)} != {len(df)}"
            )

        work = df.copy()
        work["_method_pd"] = pd_values
        panel = historical_portfolio_panel(
            work,
            pd_col="_method_pd",
            period_col=period_col,
            rating_col=rating_col,
            default_col=default_col,
            ead_col=ead_col,
            rating_order=rating_order,
        )
        panel.insert(0, "method", method)
        rows.append(panel)

    return pd.concat(rows, ignore_index=True)


def method_portfolio_summary(
    df: pd.DataFrame,
    predictions: Mapping[str, np.ndarray | pd.Series],
    default_col: str = "default",
    ead_col: str | None = None,
) -> pd.DataFrame:
    """Return one-row portfolio summaries for each PD calibration method."""

    _validate_columns(df, (default_col,))
    ead = _resolve_ead(df, ead_col)
    total_ead = ead.sum()
    if total_ead <= 0.0:
        raise ValueError("total ead must be positive")

    rows = []
    for method, values in predictions.items():
        pd_values = _as_1d(values, method)
        if len(pd_values) != len(df):
            raise ValueError(
                f"Prediction length for '{method}' must match df length: "
                f"{len(pd_values)} != {len(df)}"
            )
        default_values = df[default_col].to_numpy(dtype=float)
        default_count = float(np.sum(default_values))
        expected_default_count = float(np.sum(pd_values))
        defaulted_ead = float(np.sum(default_values * ead))
        expected_default_ead = float(np.sum(pd_values * ead))
        rows.append(
            {
                "method": method,
                "n_obs": len(df),
                "total_ead": total_ead,
                "avg_pd": float(np.mean(pd_values)),
                "ead_weighted_pd": float(np.average(pd_values, weights=ead)),
                "defaults": default_count,
                "expected_defaults": expected_default_count,
                "default_gap": default_count - expected_default_count,
                "defaulted_ead": defaulted_ead,
                "expected_default_ead": expected_default_ead,
                "default_ead_gap": defaulted_ead - expected_default_ead,
                "calibration_ratio": (
                    default_count / expected_default_count
                    if expected_default_count > 0.0
                    else np.nan
                ),
                "ead_calibration_ratio": (
                    defaulted_ead / expected_default_ead
                    if expected_default_ead > 0.0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows).set_index("method")


def assign_master_scale_ratings(
    scores: np.ndarray | pd.Series,
    reference_scores: np.ndarray | pd.Series | None = None,
    ratings: tuple[str, ...] = MASTER_SCALE_RATINGS,
) -> pd.Categorical:
    """Assign ordered master-scale ratings from model scores.

    Lower scores receive the first, best rating. Breakpoints are estimated from
    the reference sample, usually the calibration period, and then applied to the
    target portfolio.
    """

    if len(ratings) < 2:
        raise ValueError("ratings must contain at least two levels")

    ref = _as_1d(scores if reference_scores is None else reference_scores, "reference_scores")
    values = _as_1d(scores, "scores")
    _, edges = pd.qcut(ref, q=len(ratings), retbins=True, duplicates="drop")
    if len(edges) - 1 != len(ratings):
        raise ValueError(
            "reference_scores do not contain enough unique values for the rating scale"
        )
    edges[0], edges[-1] = -np.inf, np.inf
    return pd.cut(values, bins=edges, labels=ratings, include_lowest=True, ordered=True)


def calibrate_pd_to_target(
    pd_values: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series,
    target_pd: float,
) -> np.ndarray:
    """Shift PDs on the logit scale so their weighted average equals target_pd."""

    base_pd = _clip_prob(pd_values)
    weights_arr = _as_1d(weights, "weights")
    if len(base_pd) != len(weights_arr):
        raise ValueError("pd_values and weights must have the same length")
    if np.any(weights_arr < 0.0):
        raise ValueError("weights must be non-negative")
    if weights_arr.sum() <= 0.0:
        raise ValueError("weights must have positive sum")
    if not EPS < target_pd < 1.0 - EPS:
        raise ValueError("target_pd must be between 0 and 1")

    logits = np.log(base_pd / (1.0 - base_pd))

    def objective(shift: float) -> float:
        return float(np.average(expit(logits + shift), weights=weights_arr) - target_pd)

    return expit(logits + brentq(objective, -50.0, 50.0))


def rating_master_scale(
    df: pd.DataFrame,
    pd_values: np.ndarray | pd.Series,
    score_values: np.ndarray | pd.Series,
    target_pd: float,
    rating_col: str = "master_rating",
    default_col: str = "default",
    ead_col: str | None = None,
    default_asset_ead: float = DEFAULT_ASSET_EAD,
    rating_order: tuple[str, ...] = MASTER_SCALE_RATINGS,
) -> pd.DataFrame:
    """Build a rating-level PD scale calibrated to a portfolio target PD."""

    _validate_columns(df, (rating_col, default_col))
    pd_arr = _clip_prob(pd_values)
    score_arr = _as_1d(score_values, "score_values")
    if len(pd_arr) != len(df) or len(score_arr) != len(df):
        raise ValueError("pd_values and score_values must match df length")

    work = df[[rating_col, default_col]].copy()
    work["_pd"] = pd_arr
    work["_score"] = score_arr
    work["_ead"] = _resolve_rating_ead(df, ead_col, default_asset_ead)
    work["_defaulted_ead"] = work[default_col].astype(float) * work["_ead"]
    work["_weighted_pd"] = work["_pd"] * work["_ead"]

    scale = (
        work.groupby(rating_col, observed=False)
        .agg(
            n_assets=(default_col, "size"),
            total_ead=("_ead", "sum"),
            avg_score=("_score", "mean"),
            defaults=(default_col, "sum"),
            defaulted_ead=("_defaulted_ead", "sum"),
            observed_default_rate=(default_col, "mean"),
            pd_before_target=("_pd", "mean"),
            ead_weighted_pd_before_target=("_weighted_pd", "sum"),
        )
        .reindex(rating_order)
        .reset_index()
        .rename(columns={rating_col: "rating"})
    )

    if scale["n_assets"].isna().any():
        missing = scale.loc[scale["n_assets"].isna(), "rating"].tolist()
        raise ValueError(f"Rating scale has empty ratings: {missing}")

    scale["n_assets"] = scale["n_assets"].astype(int)
    scale["ead_weighted_pd_before_target"] = (
        scale["ead_weighted_pd_before_target"] / scale["total_ead"]
    )
    scale["pd_rating"] = calibrate_pd_to_target(
        scale["ead_weighted_pd_before_target"],
        scale["total_ead"],
        target_pd,
    )
    scale["one_minus_pd"] = 1.0 - scale["pd_rating"]
    scale["expected_defaults"] = scale["pd_rating"] * scale["n_assets"]
    scale["expected_default_ead"] = scale["pd_rating"] * scale["total_ead"]
    scale["portfolio_count_share"] = scale["n_assets"] / scale["n_assets"].sum()
    scale["portfolio_ead_share"] = scale["total_ead"] / scale["total_ead"].sum()
    scale["default_gap"] = scale["defaults"] - scale["expected_defaults"]
    scale["default_ead_gap"] = scale["defaulted_ead"] - scale["expected_default_ead"]
    scale["calibration_ratio"] = np.where(
        scale["expected_defaults"] > 0.0,
        scale["defaults"] / scale["expected_defaults"],
        np.nan,
    )
    return scale


def compare_methods_by_rating_master_scale(
    df: pd.DataFrame,
    predictions: Mapping[str, np.ndarray | pd.Series],
    score_values: np.ndarray | pd.Series,
    target_pd: float,
    rating_col: str = "master_rating",
    default_col: str = "default",
    ead_col: str | None = None,
    default_asset_ead: float = DEFAULT_ASSET_EAD,
    rating_order: tuple[str, ...] = MASTER_SCALE_RATINGS,
) -> pd.DataFrame:
    """Build calibrated master scales for several PD methods."""

    if not predictions:
        raise ValueError("predictions must contain at least one method")

    rows = []
    for method, pd_values in predictions.items():
        scale = rating_master_scale(
            df,
            pd_values=pd_values,
            score_values=score_values,
            target_pd=target_pd,
            rating_col=rating_col,
            default_col=default_col,
            ead_col=ead_col,
            default_asset_ead=default_asset_ead,
            rating_order=rating_order,
        )
        scale.insert(0, "method", method)
        rows.append(scale)
    return pd.concat(rows, ignore_index=True)


def summarize_rating_scale(
    scale: pd.DataFrame,
    method_col: str | None = None,
) -> pd.DataFrame:
    """Summarise portfolio-level target fit for one or more rating scales."""

    required = (
        "n_assets",
        "total_ead",
        "defaults",
        "pd_rating",
        "expected_defaults",
        "expected_default_ead",
    )
    _validate_columns(scale, required)
    grouped = (
        scale.groupby(method_col, dropna=False)
        if method_col is not None
        else [(None, scale)]
    )

    rows = []
    for key, frame in grouped:
        total_ead = frame["total_ead"].sum()
        row = {
            "n_assets": int(frame["n_assets"].sum()),
            "total_ead": float(total_ead),
            "defaults": float(frame["defaults"].sum()),
            "target_weighted_pd": float(
                np.average(frame["pd_rating"], weights=frame["total_ead"])
            ),
            "expected_defaults": float(frame["expected_defaults"].sum()),
            "expected_default_ead": float(frame["expected_default_ead"].sum()),
            "pd_min": float(frame["pd_rating"].min()),
            "pd_max": float(frame["pd_rating"].max()),
        }
        if method_col is not None:
            row[method_col] = key
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.set_index(method_col) if method_col is not None else out


def rating_scale_capital(
    scale: pd.DataFrame,
    assumptions: IRBAssumptions | None = None,
    method_col: str | None = None,
) -> pd.DataFrame:
    """Calculate IRB capital/RWA from rating-level PD and total EAD buckets."""

    _validate_columns(scale, ("pd_rating", "total_ead"))
    grouped = (
        scale.groupby(method_col, dropna=False)
        if method_col is not None
        else [(None, scale)]
    )

    rows = []
    for key, frame in grouped:
        details = calculate_irb_capital(
            frame["pd_rating"].to_numpy(),
            assumptions=assumptions,
            ead_values=frame["total_ead"].to_numpy(),
        )
        row = {
            "total_ead": float(details["ead"].sum()),
            "total_expected_loss": float(details["expected_loss"].sum()),
            "total_unexpected_loss_capital": float(
                details["unexpected_loss_capital"].sum()
            ),
            "total_rwa": float(details["rwa"].sum()),
            "total_required_capital": float(details["required_capital"].sum()),
            "rwa_rate_to_ead": float(details["rwa"].sum() / details["ead"].sum()),
            "required_capital_rate_to_ead": float(
                details["required_capital"].sum() / details["ead"].sum()
            ),
        }
        if method_col is not None:
            row[method_col] = key
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.set_index(method_col) if method_col is not None else out
