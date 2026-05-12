"""Synthetic credit portfolio generation for PD calibration experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


# Helper columns are kept for diagnostics, but must not be used as model inputs.
# In particular, risk_segment is the hidden data-generating segment; giving it to
# the model makes the experiment unrealistically easy and masks calibration risk.
FEATURE_EXCLUDE_COLUMNS = {"default", "origination_year", "risk_segment", "rating", "true_pd"}


PORTFOLIO_RATING_ORDER = ("A", "B", "C", "D", "E")
PORTFOLIO_RATING_PD_BOUNDS = {
    "A": (0.0005, 0.0010, 0.00075),
    "B": (0.0010, 0.0065, 0.00350),
    "C": (0.0065, 0.0291, 0.01780),
    "D": (0.0291, 0.2600, 0.09500),
    "E": (0.2600, 1.0000, 0.40000),
}
PORTFOLIO_YEARS = (2020, 2021, 2022, 2023, 2024)
PORTFOLIO_YEAR_PROBS = (0.18, 0.19, 0.20, 0.21, 0.22)
RATING_LABELS = {idx: rating for idx, rating in enumerate(PORTFOLIO_RATING_ORDER)}


@dataclass(frozen=True)
class RatingPortfolioConfig:
    """Parameters for synthetic portfolio construction."""

    name: str
    n_samples: int = 40_000
    rating_order: tuple[str, ...] = PORTFOLIO_RATING_ORDER
    rating_pd_bounds: Mapping[str, tuple[float, float, float]] | None = None
    rating_mix: Mapping[str, float] | None = None
    years: tuple[int, ...] = PORTFOLIO_YEARS
    year_probs: tuple[float, ...] = PORTFOLIO_YEAR_PROBS
    oot_pd_lift: float = 0.06
    pandemic_pd_lift: float = 0.04


DEFAULT_PORTFOLIO_CONFIGS = {
    "stress": RatingPortfolioConfig(
        name="stress",
        rating_pd_bounds=PORTFOLIO_RATING_PD_BOUNDS,
        rating_mix={
            "A": 0.03,
            "B": 0.36,
            "C": 0.41,
            "D": 0.16,
            "E": 0.04,
        },
        oot_pd_lift=0.08,
        pandemic_pd_lift=0.05,
    ),
    "normal": RatingPortfolioConfig(
        name="normal",
        rating_pd_bounds=PORTFOLIO_RATING_PD_BOUNDS,
        rating_mix={
            "A": 0.10,
            "B": 0.48,
            "C": 0.32,
            "D": 0.08,
            "E": 0.02,
        },
        oot_pd_lift=0.03,
        pandemic_pd_lift=0.02,
    ),
}


def _validate_probability_vector(values: Iterable[float], name: str) -> tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if len(out) == 0:
        raise ValueError(f"{name} must not be empty")
    if any(value < 0.0 for value in out):
        raise ValueError(f"{name} must not contain negative values")
    if not np.isclose(sum(out), 1.0):
        raise ValueError(f"{name} must sum to 1.0")
    return out


def get_rating_portfolio_config(portfolio: str = "stress") -> RatingPortfolioConfig:
    """Return validated parameters for the requested synthetic portfolio scenario."""

    if portfolio not in DEFAULT_PORTFOLIO_CONFIGS:
        known = ", ".join(sorted(DEFAULT_PORTFOLIO_CONFIGS))
        raise ValueError(f"portfolio must be one of: {known}")

    config = DEFAULT_PORTFOLIO_CONFIGS[portfolio]
    bounds = config.rating_pd_bounds or PORTFOLIO_RATING_PD_BOUNDS
    mix = config.rating_mix
    if mix is None:
        raise ValueError(f"Portfolio scenario '{portfolio}' has no rating mix")
    missing_bounds = set(config.rating_order) - set(bounds)
    missing_mix = set(config.rating_order) - set(mix)
    if missing_bounds:
        raise ValueError(f"Missing PD bounds for ratings: {sorted(missing_bounds)}")
    if missing_mix:
        raise ValueError(f"Missing rating mix for ratings: {sorted(missing_mix)}")

    _validate_probability_vector((mix[rating] for rating in config.rating_order), "rating_mix")
    _validate_probability_vector(config.year_probs, "year_probs")
    if len(config.years) != len(config.year_probs):
        raise ValueError("years and year_probs must have the same length")

    for rating in config.rating_order:
        lower, upper, average = bounds[rating]
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(f"Invalid PD interval for rating {rating}: {(lower, upper)}")
        if not lower <= average <= upper:
            raise ValueError(f"Representative PD is outside interval for {rating}")

    return config


RATING_FEATURE_PROFILES = {
    "A": {
        "score": 800,
        "score_slope": 45,
        "ltv_base": 0.08,
        "ltv_span": 0.25,
        "dti_base": 0.05,
        "dti_span": 0.20,
        "employment": 10.0,
        "delinq": 0.04,
        "loan_amount_mu": 11.45,
        "loan_amount_sigma": 0.35,
    },
    "B": {
        "score": 725,
        "score_slope": 55,
        "ltv_base": 0.12,
        "ltv_span": 0.38,
        "dti_base": 0.08,
        "dti_span": 0.32,
        "employment": 7.5,
        "delinq": 0.18,
        "loan_amount_mu": 11.85,
        "loan_amount_sigma": 0.45,
    },
    "C": {
        "score": 640,
        "score_slope": 70,
        "ltv_base": 0.22,
        "ltv_span": 0.48,
        "dti_base": 0.14,
        "dti_span": 0.42,
        "employment": 5.0,
        "delinq": 0.55,
        "loan_amount_mu": 12.15,
        "loan_amount_sigma": 0.55,
    },
    "D": {
        "score": 545,
        "score_slope": 80,
        "ltv_base": 0.36,
        "ltv_span": 0.50,
        "dti_base": 0.22,
        "dti_span": 0.50,
        "employment": 3.0,
        "delinq": 1.20,
        "loan_amount_mu": 12.35,
        "loan_amount_sigma": 0.70,
    },
    "E": {
        "score": 455,
        "score_slope": 75,
        "ltv_base": 0.52,
        "ltv_span": 0.42,
        "dti_base": 0.32,
        "dti_span": 0.55,
        "employment": 1.8,
        "delinq": 2.20,
        "loan_amount_mu": 12.55,
        "loan_amount_sigma": 0.85,
    },
}


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _draw_pd_inside_rating(
    rng: np.random.Generator,
    rating: str,
    year_values: np.ndarray,
    portfolio: str,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    config = get_rating_portfolio_config(portfolio)
    lower, upper, _ = (config.rating_pd_bounds or {})[rating]

    if rating in {"A", "B", "C"}:
        risk_intensity = rng.beta(1.05, 1.05, size)
    elif rating == "D":
        risk_intensity = rng.beta(0.90, 1.45, size)
    elif rating == "E":
        tail_mask = rng.random(size) < 0.18
        risk_intensity = rng.beta(0.85, 2.00, size)
        risk_intensity[tail_mask] = rng.random(int(tail_mask.sum()))
    else:
        risk_intensity = rng.random(size)

    year_shift = np.zeros(size)
    year_shift[year_values == config.years[0]] += config.pandemic_pd_lift
    year_shift[year_values == config.years[-1]] += config.oot_pd_lift
    risk_intensity = 1.0 / (1.0 + np.exp(-(_logit(risk_intensity) + year_shift)))
    true_pd = lower + risk_intensity * (upper - lower)
    return np.clip(true_pd, lower, upper), risk_intensity


def generate_credit_data(
    n_samples: int | None = None,
    random_state: int = 42,
    portfolio: str = "stress",
) -> pd.DataFrame:
    """Generate a synthetic loan-level portfolio with a 12-month default target.

    Parameters
    ----------
    n_samples:
        Number of observations. If omitted, the scenario value from
        :class:`RatingPortfolioConfig` is used.
    random_state:
        Reproducibility seed.
    portfolio:
        ``"stress"`` creates a high-risk portfolio concentrated in B-C ratings.
        ``"normal"`` creates a lower-risk portfolio closer to a typical performing
        retail/SME book.
    """

    config = get_rating_portfolio_config(portfolio)
    n_samples = config.n_samples if n_samples is None else int(n_samples)
    rng = np.random.default_rng(random_state)

    rating_order = tuple(config.rating_order)
    rating_probs = [config.rating_mix[rating] for rating in rating_order]  # type: ignore[index]
    rating = rng.choice(rating_order, size=n_samples, p=rating_probs)
    segment_map = {label: idx for idx, label in enumerate(rating_order)}
    risk_segment = np.array([segment_map[label] for label in rating], dtype=int)
    origination_year = rng.choice(
        config.years,
        size=n_samples,
        p=config.year_probs,
    )

    credit_score = np.zeros(n_samples)
    ltv = np.zeros(n_samples)
    dti = np.zeros(n_samples)
    employment_years = np.zeros(n_samples)
    loan_amount = np.zeros(n_samples)
    loan_term = np.zeros(n_samples)
    num_delinquencies = np.zeros(n_samples)
    loan_purpose = np.zeros(n_samples, dtype=int)
    true_pd = np.zeros(n_samples)

    for rating_label in rating_order:
        idx = rating == rating_label
        m = int(idx.sum())
        if m == 0:
            continue

        rating_pd, risk_intensity = _draw_pd_inside_rating(
            rng,
            rating_label,
            origination_year[idx],
            portfolio,
            m,
        )
        true_pd[idx] = rating_pd
        profile = RATING_FEATURE_PROFILES[rating_label]
        credit_score[idx] = rng.normal(
            profile["score"] - profile["score_slope"] * risk_intensity,
            28 + 20 * risk_intensity,
            m,
        )
        ltv[idx] = profile["ltv_base"] + profile["ltv_span"] * risk_intensity + rng.normal(0, 0.04, m)
        dti[idx] = profile["dti_base"] + profile["dti_span"] * risk_intensity + rng.normal(0, 0.035, m)
        employment_years[idx] = np.clip(
            rng.normal(profile["employment"] * (1.0 - 0.45 * risk_intensity), 2.5, m),
            0,
            40,
        )
        loan_amount[idx] = rng.lognormal(
            profile["loan_amount_mu"] + 0.18 * risk_intensity,
            profile["loan_amount_sigma"],
            m,
        ) / 1000
        term_probs = np.array([0.16, 0.20, 0.24, 0.20, 0.14, 0.06], dtype=float)
        term_probs = term_probs + risk_intensity.mean() * np.array([-0.05, -0.04, -0.02, 0.02, 0.04, 0.05])
        term_probs = np.clip(term_probs, 0.01, None)
        term_probs = term_probs / term_probs.sum()
        loan_term[idx] = rng.choice([12, 24, 36, 48, 60, 84], m, p=term_probs)
        num_delinquencies[idx] = rng.poisson(profile["delinq"] + 1.5 * risk_intensity, m)
        loan_purpose[idx] = rng.choice(
            [0, 1, 2],
            m,
            p=[
                max(0.10, 0.48 - 0.25 * risk_intensity.mean()),
                0.25,
                min(0.65, 0.27 + 0.25 * risk_intensity.mean()),
            ],
        )

    credit_score = np.clip(credit_score, 300, 850)
    ltv = np.clip(ltv, 0.01, 1.0)
    dti = np.clip(dti, 0.01, 1.0)
    employment_years = np.clip(employment_years, 0, 40)

    default = rng.binomial(n=1, p=true_pd)

    return pd.DataFrame(
        {
            "credit_score": credit_score.round(0),
            "ltv": ltv.round(4),
            "dti": dti.round(4),
            "employment_years": employment_years.round(1),
            "loan_amount": loan_amount.round(1),
            "loan_term": loan_term.astype(float),
            "num_delinquencies": num_delinquencies.astype(float),
            "loan_purpose": loan_purpose,
            "risk_segment": risk_segment,
            "rating": rating,
            "origination_year": origination_year,
            "true_pd": true_pd,
            "default": default,
        }
    )


def get_oot_split(df: pd.DataFrame, target_col: str = "default"):
    """Split a portfolio into one in-time modelling sample and an OOT test.

    The latest origination year is reserved for the OOT test. All earlier years
    are returned both as the RF training sample and as the calibrator fitting
    sample. This mirrors the methodology used in the notebook: estimate the base
    RF and the post-calibration mapping on the same historical in-time period,
    then evaluate only once on the future OOT portfolio.
    """

    feature_cols = [c for c in df.columns if c not in FEATURE_EXCLUDE_COLUMNS | {target_col}]
    years = sorted(df["origination_year"].unique())
    if len(years) < 2:
        raise ValueError("OOT split requires at least two origination years")
    train_calib_years = years[:-1]
    test_year = years[-1]
    train_calib_mask = df["origination_year"].isin(train_calib_years)
    test_mask = df["origination_year"] == test_year

    return (
        df.loc[train_calib_mask, feature_cols],
        df.loc[train_calib_mask, feature_cols],
        df.loc[test_mask, feature_cols],
        df.loc[train_calib_mask, target_col],
        df.loc[train_calib_mask, target_col],
        df.loc[test_mask, target_col],
    )


def portfolio_summary(df: pd.DataFrame) -> dict:
    """Return compact descriptive statistics for a generated portfolio."""

    return {
        "n_obs": int(len(df)),
        "default_rate": float(df["default"].mean()),
        "avg_true_pd": float(df["true_pd"].mean()),
        "p95_true_pd": float(df["true_pd"].quantile(0.95)),
        "max_true_pd": float(df["true_pd"].max()),
    }


if __name__ == "__main__":
    for name in ["normal", "stress"]:
        sample = generate_credit_data(portfolio=name)
        print(name, portfolio_summary(sample))
