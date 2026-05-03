"""Synthetic credit portfolio generation for PD calibration experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Helper columns are kept for diagnostics, but must not be used as model inputs.
# In particular, risk_segment is the hidden data-generating segment; giving it to
# the model makes the experiment unrealistically easy and masks calibration risk.
FEATURE_EXCLUDE_COLUMNS = {"default", "origination_year", "risk_segment", "rating", "true_pd"}


RATING_LABELS = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}


def generate_credit_data(
    n_samples: int = 40_000,
    random_state: int = 42,
    portfolio: str = "stress",
) -> pd.DataFrame:
    """Generate a synthetic loan-level portfolio with a 12-month default target.

    Parameters
    ----------
    n_samples:
        Number of observations.
    random_state:
        Reproducibility seed.
    portfolio:
        ``"stress"`` creates a high-risk portfolio with a well-populated PD tail.
        ``"normal"`` creates a lower-risk portfolio closer to a typical performing
        retail/SME book.
    """

    if portfolio not in {"normal", "stress"}:
        raise ValueError("portfolio must be either 'normal' or 'stress'")

    rng = np.random.default_rng(random_state)

    if portfolio == "stress":
        segment_probs = [0.38, 0.34, 0.20, 0.08]
        base_intercept = -5.5
        subprime_lift = 0.55
        distressed_lift = 1.10
        oot_lift = 0.18
    else:
        segment_probs = [0.52, 0.33, 0.12, 0.03]
        base_intercept = -6.35
        subprime_lift = 0.35
        distressed_lift = 0.75
        oot_lift = 0.10

    risk_segment = rng.choice([0, 1, 2, 3], size=n_samples, p=segment_probs)
    origination_year = rng.choice(
        [2019, 2020, 2021, 2022, 2023, 2024],
        size=n_samples,
        p=[0.14, 0.14, 0.18, 0.18, 0.20, 0.16],
    )

    credit_score = np.zeros(n_samples)
    ltv = np.zeros(n_samples)
    dti = np.zeros(n_samples)
    employment_years = np.zeros(n_samples)
    loan_amount = np.zeros(n_samples)
    loan_term = np.zeros(n_samples)
    num_delinquencies = np.zeros(n_samples)
    loan_purpose = np.zeros(n_samples, dtype=int)

    for seg in [0, 1, 2, 3]:
        idx = risk_segment == seg
        m = int(idx.sum())

        if seg == 0:
            credit_score[idx] = rng.normal(760, 45, m)
            ltv[idx] = 0.05 + rng.beta(2.0, 5.0, m) * 0.60
            dti[idx] = 0.03 + rng.beta(2.0, 7.0, m) * 0.45
            employment_years[idx] = np.clip(rng.normal(9, 4, m), 0, 35)
            loan_amount[idx] = rng.lognormal(11.6, 0.45, m) / 1000
            loan_term[idx] = rng.choice([12, 24, 36, 48, 60], m, p=[0.15, 0.25, 0.25, 0.20, 0.15])
            num_delinquencies[idx] = rng.poisson(0.08, m)
            loan_purpose[idx] = rng.choice([0, 1, 2], m, p=[0.50, 0.20, 0.30])
        elif seg == 1:
            credit_score[idx] = rng.normal(660, 55, m)
            ltv[idx] = 0.10 + rng.beta(2.2, 2.8, m) * 0.75
            dti[idx] = 0.05 + rng.beta(2.2, 3.5, m) * 0.60
            employment_years[idx] = np.clip(rng.normal(6, 4, m), 0, 35)
            loan_amount[idx] = rng.lognormal(12.1, 0.55, m) / 1000
            loan_term[idx] = rng.choice([12, 24, 36, 48, 60, 84], m, p=[0.08, 0.18, 0.25, 0.22, 0.20, 0.07])
            num_delinquencies[idx] = rng.poisson(0.35, m)
            loan_purpose[idx] = rng.choice([0, 1, 2], m, p=[0.40, 0.25, 0.35])
        elif seg == 2:
            credit_score[idx] = rng.normal(540, 65, m)
            ltv[idx] = 0.20 + rng.beta(2.8, 1.9, m) * 0.75
            dti[idx] = 0.10 + rng.beta(2.8, 2.0, m) * 0.70
            employment_years[idx] = np.clip(rng.normal(3.5, 3, m), 0, 30)
            loan_amount[idx] = rng.lognormal(12.5, 0.70, m) / 1000
            loan_term[idx] = rng.choice([24, 36, 48, 60, 84, 120], m, p=[0.10, 0.20, 0.20, 0.20, 0.18, 0.12])
            num_delinquencies[idx] = rng.poisson(1.0, m)
            loan_purpose[idx] = rng.choice([0, 1, 2], m, p=[0.25, 0.25, 0.50])
        else:
            credit_score[idx] = rng.normal(430, 55, m)
            ltv[idx] = 0.45 + rng.beta(3.2, 1.4, m) * 0.50
            dti[idx] = 0.22 + rng.beta(3.0, 1.4, m) * 0.65
            employment_years[idx] = np.clip(rng.normal(1.8, 2, m), 0, 20)
            loan_amount[idx] = rng.lognormal(12.7, 0.85, m) / 1000
            loan_term[idx] = rng.choice([36, 48, 60, 84, 120], m, p=[0.10, 0.18, 0.28, 0.24, 0.20])
            num_delinquencies[idx] = rng.poisson(2.0, m)
            loan_purpose[idx] = rng.choice([0, 1, 2], m, p=[0.15, 0.20, 0.65])

    credit_score = np.clip(credit_score, 300, 850)
    ltv = np.clip(ltv, 0.01, 1.0)
    dti = np.clip(dti, 0.01, 1.0)
    employment_years = np.clip(employment_years, 0, 40)

    cs_norm = (credit_score - 300) / 550
    log_odds = (
        base_intercept
        + 3.6 * (1 - cs_norm) ** 1.7
        + 2.2 * ltv**1.5
        + 1.9 * dti**1.3
        - 0.04 * employment_years
        + 0.38 * num_delinquencies
        + 0.18 * (loan_purpose == 2).astype(float)
        + 0.12 * (loan_term >= 84).astype(float)
        + 0.18 * (loan_amount > np.quantile(loan_amount, 0.80)).astype(float)
        + 0.28 * ((ltv > 0.85) & (dti > 0.55)).astype(float)
        + 0.32 * ((credit_score < 500) & (num_delinquencies >= 2)).astype(float)
        + subprime_lift * (risk_segment == 2).astype(float)
        + distressed_lift * (risk_segment == 3).astype(float)
        + 0.32 * (origination_year == 2020).astype(float)
        + oot_lift * (origination_year == 2024).astype(float)
    )

    true_pd = 1.0 / (1.0 + np.exp(-log_odds))
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
            "rating": pd.Series(risk_segment).map(RATING_LABELS).to_numpy(),
            "origination_year": origination_year,
            "true_pd": true_pd,
            "default": default,
        }
    )


def get_oot_split(df: pd.DataFrame, target_col: str = "default"):
    """Split a portfolio into train, calibration and OOT test samples."""

    feature_cols = [c for c in df.columns if c not in FEATURE_EXCLUDE_COLUMNS | {target_col}]
    train_mask = df["origination_year"].isin([2019, 2020, 2021])
    calib_mask = df["origination_year"].isin([2022, 2023])
    test_mask = df["origination_year"] == 2024

    return (
        df.loc[train_mask, feature_cols],
        df.loc[calib_mask, feature_cols],
        df.loc[test_mask, feature_cols],
        df.loc[train_mask, target_col],
        df.loc[calib_mask, target_col],
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
