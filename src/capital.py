"""Basel-style capital calculations for calibrated PD estimates.

The functions in this module implement a compact IRB-style approximation for
non-defaulted corporate exposures. They are intended for analytical comparison
of calibrated PD methods, not for production regulatory reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm


PD_FLOOR_CORPORATE = 0.0003
PD_CEILING = 1.0 - 1e-6
CAPITAL_RATIO = 0.08


@dataclass(frozen=True)
class IRBAssumptions:
    """Core assumptions for the simplified IRB capital calculation."""

    lgd: float = 0.40
    maturity_years: float = 2.5
    ead: float = 1_000_000.0
    capital_ratio: float = CAPITAL_RATIO
    pd_floor: float = PD_FLOOR_CORPORATE


def clip_pd(pd_values: np.ndarray, pd_floor: float = PD_FLOOR_CORPORATE) -> np.ndarray:
    """Clip PD values to the range used by the simplified IRB calculation."""

    return np.clip(np.asarray(pd_values, dtype=float), pd_floor, PD_CEILING)


def corporate_asset_correlation(pd_values: np.ndarray) -> np.ndarray:
    """Corporate IRB asset correlation as a function of PD."""

    pd_values = clip_pd(pd_values)
    scale = (1.0 - np.exp(-50.0 * pd_values)) / (1.0 - np.exp(-50.0))
    return 0.12 * scale + 0.24 * (1.0 - scale)


def maturity_adjustment(pd_values: np.ndarray, maturity_years: float = 2.5) -> np.ndarray:
    """Maturity adjustment for corporate IRB capital."""

    pd_values = clip_pd(pd_values)
    b = (0.11852 - 0.05478 * np.log(pd_values)) ** 2
    return (1.0 + (maturity_years - 2.5) * b) / (1.0 - 1.5 * b)


def capital_requirement_k(
    pd_values: np.ndarray,
    lgd: float = 0.40,
    maturity_years: float = 2.5,
) -> np.ndarray:
    """Unexpected-loss capital requirement per unit of EAD.

    This follows the common corporate IRB form:
    K = LGD * (N((G(PD) + sqrt(R) * G(0.999)) / sqrt(1 - R)) - PD) * MA.
    """

    pd_values = clip_pd(pd_values)
    r = corporate_asset_correlation(pd_values)
    ma = maturity_adjustment(pd_values, maturity_years=maturity_years)
    conditional_pd = norm.cdf(
        (norm.ppf(pd_values) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1.0 - r)
    )
    return lgd * (conditional_pd - pd_values) * ma


def calculate_irb_capital(
    pd_values: np.ndarray,
    assumptions: IRBAssumptions | None = None,
    ead_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return row-level Basel-style EL, capital and RWA for PD estimates."""

    if assumptions is None:
        assumptions = IRBAssumptions()

    pd_values = clip_pd(pd_values, pd_floor=assumptions.pd_floor)
    if ead_values is None:
        ead_values = np.full_like(pd_values, assumptions.ead, dtype=float)
    else:
        ead_values = np.asarray(ead_values, dtype=float)

    k = capital_requirement_k(
        pd_values,
        lgd=assumptions.lgd,
        maturity_years=assumptions.maturity_years,
    )
    expected_loss = pd_values * assumptions.lgd * ead_values
    unexpected_loss_capital = k * ead_values
    rwa = unexpected_loss_capital / assumptions.capital_ratio

    return pd.DataFrame(
        {
            "pd": pd_values,
            "ead": ead_values,
            "lgd": assumptions.lgd,
            "maturity_years": assumptions.maturity_years,
            "asset_correlation": corporate_asset_correlation(pd_values),
            "maturity_adjustment": maturity_adjustment(
                pd_values, maturity_years=assumptions.maturity_years
            ),
            "expected_loss": expected_loss,
            "capital_requirement_k": k,
            "unexpected_loss_capital": unexpected_loss_capital,
            "rwa": rwa,
            "required_capital": assumptions.capital_ratio * rwa,
        }
    )


def summarize_irb_capital(
    pd_values: np.ndarray,
    assumptions: IRBAssumptions | None = None,
    ead_values: np.ndarray | None = None,
) -> dict:
    """Summarise portfolio-level capital metrics for one PD vector."""

    details = calculate_irb_capital(pd_values, assumptions=assumptions, ead_values=ead_values)
    total_ead = details["ead"].sum()
    return {
        "avg_pd": details["pd"].mean(),
        "total_ead": total_ead,
        "total_expected_loss": details["expected_loss"].sum(),
        "total_unexpected_loss_capital": details["unexpected_loss_capital"].sum(),
        "total_rwa": details["rwa"].sum(),
        "total_required_capital": details["required_capital"].sum(),
        "expected_loss_rate_to_ead": details["expected_loss"].sum() / total_ead,
        "rwa_rate_to_ead": details["rwa"].sum() / total_ead,
        "required_capital_rate_to_ead": details["required_capital"].sum() / total_ead,
    }


def compare_irb_capital_by_method(
    predictions: Mapping[str, np.ndarray],
    assumptions: IRBAssumptions | None = None,
    baseline_method: str | None = None,
    ead_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare capital impact across multiple PD calibration methods."""

    rows = []
    for method, pd_values in predictions.items():
        row = summarize_irb_capital(
            pd_values,
            assumptions=assumptions,
            ead_values=ead_values,
        )
        row["method"] = method
        rows.append(row)

    out = pd.DataFrame(rows).set_index("method")
    if baseline_method is None:
        baseline_method = out.index[0]

    base_rwa = out.loc[baseline_method, "total_rwa"]
    base_capital = out.loc[baseline_method, "total_required_capital"]

    out["rwa_saving_vs_baseline"] = base_rwa - out["total_rwa"]
    out["rwa_saving_vs_baseline_pct"] = out["rwa_saving_vs_baseline"] / base_rwa
    out["capital_saving_vs_baseline"] = base_capital - out["total_required_capital"]
    out["capital_saving_vs_baseline_pct"] = (
        out["capital_saving_vs_baseline"] / base_capital
    )
    out["capital_ratio_if_keep_baseline_capital"] = base_capital / out["total_rwa"]
    return out
