"""Basel-style capital calculations for calibrated PD estimates.

The functions in this module implement a compact IRB-style approximation for
non-defaulted corporate exposures. They are intended for analytical comparison
of calibrated PD methods, not for production regulatory reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

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


@dataclass(frozen=True)
class CapitalStressScenario:
    """Parameter shocks for portfolio capital scenario analysis."""

    name: str
    lgd: float | None = None
    lgd_add: float = 0.0
    lgd_multiplier: float = 1.0
    ead: float | None = None
    ead_multiplier: float = 1.0
    maturity_years: float | None = None
    maturity_add: float = 0.0


def clip_pd(pd_values: np.ndarray, pd_floor: float = PD_FLOOR_CORPORATE) -> np.ndarray:
    """Clip PD values to the range used by the simplified IRB calculation."""

    return np.clip(np.asarray(pd_values, dtype=float), pd_floor, PD_CEILING)


def _as_float_array(values: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1:
        out = out.reshape(-1)
    if len(out) == 0:
        raise ValueError(f"{name} must not be empty")
    return out


def _resolve_assumptions(assumptions: IRBAssumptions | None) -> IRBAssumptions:
    return IRBAssumptions() if assumptions is None else assumptions


def _scenario_assumptions(
    base: IRBAssumptions,
    scenario: CapitalStressScenario,
) -> IRBAssumptions:
    lgd = scenario.lgd if scenario.lgd is not None else base.lgd
    lgd = (lgd + scenario.lgd_add) * scenario.lgd_multiplier
    if not 0.0 <= lgd <= 1.0:
        raise ValueError(f"Scenario '{scenario.name}' produces LGD outside [0, 1]: {lgd}")

    ead = scenario.ead if scenario.ead is not None else base.ead
    ead = ead * scenario.ead_multiplier
    if ead < 0.0:
        raise ValueError(f"Scenario '{scenario.name}' produces negative EAD: {ead}")

    maturity_years = (
        scenario.maturity_years
        if scenario.maturity_years is not None
        else base.maturity_years
    )
    maturity_years = maturity_years + scenario.maturity_add
    if maturity_years <= 0.0:
        raise ValueError(
            f"Scenario '{scenario.name}' produces non-positive maturity: {maturity_years}"
        )

    return replace(base, lgd=lgd, ead=ead, maturity_years=maturity_years)


def _resolve_ead_values(
    pd_values: np.ndarray,
    assumptions: IRBAssumptions,
    ead_values: np.ndarray | None,
) -> np.ndarray:
    if ead_values is None:
        return np.full_like(pd_values, assumptions.ead, dtype=float)

    out = _as_float_array(ead_values, "ead_values")
    if len(out) != len(pd_values):
        raise ValueError("ead_values must have the same length as pd_values")
    return out


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

    pd_values = clip_pd(_as_float_array(pd_values, "pd_values"), pd_floor=assumptions.pd_floor)
    ead_values = _resolve_ead_values(pd_values, assumptions, ead_values)

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


def default_capital_stress_scenarios() -> list[CapitalStressScenario]:
    """Return a compact default set of LGD, EAD and maturity stress scenarios."""

    return [
        CapitalStressScenario("Base"),
        CapitalStressScenario("LGD +10pp", lgd_add=0.10),
        CapitalStressScenario("LGD x1.20", lgd_multiplier=1.20),
        CapitalStressScenario("EAD x1.10", ead_multiplier=1.10),
        CapitalStressScenario("EAD x1.20", ead_multiplier=1.20),
        CapitalStressScenario("Maturity 3.5y", maturity_years=3.5),
        CapitalStressScenario("Maturity 5.0y", maturity_years=5.0),
        CapitalStressScenario(
            "Combined moderate",
            lgd_add=0.05,
            ead_multiplier=1.10,
            maturity_years=3.5,
        ),
        CapitalStressScenario(
            "Combined adverse",
            lgd_add=0.10,
            ead_multiplier=1.20,
            maturity_years=5.0,
        ),
    ]


def capital_stress_scenarios(
    pd_values: np.ndarray,
    assumptions: IRBAssumptions | None = None,
    scenarios: Iterable[CapitalStressScenario] | None = None,
    ead_values: np.ndarray | None = None,
    baseline_scenario: str = "Base",
) -> pd.DataFrame:
    """Summarise portfolio capital under LGD, EAD and maturity stress scenarios."""

    base = _resolve_assumptions(assumptions)
    pd_values = clip_pd(_as_float_array(pd_values, "pd_values"), pd_floor=base.pd_floor)
    base_ead_values = _resolve_ead_values(pd_values, base, ead_values)
    scenarios = list(default_capital_stress_scenarios() if scenarios is None else scenarios)

    rows = []
    for scenario in scenarios:
        stressed_assumptions = _scenario_assumptions(base, scenario)
        stressed_ead_values = base_ead_values * scenario.ead_multiplier
        if scenario.ead is not None:
            stressed_ead_values = np.full_like(pd_values, scenario.ead, dtype=float)

        row = summarize_irb_capital(
            pd_values,
            assumptions=stressed_assumptions,
            ead_values=stressed_ead_values,
        )
        row.update(
            {
                "scenario": scenario.name,
                "lgd": stressed_assumptions.lgd,
                "ead_multiplier": scenario.ead_multiplier,
                "maturity_years": stressed_assumptions.maturity_years,
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows).set_index("scenario")
    if baseline_scenario not in out.index:
        baseline_scenario = out.index[0]

    base_rwa = out.loc[baseline_scenario, "total_rwa"]
    base_capital = out.loc[baseline_scenario, "total_required_capital"]
    base_el = out.loc[baseline_scenario, "total_expected_loss"]

    out["delta_expected_loss"] = out["total_expected_loss"] - base_el
    out["delta_expected_loss_pct"] = out["delta_expected_loss"] / base_el
    out["delta_rwa"] = out["total_rwa"] - base_rwa
    out["delta_rwa_pct"] = out["delta_rwa"] / base_rwa
    out["delta_required_capital"] = out["total_required_capital"] - base_capital
    out["delta_required_capital_pct"] = out["delta_required_capital"] / base_capital
    return out


def capital_sensitivity_table(
    pd_values: np.ndarray,
    assumptions: IRBAssumptions | None = None,
    lgd_values: Iterable[float] | None = None,
    ead_multipliers: Iterable[float] | None = None,
    maturity_values: Iterable[float] | None = None,
    ead_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a grid of capital sensitivity to LGD, EAD and maturity assumptions."""

    base = _resolve_assumptions(assumptions)
    lgd_values = list([base.lgd, 0.45, 0.50, 0.60] if lgd_values is None else lgd_values)
    ead_multipliers = list([1.00, 1.10, 1.20] if ead_multipliers is None else ead_multipliers)
    maturity_values = list(
        [base.maturity_years, 3.5, 5.0] if maturity_values is None else maturity_values
    )

    scenarios = []
    for lgd in lgd_values:
        for ead_multiplier in ead_multipliers:
            for maturity_years in maturity_values:
                scenarios.append(
                    CapitalStressScenario(
                        name=(
                            f"LGD {lgd:.0%} | EAD x{ead_multiplier:.2f} "
                            f"| M {maturity_years:.1f}y"
                        ),
                        lgd=lgd,
                        ead_multiplier=ead_multiplier,
                        maturity_years=maturity_years,
                    )
                )

    out = capital_stress_scenarios(
        pd_values,
        assumptions=base,
        scenarios=[CapitalStressScenario("Base"), *scenarios],
        ead_values=ead_values,
        baseline_scenario="Base",
    )
    out = out.drop(index="Base")
    return out.reset_index().set_index(["lgd", "ead_multiplier", "maturity_years"])


def rwa_waterfall(
    pd_values: np.ndarray,
    base_assumptions: IRBAssumptions | None = None,
    target_assumptions: IRBAssumptions | None = None,
    ead_values: np.ndarray | None = None,
    target_ead_values: np.ndarray | None = None,
    order: tuple[str, ...] = ("lgd", "ead", "maturity"),
) -> pd.DataFrame:
    """Decompose RWA movement from base to target assumptions.

    The attribution is path-dependent: changing the order changes individual
    step deltas, while the final total movement remains the same.
    """

    base = _resolve_assumptions(base_assumptions)
    target = (
        replace(base, lgd=0.50, ead=base.ead * 1.10, maturity_years=3.5)
        if target_assumptions is None
        else target_assumptions
    )
    allowed_steps = {"lgd", "ead", "maturity"}
    unknown_steps = set(order) - allowed_steps
    if unknown_steps:
        raise ValueError(f"Unknown waterfall steps: {sorted(unknown_steps)}")
    if set(order) != allowed_steps or len(order) != len(allowed_steps):
        raise ValueError("order must contain each step exactly once: 'lgd', 'ead', 'maturity'")

    pd_values = clip_pd(_as_float_array(pd_values, "pd_values"), pd_floor=base.pd_floor)
    current_ead = _resolve_ead_values(pd_values, base, ead_values)
    target_ead = (
        _resolve_ead_values(pd_values, target, target_ead_values)
        if target_ead_values is not None
        else current_ead * (target.ead / base.ead)
    )

    current = base
    current_summary = summarize_irb_capital(pd_values, assumptions=current, ead_values=current_ead)
    base_rwa = current_summary["total_rwa"]
    rows = [
        {
            "step": "Base",
            "lgd": current.lgd,
            "ead_total": current_summary["total_ead"],
            "maturity_years": current.maturity_years,
            "rwa_before": np.nan,
            "rwa_after": current_summary["total_rwa"],
            "delta_rwa": 0.0,
            "delta_rwa_pct_of_base": 0.0,
            "required_capital_after": current_summary["total_required_capital"],
        }
    ]

    step_names = {
        "lgd": "LGD change",
        "ead": "EAD change",
        "maturity": "Maturity change",
    }
    for step in order:
        before_rwa = current_summary["total_rwa"]
        if step == "lgd":
            current = replace(current, lgd=target.lgd)
        elif step == "ead":
            current_ead = target_ead
            current = replace(current, ead=target.ead)
        elif step == "maturity":
            current = replace(current, maturity_years=target.maturity_years)

        current_summary = summarize_irb_capital(pd_values, assumptions=current, ead_values=current_ead)
        delta = current_summary["total_rwa"] - before_rwa
        rows.append(
            {
                "step": step_names[step],
                "lgd": current.lgd,
                "ead_total": current_summary["total_ead"],
                "maturity_years": current.maturity_years,
                "rwa_before": before_rwa,
                "rwa_after": current_summary["total_rwa"],
                "delta_rwa": delta,
                "delta_rwa_pct_of_base": delta / base_rwa,
                "required_capital_after": current_summary["total_required_capital"],
            }
        )

    final_rwa = rows[-1]["rwa_after"]
    rows.append(
        {
            "step": "Total movement",
            "lgd": target.lgd,
            "ead_total": target_ead.sum(),
            "maturity_years": target.maturity_years,
            "rwa_before": base_rwa,
            "rwa_after": final_rwa,
            "delta_rwa": final_rwa - base_rwa,
            "delta_rwa_pct_of_base": (final_rwa - base_rwa) / base_rwa,
            "required_capital_after": rows[-1]["required_capital_after"],
        }
    )
    return pd.DataFrame(rows)


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
