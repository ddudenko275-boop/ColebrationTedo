import numpy as np
import pandas as pd
import pytest

from src.capital import (
    CapitalStressScenario,
    IRBAssumptions,
    calculate_irb_capital,
    capital_sensitivity_table,
    capital_stress_scenarios,
    rwa_waterfall,
    summarize_irb_capital,
)


def sample_pd() -> np.ndarray:
    return np.linspace(0.005, 0.20, 100)


def test_calculate_irb_capital_rejects_ead_length_mismatch():
    with pytest.raises(ValueError, match="ead_values must have the same length"):
        calculate_irb_capital(sample_pd(), ead_values=np.array([1_000_000.0]))


def test_capital_stress_scenarios_default_shape_and_base_delta():
    out = capital_stress_scenarios(sample_pd(), assumptions=IRBAssumptions())

    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == 9
    assert "Base" in out.index
    assert out.loc["Base", "delta_rwa"] == pytest.approx(0.0)
    assert out.loc["Combined adverse", "total_rwa"] > out.loc["Base", "total_rwa"]
    assert out.loc["Combined adverse", "delta_required_capital"] > 0.0


def test_custom_stress_scenario_applies_ead_vector_multiplier():
    pd_values = sample_pd()
    ead_values = np.linspace(500_000.0, 1_500_000.0, len(pd_values))
    scenarios = [
        CapitalStressScenario("Base"),
        CapitalStressScenario("EAD up", ead_multiplier=1.25),
    ]

    out = capital_stress_scenarios(
        pd_values,
        assumptions=IRBAssumptions(),
        scenarios=scenarios,
        ead_values=ead_values,
    )

    assert out.loc["EAD up", "total_ead"] == pytest.approx(
        out.loc["Base", "total_ead"] * 1.25
    )
    assert out.loc["EAD up", "total_rwa"] == pytest.approx(
        out.loc["Base", "total_rwa"] * 1.25
    )


def test_capital_sensitivity_table_uses_base_assumptions_for_deltas():
    base = IRBAssumptions(lgd=0.40, ead=1_000_000.0, maturity_years=2.5)
    out = capital_sensitivity_table(
        sample_pd(),
        assumptions=base,
        lgd_values=[0.50],
        ead_multipliers=[1.10],
        maturity_values=[3.5],
    )

    assert out.shape[0] == 1
    row = out.iloc[0]
    assert row["total_rwa"] > row["delta_rwa"] > 0.0
    assert row["delta_rwa_pct"] > 0.0


def test_rwa_waterfall_matches_direct_target_rwa():
    pd_values = sample_pd()
    base = IRBAssumptions(lgd=0.40, ead=1_000_000.0, maturity_years=2.5)
    target = IRBAssumptions(lgd=0.50, ead=1_100_000.0, maturity_years=3.5)

    out = rwa_waterfall(pd_values, base_assumptions=base, target_assumptions=target)
    direct_target_rwa = summarize_irb_capital(pd_values, assumptions=target)["total_rwa"]

    assert out.iloc[-1]["step"] == "Total movement"
    assert out.iloc[-1]["rwa_after"] == pytest.approx(direct_target_rwa)
    assert out.iloc[-1]["delta_rwa"] == pytest.approx(
        direct_target_rwa - out.iloc[0]["rwa_after"]
    )


def test_rwa_waterfall_rejects_incomplete_order():
    with pytest.raises(ValueError, match="order must contain each step exactly once"):
        rwa_waterfall(sample_pd(), order=("lgd", "ead"))
