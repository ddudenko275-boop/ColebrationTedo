import numpy as np
import pandas as pd
import pytest

from data.generate_data import generate_credit_data, get_oot_split, get_same_year_calibration_test_split
from src.portfolio import (
    MASTER_SCALE_RATINGS,
    apply_delta_logit,
    assign_pd_master_scale_ratings,
    assign_master_scale_ratings,
    calibrate_pd_to_target,
    compare_methods_by_rating_master_scale,
    compare_methods_by_historical_panel,
    delta_logit_scenarios,
    historical_portfolio_panel,
    master_scale_pd_bound_capital,
    method_master_scale_distribution,
    method_portfolio_summary,
    master_scale_table,
    portfolio_average_pd,
    rating_migration_matrix,
    rating_master_scale,
    rating_scale_capital,
    rating_scale_capital_by_rating,
    score_distribution_table,
    summarize_rating_scale,
    validate_common_rating_structure,
)


def sample_portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origination_year": [2022, 2022, 2022, 2023, 2023],
            "rating": ["A", "A", "B", "A", "B"],
            "default": [0, 1, 1, 0, 1],
            "pd_model": [0.01, 0.02, 0.08, 0.03, 0.10],
            "ead": [100.0, 200.0, 300.0, 100.0, 400.0],
        }
    )


def test_generated_rating_is_kept_out_of_model_features():
    df = generate_credit_data(n_samples=500, random_state=7)
    x_train, x_calib, x_test, *_ = get_oot_split(df)

    assert "rating" in df.columns
    assert set(df["rating"].unique()) <= {"A", "B", "C", "D"}
    assert "rating" not in x_train.columns
    assert "rating" not in x_calib.columns
    assert "rating" not in x_test.columns


def test_normal_portfolio_is_lower_risk_than_stress_portfolio():
    normal = generate_credit_data(n_samples=2_000, random_state=7, portfolio="normal")
    stress = generate_credit_data(n_samples=2_000, random_state=7, portfolio="stress")

    assert normal["true_pd"].mean() < stress["true_pd"].mean()
    assert normal["true_pd"].quantile(0.95) < stress["true_pd"].quantile(0.95)


def test_same_year_calibration_test_split_uses_2024_for_both_calib_and_test():
    df = generate_credit_data(n_samples=500, random_state=7)
    x_train, x_calib, x_test, y_train, y_calib, y_test = get_same_year_calibration_test_split(df)

    assert set(df.loc[x_train.index, "origination_year"].unique()) <= {2019, 2020, 2021, 2022, 2023}
    assert set(df.loc[x_calib.index, "origination_year"].unique()) == {2024}
    assert x_calib.index.equals(x_test.index)
    assert y_calib.index.equals(y_test.index)


def test_historical_portfolio_panel_preserves_period_rating_structure():
    df = sample_portfolio()

    panel = historical_portfolio_panel(df, pd_col="pd_model")
    row = panel[
        (panel["origination_year"] == 2022)
        & (panel["rating"] == "A")
    ].iloc[0]

    assert row["n_obs"] == 2
    assert row["defaults"] == 1
    assert row["avg_pd"] == pytest.approx(0.015)
    assert row["pd_min"] == pytest.approx(0.01)
    assert row["pd_max"] == pytest.approx(0.02)
    assert row["expected_defaults"] == pytest.approx(0.03)
    assert row["observed_default_rate"] == pytest.approx(0.50)
    assert row["portfolio_count_share"] == pytest.approx(2 / 3)


def test_historical_portfolio_panel_supports_ead_weighted_pd():
    df = sample_portfolio()

    panel = historical_portfolio_panel(df, pd_col="pd_model", ead_col="ead")
    row = panel[
        (panel["origination_year"] == 2022)
        & (panel["rating"] == "A")
    ].iloc[0]

    assert row["total_ead"] == pytest.approx(300.0)
    assert row["ead_weighted_pd"] == pytest.approx((0.01 * 100 + 0.02 * 200) / 300)
    assert row["expected_defaults"] == pytest.approx(0.01 + 0.02)
    assert row["expected_default_ead"] == pytest.approx(0.01 * 100 + 0.02 * 200)
    assert row["defaulted_ead"] == pytest.approx(200.0)


def test_portfolio_average_pd_is_only_top_line_aggregate():
    df = sample_portfolio()

    assert portfolio_average_pd(df, "pd_model") == pytest.approx(np.mean(df["pd_model"]))
    assert portfolio_average_pd(df, "pd_model", ead_col="ead") == pytest.approx(
        np.average(df["pd_model"], weights=df["ead"])
    )


def test_compare_methods_by_historical_panel_adds_method_dimension():
    df = sample_portfolio()
    predictions = {
        "raw": df["pd_model"],
        "calibrated": df["pd_model"] * 1.2,
    }

    panel = compare_methods_by_historical_panel(df, predictions)

    assert set(panel["method"]) == {"raw", "calibrated"}
    assert panel.groupby("method").size().to_dict() == {"calibrated": 4, "raw": 4}


def test_compare_methods_rejects_prediction_length_mismatch():
    df = sample_portfolio()

    with pytest.raises(ValueError, match="Prediction length"):
        compare_methods_by_historical_panel(df, {"bad": np.array([0.01, 0.02])})


def test_method_portfolio_summary_compares_aggregate_pd_methods():
    df = sample_portfolio()
    predictions = {
        "raw": df["pd_model"],
        "calibrated": df["pd_model"] * 1.2,
    }

    out = method_portfolio_summary(df, predictions)

    assert list(out.index) == ["raw", "calibrated"]
    assert out.loc["calibrated", "expected_defaults"] > out.loc["raw", "expected_defaults"]
    assert out.loc["raw", "defaults"] == pytest.approx(df["default"].sum())


def test_assign_master_scale_ratings_uses_reference_breakpoints():
    ref = np.arange(13, dtype=float)
    scores = np.array([0.0, 6.0, 12.0])

    assigned = assign_master_scale_ratings(scores, reference_scores=ref)

    assert list(assigned.astype(str)) == ["A1", "C1", "E"]
    assert len(MASTER_SCALE_RATINGS) == 13


def test_assign_pd_master_scale_ratings_uses_fixed_pd_boundaries():
    assigned = assign_pd_master_scale_ratings(np.array([0.0003, 0.0007, 0.20, 0.60]))

    assert list(assigned.astype(str)) == ["A1", "A2", "D3", "E"]
    scale = master_scale_table()
    assert list(scale.columns) == ["rating", "pd_lower", "pd_upper", "pd_avg"]
    assert scale.loc[scale["rating"] == "E", "pd_upper"].iloc[0] == pytest.approx(1.0)


def test_calibrate_pd_to_target_matches_weighted_average():
    pd_values = np.array([0.01, 0.03, 0.10, 0.30])
    weights = np.array([100.0, 200.0, 300.0, 400.0])

    calibrated = calibrate_pd_to_target(pd_values, weights, target_pd=0.12)

    assert np.average(calibrated, weights=weights) == pytest.approx(0.12)
    assert np.all(np.diff(calibrated) > 0)


def test_delta_logit_scenarios_shift_pd_without_changing_order():
    pd_values = np.array([0.01, 0.03, 0.10])
    shifted = apply_delta_logit(pd_values, delta=0.5)
    scenarios = delta_logit_scenarios(pd_values, deltas=[-0.5, 0.0, 0.5])

    assert np.all(np.diff(shifted) > 0)
    assert shifted.mean() > pd_values.mean()
    assert scenarios.loc["delta +0.500", "avg_pd"] > scenarios.loc["delta +0.000", "avg_pd"]


def test_rating_master_scale_calibrates_rating_pd_to_target():
    ratings = ["A1", "A2", "A3", "B1"]
    df = pd.DataFrame(
        {
            "master_rating": ratings * 3,
            "default": [0, 0, 0, 1] * 3,
        }
    )
    pd_values = np.tile([0.01, 0.03, 0.08, 0.20], 3)
    score_values = np.tile([0.1, 0.2, 0.3, 0.4], 3)

    scale = rating_master_scale(
        df,
        pd_values=pd_values,
        score_values=score_values,
        target_pd=0.10,
        rating_order=tuple(ratings),
    )
    summary = summarize_rating_scale(scale)

    assert list(scale["rating"]) == ratings
    assert summary.loc[0, "target_weighted_pd"] == pytest.approx(0.10)
    assert scale["total_ead"].sum() == pytest.approx(len(df) * 1_000_000.0)
    assert np.all(np.diff(scale["pd_rating"]) > 0)
    assert np.allclose(scale["one_minus_pd"], 1.0 - scale["pd_rating"])


def test_compare_methods_by_rating_master_scale_adds_method_dimension():
    ratings = ["A1", "A2", "A3", "B1"]
    df = pd.DataFrame(
        {
            "master_rating": ratings * 3,
            "default": [0, 0, 0, 1] * 3,
        }
    )
    score_values = np.tile([0.1, 0.2, 0.3, 0.4], 3)
    predictions = {
        "low": np.tile([0.01, 0.03, 0.08, 0.20], 3),
        "high": np.tile([0.02, 0.04, 0.10, 0.25], 3),
    }

    scale = compare_methods_by_rating_master_scale(
        df,
        predictions,
        score_values=score_values,
        target_pd=0.10,
        rating_order=tuple(ratings),
    )
    summary = summarize_rating_scale(scale, method_col="method")
    structure_check = validate_common_rating_structure(scale)

    assert set(scale["method"]) == {"low", "high"}
    assert summary.loc["low", "target_weighted_pd"] == pytest.approx(0.10)
    assert summary.loc["high", "target_weighted_pd"] == pytest.approx(0.10)
    assert structure_check["is_common"].all()


def test_validate_common_rating_structure_rejects_method_specific_counts():
    scale = pd.DataFrame(
        {
            "method": ["m1", "m1", "m2", "m2"],
            "rating": ["A1", "A2", "A1", "A2"],
            "n_assets": [10, 20, 11, 19],
            "total_ead": [10.0, 20.0, 11.0, 19.0],
            "defaults": [0, 1, 0, 1],
            "observed_default_rate": [0.0, 0.05, 0.0, 0.05],
            "portfolio_count_share": [1 / 3, 2 / 3, 11 / 30, 19 / 30],
            "portfolio_ead_share": [1 / 3, 2 / 3, 11 / 30, 19 / 30],
        }
    )

    with pytest.raises(ValueError, match="Rating structure differs"):
        validate_common_rating_structure(scale)


def test_rating_scale_capital_uses_rating_level_ead_buckets():
    scale = pd.DataFrame(
        {
            "method": ["m1", "m1"],
            "rating": ["A1", "E"],
            "pd_rating": [0.01, 0.20],
            "total_ead": [2_000_000.0, 1_000_000.0],
        }
    )

    out = rating_scale_capital(scale, method_col="method")

    assert out.loc["m1", "total_ead"] == pytest.approx(3_000_000.0)
    assert out.loc["m1", "total_expected_loss"] > 0.0
    assert out.loc["m1", "total_rwa"] > out.loc["m1", "total_expected_loss"]
    assert out.loc["m1", "total_capital_true"] == pytest.approx(
        out.loc["m1", "total_expected_loss"] + out.loc["m1", "total_rwa"]
    )


def test_rating_scale_capital_by_rating_preserves_empty_master_scale_start():
    scale = pd.DataFrame(
        {
            "method": ["m1", "m1", "m1"],
            "rating": ["A1", "A2", "B1"],
            "n_assets": [0, 0, 3],
            "portfolio_count_share": [0.0, 0.0, 1.0],
            "observed_default_rate": [np.nan, np.nan, 1 / 3],
            "total_ead": [0.0, 0.0, 3_000_000.0],
            "pd_min": [np.nan, np.nan, 0.0011],
            "avg_pd": [np.nan, np.nan, 0.0012],
            "pd_max": [np.nan, np.nan, 0.0013],
            "pd_rating": [0.0005, 0.0007, 0.0012],
        }
    )

    by_rating = rating_scale_capital_by_rating(
        scale,
        method_col="method",
        rating_order=("A1", "A2", "B1"),
    )
    summary = rating_scale_capital(scale, method_col="method")

    assert list(by_rating["rating"].astype(str)) == ["A1", "A2", "B1"]
    assert by_rating.loc[by_rating["rating"].astype(str) == "A1", "total_rwa"].iloc[0] == 0.0
    assert by_rating["total_rwa"].sum() == pytest.approx(summary.loc["m1", "total_rwa"])
    assert by_rating["rwa_share"].fillna(0.0).sum() == pytest.approx(1.0)


def test_method_master_scale_distribution_uses_pd_boundaries_per_method():
    df = pd.DataFrame({"default": [0, 0, 1, 0], "ead": [1.0, 2.0, 3.0, 4.0]})
    predictions = {
        "low": np.array([0.0003, 0.0007, 0.0020, 0.20]),
        "high": np.array([0.0007, 0.0020, 0.20, 0.30]),
    }

    out = method_master_scale_distribution(df, predictions, ead_col="ead")

    assert set(out["method"]) == {"low", "high"}
    assert out.loc[(out["method"] == "low") & (out["rating"] == "A1"), "n_assets"].iloc[0] == 1
    assert out.loc[(out["method"] == "high") & (out["rating"] == "E"), "n_assets"].iloc[0] == 1


def test_rating_migration_matrix_counts_moves_from_baseline():
    predictions = {
        "base": np.array([0.0005, 0.0007, 0.0020]),
        "shifted": np.array([0.0007, 0.0020, 0.20]),
    }

    out = rating_migration_matrix(predictions, baseline_method="base")
    shifted = out[out["method"] == "shifted"]

    assert shifted["n_assets"].sum() == 3
    assert shifted.loc[
        (shifted["baseline_rating"] == "A1") & (shifted["method_rating"] == "A2"),
        "n_assets",
    ].iloc[0] == 1


def test_score_distribution_table_counts_defaults_by_bin():
    out = score_distribution_table(
        np.array([0.1, 0.2, 0.8, 0.9]),
        defaults=np.array([0, 1, 0, 1]),
        bins=np.array([0.0, 0.5, 1.0]),
    )

    assert out["n_assets"].sum() == 4
    assert out.loc[0, "defaults"] == pytest.approx(1.0)
    assert out.loc[1, "observed_default_rate"] == pytest.approx(0.5)


def test_master_scale_pd_bound_capital_builds_lower_avg_upper_sensitivity():
    distribution = pd.DataFrame(
        {
            "method": ["m1", "m1"],
            "rating": ["A1", "E"],
            "total_ead": [1_000_000.0, 2_000_000.0],
            "pd_lower": [0.0000, 0.26],
            "pd_avg_master": [0.0005, 0.40],
            "pd_upper": [0.0006, 1.00],
        }
    )

    out = master_scale_pd_bound_capital(distribution)

    assert ("m1", "pd_lower") in out.index
    assert ("m1", "pd_avg_master") in out.index
    assert out.loc[("m1", "pd_upper"), "total_capital_true"] > out.loc[
        ("m1", "pd_avg_master"), "total_capital_true"
    ]
