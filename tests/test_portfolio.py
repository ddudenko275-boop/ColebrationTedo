import numpy as np
import pandas as pd
import pytest

from data.generate_data import generate_credit_data, get_oot_split
from src.portfolio import (
    MASTER_SCALE_RATINGS,
    assign_master_scale_ratings,
    calibrate_pd_to_target,
    compare_methods_by_rating_master_scale,
    compare_methods_by_historical_panel,
    historical_portfolio_panel,
    method_portfolio_summary,
    portfolio_average_pd,
    rating_master_scale,
    rating_scale_capital,
    summarize_rating_scale,
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


def test_calibrate_pd_to_target_matches_weighted_average():
    pd_values = np.array([0.01, 0.03, 0.10, 0.30])
    weights = np.array([100.0, 200.0, 300.0, 400.0])

    calibrated = calibrate_pd_to_target(pd_values, weights, target_pd=0.12)

    assert np.average(calibrated, weights=weights) == pytest.approx(0.12)
    assert np.all(np.diff(calibrated) > 0)


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

    assert set(scale["method"]) == {"low", "high"}
    assert summary.loc["low", "target_weighted_pd"] == pytest.approx(0.10)
    assert summary.loc["high", "target_weighted_pd"] == pytest.approx(0.10)


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
