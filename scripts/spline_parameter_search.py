"""Search spline calibration parameters while preserving fit central tendency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_data import generate_credit_data, get_oot_split, get_rating_portfolio_config
from src.calibrators import FrenchSplineCalibrator, LogitCalibrator, MonotoneSplineCalibrator
from src.capital import IRBAssumptions, compare_irb_capital_by_method
from src.metrics import summary_metrics


RANDOM_STATE = 42
TARGET_RWA_DELTA_VS_LOGIT = -0.01


def _quantile_reliability_table(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "pd": np.asarray(probs, dtype=float)})
    frame["bin"] = pd.qcut(frame["pd"], q=n_bins, duplicates="drop")
    out = (
        frame.groupby("bin", observed=True)
        .agg(
            n_assets=("y", "size"),
            defaults=("y", "sum"),
            mean_pred=("pd", "mean"),
            observed_default_rate=("y", "mean"),
        )
        .reset_index(drop=True)
    )
    return out


def _binomial_summary(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    table = _quantile_reliability_table(y_true, probs, n_bins=n_bins)
    n = table["n_assets"].to_numpy(dtype=float)
    defaults = table["defaults"].to_numpy(dtype=float)
    p = np.clip(table["mean_pred"].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
    observed = table["observed_default_rate"].to_numpy(dtype=float)
    se = np.sqrt(p * (1.0 - p) / n)
    z = np.divide(observed - p, se, out=np.full_like(p, np.nan), where=se > 0)
    p_values = np.array(
        [binomtest(int(round(k)), int(round(n_i)), float(p_i)).pvalue for k, n_i, p_i in zip(defaults, n, p)]
    )
    return {
        "n_binomial_bins": float(len(table)),
        "significant_bins_5pct": float(np.sum(p_values < 0.05)),
        "max_abs_z": float(np.nanmax(np.abs(z))),
        "min_binomial_p_value": float(np.nanmin(p_values)),
        "mean_abs_dr_minus_pd": float(np.nanmean(np.abs(observed - p))),
    }


def _evaluate_variant(
    name: str,
    calibrator,
    scores_calib: np.ndarray,
    y_calib: np.ndarray,
    scores_test: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, float],
    logit_pred_test: np.ndarray,
    logit_rwa: float,
    capital_assumptions: IRBAssumptions,
) -> dict[str, float | str]:
    calibrator.fit(scores_calib, y_calib)
    pred_fit = calibrator.predict(scores_calib)
    pred_test = calibrator.predict(scores_test)
    oot_metrics = summary_metrics(y_test, pred_test, name=name)
    binom = _binomial_summary(y_test, pred_test)
    diagnostics = calibrator.knot_diagnostics()
    variant_rwa = float(
        compare_irb_capital_by_method(
            {"logit": logit_pred_test, name: pred_test},
            assumptions=capital_assumptions,
            baseline_method="logit",
        ).loc[name, "total_rwa"]
    )
    rwa_delta_vs_logit = (logit_rwa - variant_rwa) / logit_rwa

    row: dict[str, float | str] = {
        "method": name,
        **params,
        "fit_ct": float(np.mean(y_calib)),
        "fit_mean_pd": float(np.mean(pred_fit)),
        "fit_ct_gap_abs": float(abs(np.mean(pred_fit) - np.mean(y_calib))),
        "oot_default_rate": float(np.mean(y_test)),
        "oot_mean_pd": float(np.mean(pred_test)),
        "oot_default_gap": float(np.sum(y_test) - np.sum(pred_test)),
        "oot_abs_default_gap": float(abs(np.sum(y_test) - np.sum(pred_test))),
        "brier_score": float(oot_metrics["brier_score"]),
        "log_loss": float(oot_metrics["log_loss"]),
        "ece_quantile": float(oot_metrics["ece_quantile"]),
        "cal_slope": float(oot_metrics["cal_slope"]),
        "cal_intercept": float(oot_metrics["cal_intercept"]),
        "rwa_bln": float(variant_rwa / 1_000_000_000),
        "rwa_delta_vs_logit_pct": float(rwa_delta_vs_logit),
        "rwa_abs_delta_to_minus_1pct": float(abs(rwa_delta_vs_logit - TARGET_RWA_DELTA_VS_LOGIT)),
        "candidate_switch_knots": float(diagnostics["candidate_switch_knot"].sum()),
    }
    row.update(binom)
    return row


def run_search() -> pd.DataFrame:
    portfolio_config = get_rating_portfolio_config("stress")
    df = generate_credit_data(random_state=RANDOM_STATE, portfolio=portfolio_config.name)
    x_train, x_calib, x_test, y_train, y_calib, y_test = get_oot_split(df)

    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    base_model.fit(x_train, y_train)
    scores_calib = np.clip(base_model.predict_proba(x_calib)[:, 1], 1e-6, 1.0 - 1e-6)
    scores_test = np.clip(base_model.predict_proba(x_test)[:, 1], 1e-6, 1.0 - 1e-6)

    y_calib_arr = y_calib.to_numpy(dtype=float)
    y_test_arr = y_test.to_numpy(dtype=float)
    logit = LogitCalibrator().fit(scores_calib, y_calib_arr)
    logit_pred_test = np.clip(logit.predict(scores_test), 1e-6, 1.0 - 1e-6)
    capital_assumptions = IRBAssumptions(lgd=0.40, maturity_years=2.5, ead=1_000_000.0)
    logit_rwa = float(
        compare_irb_capital_by_method(
            {"logit": logit_pred_test},
            assumptions=capital_assumptions,
        ).loc["logit", "total_rwa"]
    )
    rows = []
    n_bins_grid = [10, 15, 20, 25, 30, 35, 40, 50, 80]
    alpha_grid = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0]

    for n_bins in n_bins_grid:
        for alpha in alpha_grid:
            rows.append(
                _evaluate_variant(
                    "Монотонный сплайн",
                    MonotoneSplineCalibrator(n_bins=n_bins, alpha=alpha),
                    scores_calib,
                    y_calib_arr,
                    scores_test,
                    y_test_arr,
                    {"n_bins": float(n_bins), "alpha": alpha, "shrinkage": np.nan},
                    logit_pred_test,
                    logit_rwa,
                    capital_assumptions,
                )
            )
            for shrinkage in [0.15, 0.25, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]:
                rows.append(
                    _evaluate_variant(
                        "Французский сплайн",
                        FrenchSplineCalibrator(n_bins=n_bins, alpha=alpha, shrinkage=shrinkage),
                        scores_calib,
                        y_calib_arr,
                        scores_test,
                        y_test_arr,
                        {"n_bins": float(n_bins), "alpha": alpha, "shrinkage": shrinkage},
                        logit_pred_test,
                        logit_rwa,
                        capital_assumptions,
                    )
                )

    results = pd.DataFrame(rows)
    return results.sort_values(
        [
            "rwa_abs_delta_to_minus_1pct",
            "significant_bins_5pct",
            "brier_score",
            "oot_abs_default_gap",
        ],
        ascending=[True, True, True, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/spline_parameter_search.csv"))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    results = run_search()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False, encoding="utf-8-sig")

    display_cols = [
        "method",
        "n_bins",
        "alpha",
        "shrinkage",
        "fit_ct_gap_abs",
        "oot_mean_pd",
        "oot_abs_default_gap",
        "brier_score",
        "log_loss",
        "ece_quantile",
        "rwa_bln",
        "rwa_delta_vs_logit_pct",
        "rwa_abs_delta_to_minus_1pct",
        "significant_bins_5pct",
        "min_binomial_p_value",
        "mean_abs_dr_minus_pd",
        "candidate_switch_knots",
    ]
    print(f"Saved full search to {args.out}")
    print(results[display_cols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
