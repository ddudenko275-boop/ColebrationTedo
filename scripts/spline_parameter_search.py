"""Select spline calibration parameters from data-driven criteria, not a target outcome.

Historically this script picked n_bins/alpha/shrinkage by grid-searching for
whichever combination hit a target RWA delta vs logit (~-1%). Poirier (1981),
citing Wold, warns explicitly against exactly this: knot/smoothing choices
should follow the data (or a formal significance test), not be reverse
engineered from a desired economic answer -- that is "fishing" (see
docs/spline_methodology.md, §3.7 and §4, point 4).

This version instead:
  1. Picks n_bins from a Wold-style data floor: the largest n_bins such that
     the average bin still has at least MIN_EXPECTED_DEFAULTS_PER_BIN expected
     defaults, so bin-level default rates stay statistically stable and
     isotonic regression is not smoothing pure noise.
  2. Picks the French spline's shrinkage by minimizing OOT Brier score (a
     statistical fit criterion) at that n_bins, not by RWA.
  3. Reports capital impact (RWA/EL/UL) only as a downstream *observation*
     for the chosen parameters, never as the selection criterion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_data import generate_credit_data, get_oot_split, get_rating_portfolio_config
from src.calibrators import FrenchSplineCalibrator, LogitCalibrator, MonotoneSplineCalibrator
from src.capital import IRBAssumptions, compare_irb_capital_by_method
from src.metrics import summary_metrics
from src.score_model import fit_oof_score_model


RANDOM_STATE = 42
MIN_EXPECTED_DEFAULTS_PER_BIN = 20.0
N_BINS_CANDIDATES = (10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 100, 125, 150)
SHRINKAGE_GRID = (0.15, 0.25, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)


def select_n_bins_by_data_floor(
    n_obs: int,
    base_rate: float,
    candidates: tuple[int, ...] = N_BINS_CANDIDATES,
    min_expected_defaults_per_bin: float = MIN_EXPECTED_DEFAULTS_PER_BIN,
) -> pd.DataFrame:
    """Rank n_bins candidates by a Wold-style "enough observations per interval" floor.

    Wold's rule (as cited by Poirier) asks for a handful of observations per
    interval; for a rare binary outcome the informative quantity is expected
    *defaults* per bin, not raw row count, since that is what actually drives
    the variance of the bin-level default rate isotonic regression consumes.
    The recommended n_bins is the largest candidate that still clears the
    floor -- as many bins as the data can support without smoothing noise.
    """

    rows = []
    for n_bins in candidates:
        avg_n_per_bin = n_obs / n_bins
        expected_defaults_per_bin = avg_n_per_bin * base_rate
        rows.append(
            {
                "n_bins": n_bins,
                "avg_n_per_bin": avg_n_per_bin,
                "expected_defaults_per_bin": expected_defaults_per_bin,
                "clears_floor": expected_defaults_per_bin >= min_expected_defaults_per_bin,
            }
        )
    table = pd.DataFrame(rows)
    recommended = table.loc[table["clears_floor"], "n_bins"]
    table["recommended"] = False
    if len(recommended) > 0:
        table.loc[table["n_bins"] == recommended.max(), "recommended"] = True
    else:
        table.loc[table["expected_defaults_per_bin"].idxmax(), "recommended"] = True
    return table


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
        "candidate_switch_knots": float(diagnostics["candidate_switch_knot"].sum()),
        # summary_metrics rounds brier_score to 5 decimals for display, which on
        # this portfolio collapses every shrinkage to the same 0.02578 -- an
        # idxmin over that column silently returns whichever value happens to be
        # first in the grid, not the best one. Keep an unrounded copy for the
        # selection step.
        "brier_score_full": float(brier_score_loss(y_test, np.clip(pred_test, 1e-7, 1.0 - 1e-7))),
    }
    row.update(binom)
    return row


def run_search() -> tuple[pd.DataFrame, pd.DataFrame, int, float]:
    portfolio_config = get_rating_portfolio_config("stress")
    df = generate_credit_data(random_state=RANDOM_STATE, portfolio=portfolio_config.name)
    x_train, x_calib, x_test, y_train, y_calib, y_test = get_oot_split(df)

    _, scores_calib, scores_test, _ = fit_oof_score_model(
        x_train,
        y_train,
        x_test,
        random_state=RANDOM_STATE,
    )

    y_calib_arr = y_calib.to_numpy(dtype=float)
    y_test_arr = y_test.to_numpy(dtype=float)

    # Step 1: n_bins from the data floor (§ module docstring), not from RWA.
    n_bins_table = select_n_bins_by_data_floor(
        n_obs=len(y_calib_arr),
        base_rate=float(y_calib_arr.mean()),
    )
    recommended_n_bins = int(n_bins_table.loc[n_bins_table["recommended"], "n_bins"].iloc[0])

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
    rows.append(
        _evaluate_variant(
            "Монотонный сплайн",
            MonotoneSplineCalibrator(n_bins=recommended_n_bins),
            scores_calib,
            y_calib_arr,
            scores_test,
            y_test_arr,
            {"n_bins": float(recommended_n_bins), "shrinkage": np.nan},
            logit_pred_test,
            logit_rwa,
            capital_assumptions,
        )
    )

    # Step 2: shrinkage chosen by minimizing OOT Brier at the recommended n_bins
    # (a statistical fit criterion), RWA is reported for the winner only as an
    # observation, not used to pick among candidates.
    for shrinkage in SHRINKAGE_GRID:
        rows.append(
            _evaluate_variant(
                "Французский сплайн",
                FrenchSplineCalibrator(n_bins=recommended_n_bins, shrinkage=shrinkage),
                scores_calib,
                y_calib_arr,
                scores_test,
                y_test_arr,
                {"n_bins": float(recommended_n_bins), "shrinkage": shrinkage},
                logit_pred_test,
                logit_rwa,
                capital_assumptions,
            )
        )

    results = pd.DataFrame(rows)
    french_rows = results[results["method"] == "Французский сплайн"]
    recommended_shrinkage = float(
        french_rows.loc[french_rows["brier_score_full"].idxmin(), "shrinkage"]
    )

    results = results.sort_values(["method", "brier_score_full"])
    return results, n_bins_table, recommended_n_bins, recommended_shrinkage, y_test_arr


def shrinkage_selection_resolution(french_rows: pd.DataFrame, y_test: np.ndarray) -> dict:
    """Can the Brier criterion actually tell the shrinkage candidates apart?

    Compares the spread of OOT Brier across the grid against the standard error
    of the Brier score itself on this sample. If the spread is far below that
    error, every candidate is statistically indistinguishable and the "winner"
    is noise -- worth stating outright rather than reporting a false precision.
    """

    brier = french_rows["brier_score_full"].to_numpy(dtype=float)
    spread = float(brier.max() - brier.min())
    se = float(np.std(y_test, ddof=1) / np.sqrt(len(y_test)))
    return {
        "brier_spread": spread,
        "brier_se": se,
        "spread_over_se": spread / se if se > 0 else float("nan"),
        "criterion_discriminates": spread > se,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/spline_parameter_search.csv"))
    args = parser.parse_args()

    results, n_bins_table, recommended_n_bins, recommended_shrinkage, y_test_arr = run_search()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("Шаг 1: выбор n_bins по data floor (не по целевому RWA)")
    print(
        n_bins_table.to_string(
            index=False,
            formatters={
                "avg_n_per_bin": "{:.1f}".format,
                "expected_defaults_per_bin": "{:.1f}".format,
            },
        )
    )
    print(f"\nВыбрано n_bins = {recommended_n_bins} (наибольший n_bins, где expected_defaults_per_bin >= {MIN_EXPECTED_DEFAULTS_PER_BIN:.0f})")

    print(f"\nШаг 2: shrinkage французского сплайна выбран по минимальному OOT Brier = {recommended_shrinkage}")

    res = shrinkage_selection_resolution(results[results["method"] == "Французский сплайн"], y_test_arr)
    print(f"  Разброс Brier по сетке shrinkage: {res['brier_spread']:.3e}")
    print(f"  Стандартная ошибка самого Brier:  {res['brier_se']:.3e}")
    if not res["criterion_discriminates"]:
        print(f"  ВНИМАНИЕ: разброс в {1 / res['spread_over_se']:.0f} раз меньше собственной ошибки метрики —")
        print("  критерий НЕ различает кандидатов, выбор статистически произволен. Любое значение")
        print("  shrinkage на этой сетке эквивалентно; дефолт взят как формальный argmin.")

    display_cols = [
        "method",
        "n_bins",
        "shrinkage",
        "fit_ct_gap_abs",
        "oot_mean_pd",
        "oot_abs_default_gap",
        "brier_score",
        "log_loss",
        "ece_quantile",
        "rwa_bln",
        "rwa_delta_vs_logit_pct",
        "significant_bins_5pct",
        "min_binomial_p_value",
        "candidate_switch_knots",
    ]
    print(f"\nSaved full results to {args.out}")
    print("\nRWA/Brier/etc ниже — это результат подобранных по данным параметров, не критерий отбора:")
    print(results[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
