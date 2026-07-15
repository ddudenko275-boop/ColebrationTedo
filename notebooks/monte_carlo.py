"""Portfolio-resampling stability check for PD calibration methods.

RF and all calibrators are fitted once on the full in-time sample (no
retraining per scenario). Each Monte Carlo scenario independently resamples a
fraction of the in-time and OOT populations without replacement and scores
them through the fixed pipeline:

    score -> calibrated PD -> EL / UL / Capital / RWA

This isolates sensitivity to portfolio composition from sensitivity to model
fitting. See docs/calibration_change_report.md for the underlying calibrator
fix this check is meant to stress-test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_data import generate_credit_data, get_oot_split
from src.calibrators import get_all_calibrators
from src.capital import IRBAssumptions, calculate_irb_capital
from src.portfolio import (
    MASTER_SCALE_RATINGS,
    assign_pd_master_scale_ratings,
    master_scale_bounds_table,
)

RANDOM_STATE = 42
DEFAULT_CAPITAL_ASSUMPTIONS = IRBAssumptions(lgd=0.40, maturity_years=2.5, ead=1_000_000.0)
METRIC_COLUMNS = (
    "p_value_intime",
    "p_value_oot",
    "oot_mean_pd",
    "expected_loss",
    "unexpected_loss_capital",
    "el_plus_ul",
    "rwa",
)
# The cross-method dispersion table (see cross_method_spread) is built on this
# metric: total economic capital as EL + UL. Note that the old required_capital
# column is identically equal to unexpected_loss_capital (required_capital =
# capital_ratio * rwa = capital_ratio * UL / capital_ratio = UL), so it is
# dropped in favour of EL + UL, which is a genuinely different, richer figure.
SPREAD_METRIC = "el_plus_ul"
# Per-row IRB capital columns that are additive over the portfolio, so a
# subsample total is just the sum of the sampled rows. This is what lets a
# 1000-scenario run reuse one precomputed per-row table instead of recomputing
# the IRB formula in every scenario. el_plus_ul is likewise additive (it is a
# per-row sum of two additive columns), so it is precomputed the same way.
_CAPITAL_ROW_COLUMNS = {
    "expected_loss": "expected_loss",
    "unexpected_loss_capital": "unexpected_loss_capital",
    "rwa": "rwa",
}


def _master_scale_representative_pd() -> np.ndarray:
    """Representative PD per fixed master-scale grade, in MASTER_SCALE_RATINGS order.

    Matches section 8.4 of the calibration notebook, where the binomial test's
    expected PD per grade is pd_rating = pd_avg_master -- the mentor's fixed
    representative PD of the bucket -- rather than the mean of the calibrated PDs.
    """

    scale = master_scale_bounds_table()
    return scale["pd_avg_master"].to_numpy(dtype=float)


def _master_scale_grade_codes(probs: np.ndarray) -> np.ndarray:
    """Fixed A1...E grade index (0..12) for each calibrated PD.

    Grade assignment (assign_pd_master_scale_ratings) depends only on the PD
    value through the fixed mentor bounds, so a row's grade is identical in
    every scenario and can be precomputed once, exactly like its capital.
    """

    ratings = assign_pd_master_scale_ratings(probs)
    return np.asarray(ratings.codes, dtype=int)


def master_scale_min_binomial_p_value(
    y_true: np.ndarray,
    grade_codes: np.ndarray,
    representative_pd: np.ndarray,
) -> float:
    """Smallest per-grade exact binomial p-value on the fixed A1...E master scale.

    Rows are bucketed into fixed master-scale grades and each non-empty grade's
    observed default count is tested against that grade's representative PD.
    This is the same test as the master-scale binomial calibration check in
    section 8.4 of the calibration notebook (min p-value over non-empty
    buckets), so a scenario's p-value is on the same footing as the
    single-portfolio number reported there -- and, importantly, the test runs
    AFTER rating assignment, not on dynamic quantile bins of raw PD.
    """

    n_grades = len(representative_pd)
    counts = np.bincount(grade_codes, minlength=n_grades).astype(float)
    defaults = np.bincount(
        grade_codes, weights=np.asarray(y_true, dtype=float), minlength=n_grades
    )
    p_values = [
        binomtest(int(round(defaults[g])), int(round(counts[g])), float(representative_pd[g])).pvalue
        for g in range(n_grades)
        if counts[g] > 0
    ]
    return float(np.min(p_values)) if p_values else float("nan")


def prepare_fixed_pipeline(
    portfolio: str = "stress",
    random_state: int = RANDOM_STATE,
    capital_assumptions: IRBAssumptions | None = None,
) -> dict:
    """Fit RF and all calibrators once and precompute per-row PD and capital.

    Because RF and every calibrator are fixed and score each borrower row
    independently, a row's calibrated PD -- and its additive IRB capital
    contribution -- is identical in every scenario. So they are computed once
    here on the full in-time and OOT samples; a scenario then only indexes the
    rows its random ~80% subsample selected. Nothing is refit or re-scored
    inside a scenario. The scenario subsamples are still drawn independently
    per seed, so each scenario sees a different 80% of the same base.
    """

    assumptions = capital_assumptions or DEFAULT_CAPITAL_ASSUMPTIONS
    df = generate_credit_data(random_state=random_state, portfolio=portfolio)
    x_train, x_calib, x_test, y_train, y_calib, y_test = get_oot_split(df)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=20,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)

    scores_calib_full = np.clip(rf.predict_proba(x_calib)[:, 1], 1e-6, 1 - 1e-6)
    scores_test_full = np.clip(rf.predict_proba(x_test)[:, 1], 1e-6, 1 - 1e-6)

    representative_pd = _master_scale_representative_pd()

    calibrators = get_all_calibrators()
    pred_calib_full: dict[str, np.ndarray] = {}
    pred_test_full: dict[str, np.ndarray] = {}
    capital_rows: dict[str, dict[str, np.ndarray]] = {}
    grade_calib_full: dict[str, np.ndarray] = {}
    grade_test_full: dict[str, np.ndarray] = {}
    for method, calibrator in calibrators.items():
        calibrator.fit(scores_calib_full, y_calib.to_numpy(dtype=float))
        pred_calib = np.asarray(calibrator.predict(scores_calib_full), dtype=float)
        pred_test = np.asarray(calibrator.predict(scores_test_full), dtype=float)
        pred_calib_full[method] = pred_calib
        pred_test_full[method] = pred_test
        grade_calib_full[method] = _master_scale_grade_codes(pred_calib)
        grade_test_full[method] = _master_scale_grade_codes(pred_test)

        row_capital = calculate_irb_capital(pred_test, assumptions=assumptions)
        rows = {
            metric: row_capital[col].to_numpy(dtype=float)
            for metric, col in _CAPITAL_ROW_COLUMNS.items()
        }
        # Total economic capital per row, additive over the portfolio.
        rows["el_plus_ul"] = rows["expected_loss"] + rows["unexpected_loss_capital"]
        capital_rows[method] = rows

    return {
        "methods": list(calibrators),
        "y_calib": y_calib.to_numpy(dtype=float),
        "y_test": y_test.to_numpy(dtype=float),
        "pred_calib_full": pred_calib_full,
        "pred_test_full": pred_test_full,
        "capital_rows": capital_rows,
        "grade_calib_full": grade_calib_full,
        "grade_test_full": grade_test_full,
        "representative_pd": representative_pd,
    }


def run_scenario(
    pipeline: dict,
    scenario_seed: int,
    sample_frac: float = 0.8,
) -> pd.DataFrame:
    """Score one resampled scenario by indexing the precomputed per-row values.

    In-time and OOT rows are resampled independently (without replacement) with
    a seed unique to this scenario, so every scenario is a different ~80% of the
    same fixed base, and p_value_intime / p_value_oot each reflect sampling
    variability in their own population.
    """

    rng = np.random.default_rng(scenario_seed)

    y_calib, y_test = pipeline["y_calib"], pipeline["y_test"]
    idx_calib = rng.choice(len(y_calib), size=int(round(len(y_calib) * sample_frac)), replace=False)
    idx_test = rng.choice(len(y_test), size=int(round(len(y_test) * sample_frac)), replace=False)

    yc, yt = y_calib[idx_calib], y_test[idx_test]
    representative_pd = pipeline["representative_pd"]

    rows = []
    for method in pipeline["methods"]:
        pred_test = pipeline["pred_test_full"][method][idx_test]
        grade_calib = pipeline["grade_calib_full"][method][idx_calib]
        grade_test = pipeline["grade_test_full"][method][idx_test]
        capital = pipeline["capital_rows"][method]

        rows.append(
            {
                "scenario": scenario_seed,
                "method": method,
                "p_value_intime": master_scale_min_binomial_p_value(yc, grade_calib, representative_pd),
                "p_value_oot": master_scale_min_binomial_p_value(yt, grade_test, representative_pd),
                "oot_mean_pd": float(np.mean(pred_test)),
                "expected_loss": float(capital["expected_loss"][idx_test].sum()),
                "unexpected_loss_capital": float(capital["unexpected_loss_capital"][idx_test].sum()),
                "el_plus_ul": float(capital["el_plus_ul"][idx_test].sum()),
                "rwa": float(capital["rwa"][idx_test].sum()),
            }
        )
    return pd.DataFrame(rows)


def run_monte_carlo(
    n_scenarios: int = 1000,
    sample_frac: float = 0.8,
    portfolio: str = "stress",
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run n_scenarios resampling scenarios; one row per (scenario, method).

    Each scenario draws its own independent ~sample_frac subsample from the
    same fixed base, so results reflect sensitivity to portfolio composition.
    Per-row PD and capital are precomputed in prepare_fixed_pipeline, so the
    1000-scenario run stays fast.
    """

    pipeline = prepare_fixed_pipeline(portfolio=portfolio, random_state=random_state)
    frames = [
        run_scenario(pipeline, scenario_seed=i, sample_frac=sample_frac)
        for i in range(n_scenarios)
    ]
    return pd.concat(frames, ignore_index=True)


def summarize_monte_carlo(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scenario-level results into mean / min / max / range per method."""

    agg = results.groupby("method")[list(METRIC_COLUMNS)].agg(["mean", "min", "max"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    for metric in METRIC_COLUMNS:
        agg[f"{metric}_range"] = agg[f"{metric}_max"] - agg[f"{metric}_min"]
    return agg.reset_index()


def cross_method_spread(results: pd.DataFrame, metric: str = SPREAD_METRIC) -> pd.DataFrame:
    """Per-scenario dispersion of a metric ACROSS methods (not across scenarios).

    summarize_monte_carlo measures how one method moves over 1000 scenarios.
    This is the orthogonal cut the analysis asks for: within a single scenario
    (one fixed ~80% subsample, all 5 methods scoring the same borrowers), how
    far apart are the methods? For each scenario it reports the cheapest and the
    most expensive method by ``metric`` (EL + UL by default), the absolute
    difference (max - min) and the relative difference expressed against the
    minimum method -- i.e. how much more capital the most conservative method
    demands than the leanest one.

    One row per scenario (the transposed orientation), so the table stays a
    tidy 1000-row frame that sorts and exports cleanly to CSV.
    """

    grouped = results.groupby("scenario")[["method", metric]]
    rows = []
    for scenario, frame in grouped:
        values = frame.set_index("method")[metric]
        min_method = values.idxmin()
        max_method = values.idxmax()
        min_value = float(values.loc[min_method])
        max_value = float(values.loc[max_method])
        abs_diff = max_value - min_value
        rel_diff = abs_diff / min_value if min_value != 0.0 else float("nan")
        rows.append(
            {
                "scenario": int(scenario),
                "min_method": min_method,
                "max_method": max_method,
                f"min_{metric}": min_value,
                f"max_{metric}": max_value,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            }
        )
    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)


def summarize_cross_method_spread(spread: pd.DataFrame, metric: str = SPREAD_METRIC) -> pd.DataFrame:
    """Compact summary of the per-scenario cross-method spread over all scenarios.

    Reports mean / median / min / max of both the absolute and the relative
    difference, plus how often each method is the cheapest and the most
    expensive across scenarios, so the 1000-row table has a one-glance takeaway.
    """

    stat_rows = []
    for label, col in (("abs_diff", "abs_diff"), ("rel_diff", "rel_diff")):
        series = spread[col]
        stat_rows.append(
            {
                "quantity": label,
                "mean": float(series.mean()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    stats = pd.DataFrame(stat_rows)

    counts = pd.DataFrame(
        {
            "times_cheapest": spread["min_method"].value_counts(),
            "times_most_expensive": spread["max_method"].value_counts(),
        }
    ).fillna(0.0).astype(int)
    counts.index.name = "method"
    return stats, counts.reset_index()


def print_monte_carlo_report(results: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Print scenario-level rows and a per-metric summary in readable blocks.

    One flat 29-column table is unreadable in a terminal; instead the report
    shows every scenario row first, then one compact mean/min/max/range block
    per metric.
    """

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    fmt_pct = "{:.4%}".format
    fmt_bln = lambda v: f"{v / 1e9:,.4f}"
    metric_formats = {
        "p_value_intime": "{:.6f}".format,
        "p_value_oot": "{:.6f}".format,
        "oot_mean_pd": fmt_pct,
        "expected_loss": fmt_bln,
        "unexpected_loss_capital": fmt_bln,
        "el_plus_ul": fmt_bln,
        "rwa": fmt_bln,
    }
    metric_titles = {
        "p_value_intime": "Min binomial p-value (master scale A1...E), IN-TIME",
        "p_value_oot": "Min binomial p-value (master scale A1...E), OOT",
        "oot_mean_pd": "Средний PD на OOT",
        "expected_loss": "Expected Loss, млрд",
        "unexpected_loss_capital": "UL capital, млрд",
        "el_plus_ul": "EL + UL (экономический капитал), млрд",
        "rwa": "RWA, млрд",
    }

    n_scenarios = results["scenario"].nunique()
    print("=" * 100)
    print(f"MONTE CARLO: {n_scenarios} сценариев x {results['method'].nunique()} методов; "
          "RF и калибраторы зафиксированы, ресемплится только состав портфеля")
    print("=" * 100)

    print("\n--- Сценарный уровень: все строки ---")
    scenario_view = results.copy()
    for col, fmt in metric_formats.items():
        scenario_view[col] = scenario_view[col].map(fmt)
    print(scenario_view.to_string(index=False))

    print("\n--- Сводка по методам: mean / min / max / range на метрику ---")
    for metric in METRIC_COLUMNS:
        fmt = metric_formats[metric]
        block = summary[["method"] + [f"{metric}_{s}" for s in ("mean", "min", "max", "range")]].copy()
        block.columns = ["method", "mean", "min", "max", "range (max-min)"]
        for col in ("mean", "min", "max", "range (max-min)"):
            block[col] = block[col].map(fmt)
        print(f"\n{metric_titles[metric]}:")
        print(block.to_string(index=False))

    spread = cross_method_spread(results)
    spread_stats, spread_counts = summarize_cross_method_spread(spread)
    print("\n--- Межметодный разброс EL + UL внутри сценария (по всем сценариям) ---")
    print("На каждый сценарий: самый дешёвый и самый дорогой по EL+UL метод, abs = max-min,")
    print("rel = (max-min) / min (относительно минимального метода).")
    stat_cols = ("mean", "median", "min", "max")
    row_fmt = {"abs_diff": fmt_bln, "rel_diff": "{:.4%}".format}
    stats_view = spread_stats.copy()
    for col in stat_cols:
        stats_view[col] = [
            row_fmt[q](v) for q, v in zip(spread_stats["quantity"], spread_stats[col])
        ]
    stats_view["quantity"] = stats_view["quantity"].map(
        {"abs_diff": "abs (max-min), млрд", "rel_diff": "rel (max-min)/min"}
    )
    print(stats_view.to_string(index=False))
    print("\nКак часто метод оказывался самым дешёвым / самым дорогим по капиталу:")
    print(spread_counts.to_string(index=False))

    print("\nКак читать: узкий range у oot_mean_pd и денежных метрик = результат устойчив к составу")
    print("портфеля. Широкий range у p-value ожидаем (min binomial p-value чувствителен к случайному")
    print("числу дефолтов в отдельном грейде мастер-шкалы); структурный сигнал — это p_value_oot_max")
    print("около нуля, то есть тест значим на ВСЕХ сценариях, а не в среднем. Межметодный rel показывает,")
    print("на сколько процентов самый консервативный метод дороже самого экономного при том же портфеле.")


if __name__ == "__main__":
    scenario_results = run_monte_carlo(n_scenarios=10)
    mc_summary = summarize_monte_carlo(scenario_results)
    print_monte_carlo_report(scenario_results, mc_summary)
