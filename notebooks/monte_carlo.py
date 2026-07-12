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

RANDOM_STATE = 42
DEFAULT_CAPITAL_ASSUMPTIONS = IRBAssumptions(lgd=0.40, maturity_years=2.5, ead=1_000_000.0)
METRIC_COLUMNS = (
    "p_value_intime",
    "p_value_oot",
    "oot_mean_pd",
    "expected_loss",
    "unexpected_loss_capital",
    "rwa",
    "required_capital",
)
# Per-row IRB capital columns that are additive over the portfolio, so a
# subsample total is just the sum of the sampled rows. This is what lets a
# 1000-scenario run reuse one precomputed per-row table instead of recomputing
# the IRB formula in every scenario.
_CAPITAL_ROW_COLUMNS = {
    "expected_loss": "expected_loss",
    "unexpected_loss_capital": "unexpected_loss_capital",
    "rwa": "rwa",
    "required_capital": "required_capital",
}


def _quantile_reliability_table(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "pd": np.asarray(probs, dtype=float)})
    frame["bin"] = pd.qcut(frame["pd"], q=n_bins, duplicates="drop")
    return (
        frame.groupby("bin", observed=True)
        .agg(
            n_assets=("y", "size"),
            defaults=("y", "sum"),
            mean_pred=("pd", "mean"),
        )
        .reset_index(drop=True)
    )


def min_binomial_p_value(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Smallest per-bin exact binomial p-value across quantile reliability bins.

    Same construction as scripts/spline_parameter_search.py and the notebook's
    binomial calibration test, so scenario results stay comparable to the
    single-portfolio numbers already reported.
    """

    table = _quantile_reliability_table(y_true, probs, n_bins=n_bins)
    n = table["n_assets"].to_numpy(dtype=float)
    defaults = table["defaults"].to_numpy(dtype=float)
    p = np.clip(table["mean_pred"].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
    p_values = [
        binomtest(int(round(k)), int(round(n_i)), float(p_i)).pvalue
        for k, n_i, p_i in zip(defaults, n, p)
    ]
    return float(np.min(p_values))


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

    calibrators = get_all_calibrators()
    pred_calib_full: dict[str, np.ndarray] = {}
    pred_test_full: dict[str, np.ndarray] = {}
    capital_rows: dict[str, dict[str, np.ndarray]] = {}
    for method, calibrator in calibrators.items():
        calibrator.fit(scores_calib_full, y_calib.to_numpy(dtype=float))
        pred_calib_full[method] = np.asarray(calibrator.predict(scores_calib_full), dtype=float)
        pred_test = np.asarray(calibrator.predict(scores_test_full), dtype=float)
        pred_test_full[method] = pred_test
        row_capital = calculate_irb_capital(pred_test, assumptions=assumptions)
        capital_rows[method] = {
            metric: row_capital[col].to_numpy(dtype=float)
            for metric, col in _CAPITAL_ROW_COLUMNS.items()
        }

    return {
        "methods": list(calibrators),
        "y_calib": y_calib.to_numpy(dtype=float),
        "y_test": y_test.to_numpy(dtype=float),
        "pred_calib_full": pred_calib_full,
        "pred_test_full": pred_test_full,
        "capital_rows": capital_rows,
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

    rows = []
    for method in pipeline["methods"]:
        pred_calib = pipeline["pred_calib_full"][method][idx_calib]
        pred_test = pipeline["pred_test_full"][method][idx_test]
        capital = pipeline["capital_rows"][method]

        rows.append(
            {
                "scenario": scenario_seed,
                "method": method,
                "p_value_intime": min_binomial_p_value(yc, pred_calib),
                "p_value_oot": min_binomial_p_value(yt, pred_test),
                "oot_mean_pd": float(np.mean(pred_test)),
                "expected_loss": float(capital["expected_loss"][idx_test].sum()),
                "unexpected_loss_capital": float(capital["unexpected_loss_capital"][idx_test].sum()),
                "rwa": float(capital["rwa"][idx_test].sum()),
                "required_capital": float(capital["required_capital"][idx_test].sum()),
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
        "rwa": fmt_bln,
        "required_capital": fmt_bln,
    }
    metric_titles = {
        "p_value_intime": "Min binomial p-value, IN-TIME (доля единицы)",
        "p_value_oot": "Min binomial p-value, OOT (доля единицы)",
        "oot_mean_pd": "Средний PD на OOT",
        "expected_loss": "Expected Loss, млрд",
        "unexpected_loss_capital": "UL capital, млрд",
        "rwa": "RWA, млрд",
        "required_capital": "Required capital, млрд",
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

    print("\nКак читать: узкий range у oot_mean_pd и денежных метрик = результат устойчив к составу")
    print("портфеля. Широкий range у p-value ожидаем (min binomial p-value чувствителен к случайному")
    print("числу дефолтов в отдельном бине); структурный сигнал — это p_value_oot_max около нуля,")
    print("то есть тест значим на ВСЕХ сценариях, а не в среднем.")


if __name__ == "__main__":
    scenario_results = run_monte_carlo(n_scenarios=10)
    mc_summary = summarize_monte_carlo(scenario_results)
    print_monte_carlo_report(scenario_results, mc_summary)
