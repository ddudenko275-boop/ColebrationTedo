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
from src.capital import IRBAssumptions, summarize_irb_capital

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


def prepare_fixed_pipeline(portfolio: str = "stress", random_state: int = RANDOM_STATE) -> dict:
    """Fit RF and all calibrators once on the full in-time sample.

    Returns everything needed to run scenarios by resampling and predicting
    only -- nothing here is refit inside a scenario.
    """

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

    calibrators = get_all_calibrators()
    for calibrator in calibrators.values():
        calibrator.fit(scores_calib_full, y_calib.to_numpy(dtype=float))

    return {
        "rf": rf,
        "calibrators": calibrators,
        "x_calib": x_calib,
        "y_calib": y_calib,
        "x_test": x_test,
        "y_test": y_test,
    }


def run_scenario(
    pipeline: dict,
    scenario_seed: int,
    sample_frac: float = 0.8,
    capital_assumptions: IRBAssumptions | None = None,
) -> pd.DataFrame:
    """Score one resampled scenario through every fixed calibrator.

    In-time and OOT rows are resampled independently (without replacement)
    from the same fixed pipeline, so p_value_intime and p_value_oot each
    reflect sampling variability in their own population.
    """

    assumptions = capital_assumptions or DEFAULT_CAPITAL_ASSUMPTIONS
    rng = np.random.default_rng(scenario_seed)

    x_calib, y_calib = pipeline["x_calib"], pipeline["y_calib"]
    x_test, y_test = pipeline["x_test"], pipeline["y_test"]

    idx_calib = rng.choice(len(x_calib), size=int(round(len(x_calib) * sample_frac)), replace=False)
    idx_test = rng.choice(len(x_test), size=int(round(len(x_test) * sample_frac)), replace=False)

    xc, yc = x_calib.iloc[idx_calib], y_calib.iloc[idx_calib].to_numpy(dtype=float)
    xt, yt = x_test.iloc[idx_test], y_test.iloc[idx_test].to_numpy(dtype=float)

    scores_calib = np.clip(pipeline["rf"].predict_proba(xc)[:, 1], 1e-6, 1 - 1e-6)
    scores_test = np.clip(pipeline["rf"].predict_proba(xt)[:, 1], 1e-6, 1 - 1e-6)

    rows = []
    for method, calibrator in pipeline["calibrators"].items():
        pred_calib = calibrator.predict(scores_calib)
        pred_test = calibrator.predict(scores_test)
        capital = summarize_irb_capital(pred_test, assumptions=assumptions)

        rows.append(
            {
                "scenario": scenario_seed,
                "method": method,
                "p_value_intime": min_binomial_p_value(yc, pred_calib),
                "p_value_oot": min_binomial_p_value(yt, pred_test),
                "oot_mean_pd": float(np.mean(pred_test)),
                "expected_loss": capital["total_expected_loss"],
                "unexpected_loss_capital": capital["total_unexpected_loss_capital"],
                "rwa": capital["total_rwa"],
                "required_capital": capital["total_required_capital"],
            }
        )
    return pd.DataFrame(rows)


def run_monte_carlo(
    n_scenarios: int = 10,
    sample_frac: float = 0.8,
    portfolio: str = "stress",
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run n_scenarios resampling scenarios; one row per (scenario, method).

    Start with a small n_scenarios (default 10) to validate the pipeline
    before scaling up to the full 1000-scenario run.
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
