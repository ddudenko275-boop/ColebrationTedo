import numpy as np
import pytest

from src.calibrators import FrenchSplineCalibrator, MonotoneSplineCalibrator


def synthetic_scores_and_defaults() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    scores = np.linspace(0.01, 0.55, 600)
    latent_pd = 0.015 + 0.22 * scores + 0.025 * np.sin(scores * 18.0)
    y = rng.binomial(1, np.clip(latent_pd, 0.0, 1.0))
    return scores, y.astype(int)


@pytest.mark.parametrize(
    "calibrator",
    [
        MonotoneSplineCalibrator(n_bins=20, alpha=35.0),
        FrenchSplineCalibrator(n_bins=20, alpha=35.0, shrinkage=0.6),
    ],
)
def test_spline_calibrators_preserve_fit_central_tendency(calibrator):
    scores, y = synthetic_scores_and_defaults()

    calibrator.fit(scores, y)
    pred = calibrator.predict(scores)

    assert pred.mean() == pytest.approx(y.mean(), abs=1e-12)
    assert np.all((pred > 0.0) & (pred < 1.0))
    assert np.all(np.diff(calibrator.predict(np.linspace(scores.min(), scores.max(), 300))) >= -1e-10)


def test_monotone_spline_extends_support_to_observed_score_bounds():
    scores, y = synthetic_scores_and_defaults()
    calibrator = MonotoneSplineCalibrator(n_bins=20, alpha=35.0)

    calibrator.fit(scores, y)

    assert calibrator.x_min_ == pytest.approx(scores.min())
    assert calibrator.x_max_ == pytest.approx(scores.max())
    grid = np.linspace(scores.min(), scores.max(), 300)
    pred = calibrator.predict(grid)
    assert pred[-1] > pred[-20]


def test_french_spline_does_not_clamp_at_last_bin_mean():
    scores, y = synthetic_scores_and_defaults()
    calibrator = FrenchSplineCalibrator(n_bins=20, alpha=35.0, shrinkage=0.6)

    calibrator.fit(scores, y)

    grid = np.linspace(scores.min(), scores.max(), 300)
    pred = calibrator.predict(grid)
    assert pred[-1] > pred[-20]


@pytest.mark.parametrize(
    "calibrator",
    [
        MonotoneSplineCalibrator(n_bins=20, alpha=10.0),
        FrenchSplineCalibrator(n_bins=20, alpha=10.0, shrinkage=0.6),
    ],
)
def test_spline_knot_diagnostics_keep_ct_and_expose_local_slope(calibrator):
    scores, y = synthetic_scores_and_defaults()

    calibrator.fit(scores, y)
    diagnostics = calibrator.knot_diagnostics()

    assert calibrator.predict(scores).mean() == pytest.approx(y.mean(), abs=1e-12)
    assert len(diagnostics) == len(calibrator.bin_stats_)
    assert {"local_slope", "local_slope_change", "candidate_switch_knot"}.issubset(diagnostics.columns)
    assert diagnostics["candidate_switch_knot"].dtype == bool
