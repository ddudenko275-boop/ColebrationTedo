"""Tests for the two calibration statistics used by the Monte Carlo engine."""

import numpy as np
import pytest

from src.monte_carlo import (
    master_scale_grade_binomial_p_value,
    whole_model_binomial_p_value,
)


def test_whole_model_passes_when_predictions_match_observed_rate():
    rng = np.random.default_rng(0)
    pred = np.full(5000, 0.05)
    y = (rng.random(5000) < 0.05).astype(float)

    assert whole_model_binomial_p_value(y, pred) > 0.05


def test_whole_model_rejects_a_clear_level_miss():
    y = np.concatenate([np.ones(500), np.zeros(4500)])  # observed 10%
    pred = np.full(5000, 0.05)  # model says 5%

    assert whole_model_binomial_p_value(y, pred) < 0.01


def test_grade_p_value_is_corrected_for_the_number_of_grades_tested():
    """The raw minimum over many grades must not be returned as-is.

    Taking a minimum over ~13 grades makes small p-values common even under a
    perfect model, so the statistic has to carry a multiplicity correction or it
    rejects a well-calibrated model about half the time.
    """

    rep = np.array([0.02, 0.02, 0.02])
    rng = np.random.default_rng(1)
    n = 3000
    grades = rng.integers(0, 3, n)
    y = (rng.random(n) < 0.02).astype(float)

    corrected = master_scale_grade_binomial_p_value(y, grades, rep)

    # Reconstruct the uncorrected minimum the function must not return.
    from scipy.stats import binomtest

    raw_min = min(
        binomtest(int(y[grades == g].sum()), int((grades == g).sum()), rep[g]).pvalue
        for g in range(3)
    )
    assert corrected == pytest.approx(min(1.0, raw_min * 3))
    assert corrected >= raw_min


def test_grade_p_value_only_counts_non_empty_grades():
    """Empty grades are not tested, so they must not inflate the correction."""

    rep = np.array([0.02, 0.02, 0.02, 0.02])
    rng = np.random.default_rng(2)
    n = 2000
    grades = rng.integers(0, 2, n)  # grades 2 and 3 stay empty
    y = (rng.random(n) < 0.02).astype(float)

    from scipy.stats import binomtest

    raw_min = min(
        binomtest(int(y[grades == g].sum()), int((grades == g).sum()), rep[g]).pvalue
        for g in range(2)
    )
    corrected = master_scale_grade_binomial_p_value(y, grades, rep)

    assert corrected == pytest.approx(min(1.0, raw_min * 2))


def test_grade_p_value_rejects_a_grade_that_misses_its_representative_pd():
    rep = np.array([0.02, 0.40])
    grades = np.concatenate([np.zeros(1000, dtype=int), np.ones(1000, dtype=int)])
    # Grade 1 realises 20% against a declared 40%: a real, large miss.
    y = np.concatenate([np.zeros(980), np.ones(20), np.zeros(800), np.ones(200)])

    assert master_scale_grade_binomial_p_value(y, grades, rep) < 0.01


def test_grade_p_value_is_nan_without_any_populated_grade():
    rep = np.array([0.02, 0.05])
    assert np.isnan(
        master_scale_grade_binomial_p_value(
            np.array([]), np.array([], dtype=int), rep
        )
    )
