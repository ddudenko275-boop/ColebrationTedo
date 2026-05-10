import numpy as np

from src.metrics import calibration_bin_table


def test_calibration_bin_table_marks_sparse_uniform_bins():
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.05, 0.15, 0.25, 0.55, 0.85])

    table = calibration_bin_table(y_true, y_prob, n_bins=10, strategy="uniform", min_count=2)

    sparse = table.loc[np.isclose(table["bin_left"], 0.5) & np.isclose(table["bin_right"], 0.6)]
    assert len(sparse) == 1
    assert sparse["n"].iloc[0] == 1
    assert sparse["is_sparse"].iloc[0] is True or sparse["is_sparse"].iloc[0] == np.bool_(True)


def test_calibration_bin_table_quantile_bins_avoid_empty_uniform_gaps():
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.01, 0.01, 0.01, 0.60, 0.70, 0.90])

    table = calibration_bin_table(y_true, y_prob, n_bins=10, strategy="quantile", min_count=2)

    assert table["n"].min() >= 1
    assert table["avg_pd"].is_monotonic_increasing
    assert table["bin_left"].iloc[0] == 0.0
    assert table["bin_right"].iloc[-1] == 1.0


def test_calibration_bin_table_ordinal_keeps_requested_bin_count_with_ties():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    y_prob = np.array([0.01, 0.01, 0.01, 0.01, 0.20, 0.20, 0.20, 0.60, 0.60, 0.60])

    table = calibration_bin_table(y_true, y_prob, n_bins=5, strategy="ordinal", min_count=2)

    assert len(table) == 5
    assert table["n"].tolist() == [2, 2, 2, 2, 2]
    assert table["avg_pd"].is_monotonic_increasing


def test_calibration_bin_table_ordinal_can_use_common_sort_values():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.40, 0.10, 0.30, 0.20])
    common_score = np.array([0.01, 0.02, 0.90, 0.95])

    table = calibration_bin_table(
        y_true,
        y_prob,
        n_bins=2,
        strategy="ordinal",
        min_count=1,
        sort_by=common_score,
    )

    assert table["n"].tolist() == [2, 2]
    assert np.isclose(table["avg_pd"].iloc[0], 0.25)
    assert np.isclose(table["avg_pd"].iloc[1], 0.25)
    assert table["defaults"].tolist() == [1, 1]
