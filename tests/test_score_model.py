import numpy as np

from data.generate_data import generate_credit_data, get_oot_split
from src.score_model import fit_oof_score_model


def test_fit_oof_score_model_returns_oof_and_oot_scores():
    df = generate_credit_data(n_samples=1500, random_state=123, portfolio="stress")
    x_train, _, x_test, y_train, _, _ = get_oot_split(df)

    _, scores_calib, scores_test, fold_ids = fit_oof_score_model(
        x_train,
        y_train,
        x_test,
        random_state=123,
        n_splits=3,
    )

    assert scores_calib.shape == (len(x_train),)
    assert scores_test.shape == (len(x_test),)
    assert fold_ids.shape == (len(x_train),)
    assert set(np.unique(fold_ids)) == {1, 2, 3}
    assert np.all((scores_calib > 0.0) & (scores_calib < 1.0))
    assert np.all((scores_test > 0.0) & (scores_test < 1.0))
