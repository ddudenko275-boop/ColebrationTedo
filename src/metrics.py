"""
Метрики качества калибровки PD моделей.

Метрики:
    - Brier Score              — общая точность вероятностных прогнозов
    - Log-Loss                 — логарифмическая функция потерь
    - ECE                      — Expected Calibration Error
    - Hosmer-Lemeshow test     — статистический тест калибровки (стандарт в банках)
    - Calibration Slope        — наклон регрессии реальных PD на предсказанные
    - Calibration Intercept    — сдвиг той же регрессии
    - reliability_data         — данные для построения диаграммы надёжности
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from scipy import stats


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Brier Score — среднеквадратичная ошибка вероятностных прогнозов.
    Диапазон: [0, 1], чем меньше — тем лучше. Идеал: 0.
    """
    return brier_score_loss(y_true, y_prob)


def log_loss_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Log-Loss (бинарная кросс-энтропия).
    Штрафует сильнее за уверенные неверные прогнозы.
    Чем меньше — тем лучше.
    """
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return log_loss(y_true, y_prob_clipped)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE — взвешенное среднее абсолютных отклонений между
    средним предсказанием и реальной частотой дефолтов в каждом бине.
    Диапазон: [0, 1], чем меньше — тем лучше.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc  = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return ece


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Тест Хосмера-Лемешова (Hosmer-Lemeshow).

    Стандартный статистический тест калибровки в банковском риск-менеджменте
    (требование Basel III / IRB-подход).

    Гипотезы:
        H0: модель хорошо откалибрована (нет значимых отклонений)
        H1: модель плохо откалибрована

    Интерпретация p-value:
        p > 0.05  — нет оснований отвергнуть H0 (калибровка приемлема)
        p < 0.05  — калибровка статистически значимо плохая

    Returns:
        dict с ключами: chi2, p_value, df, verdict
    """
    # Разбивка на децили по предсказанным вероятностям
    quantiles = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    quantiles = np.unique(quantiles)

    chi2_stat = 0.0
    df = 0

    for i in range(len(quantiles) - 1):
        if i == len(quantiles) - 2:
            mask = (y_prob >= quantiles[i]) & (y_prob <= quantiles[i + 1])
        else:
            mask = (y_prob >= quantiles[i]) & (y_prob < quantiles[i + 1])

        if mask.sum() == 0:
            continue

        n_i       = mask.sum()
        observed  = y_true[mask].sum()
        expected  = y_prob[mask].sum()

        if expected > 0 and (n_i - expected) > 0:
            chi2_stat += (observed - expected) ** 2 / (expected * (1 - expected / n_i))
            df += 1

    df = max(df - 2, 1)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=df)
    verdict = "Калибровка приемлема (p > 0.05)" if p_value > 0.05 else "Калибровка значимо плохая (p ≤ 0.05)"

    return {
        "chi2":    round(chi2_stat, 4),
        "p_value": round(p_value, 4),
        "df":      df,
        "verdict": verdict,
    }


def calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Calibration Slope и Intercept.

    Регрессия логит(реальные PD) ~ a + b * логит(предсказанные PD).

    Идеальные значения:
        intercept = 0   — нет систематического смещения
        slope     = 1   — масштаб предсказаний верный

    slope < 1 → модель "уверена" (скоры слишком широко разбросаны)
    slope > 1 → модель "не уверена" (скоры сжаты к центру)
    intercept ≠ 0 → систематическое завышение/занижение PD
    """
    eps = 1e-7
    logit_prob = np.log(np.clip(y_prob, eps, 1 - eps) / (1 - np.clip(y_prob, eps, 1 - eps)))

    model = LogisticRegression(solver="lbfgs", max_iter=500)
    model.fit(logit_prob.reshape(-1, 1), y_true)

    slope     = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])

    return {
        "slope":     round(slope, 4),
        "intercept": round(intercept, 4),
    }


def get_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
):
    """
    Данные для построения Reliability Diagram.

    Returns:
        mean_predicted:      средний прогноз в каждом бине
        fraction_of_positives: реальная частота дефолтов в каждом бине
    """
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    return mean_predicted, fraction_of_positives


def summary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str = "",
) -> dict:
    """
    Полный набор метрик для одного метода калибровки.

    Returns:
        dict с метриками: Brier, Log-Loss, ECE, HL p-value, Slope, Intercept
    """
    hl   = hosmer_lemeshow_test(y_true, y_prob)
    si   = calibration_slope_intercept(y_true, y_prob)

    return {
        "method":      name,
        "brier_score": round(brier_score(y_true, y_prob), 5),
        "log_loss":    round(log_loss_score(y_true, y_prob), 5),
        "ece":         round(expected_calibration_error(y_true, y_prob), 5),
        "hl_chi2":     hl["chi2"],
        "hl_p_value":  hl["p_value"],
        "cal_slope":   si["slope"],
        "cal_intercept": si["intercept"],
    }
