"""
Метрики качества PD-моделей: калибровка, дискриминация, стабильность.

Калибровка:
    - Brier Score              — среднеквадратичная ошибка вероятностных прогнозов
    - Log-Loss                 — логарифмическая функция потерь
    - ECE                      — Expected Calibration Error
    - Hosmer-Lemeshow test     — статистический тест калибровки (стандарт Basel III)
    - Calibration Slope/Intercept — наклон и сдвиг регрессии реальных PD на предсказанные

Дискриминация:
    - AUC-ROC                  — площадь под ROC-кривой
    - Gini coefficient         — 2 × AUC − 1, стандарт в кредитном скоринге
    - KS statistic             — максимальное расхождение кумулятивных распределений

Стабильность:
    - PSI                      — Population Stability Index, мониторинг дрейфа

Надёжность метрик:
    - bootstrap_ci             — доверительный интервал любой метрики через Bootstrap
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
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
    Полный набор метрик калибровки для одного метода.

    Returns:
        dict с метриками: Brier, Log-Loss, ECE, HL p-value, Slope, Intercept
    """
    hl   = hosmer_lemeshow_test(y_true, y_prob)
    si   = calibration_slope_intercept(y_true, y_prob)

    return {
        "method":        name,
        "brier_score":   round(brier_score(y_true, y_prob), 5),
        "log_loss":      round(log_loss_score(y_true, y_prob), 5),
        "ece":           round(expected_calibration_error(y_true, y_prob), 5),
        "hl_chi2":       hl["chi2"],
        "hl_p_value":    hl["p_value"],
        "cal_slope":     si["slope"],
        "cal_intercept": si["intercept"],
    }


# ---------------------------------------------------------------------------
# Метрики дискриминации
# ---------------------------------------------------------------------------

def discrimination_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str = "",
) -> dict:
    """
    Метрики дискриминирующей способности модели.

    AUC-ROC:
        Вероятность того, что модель выше оценит дефолтного заёмщика,
        чем недефолтного. Диапазон [0.5, 1.0], идеал: 1.0.

    Gini coefficient:
        Gini = 2 × AUC − 1. Стандарт кредитного скоринга.
        Диапазон [0, 1]: > 0.4 — приемлемо, > 0.6 — хорошо.

    KS statistic (Kolmogorov-Smirnov):
        Максимальное расстояние между кумулятивными распределениями
        дефолтных и недефолтных клиентов.
        Диапазон [0, 1]: > 0.3 — приемлемо, > 0.5 — хорошо.

    Важно: дискриминация и калибровка — независимые свойства.
    Хорошо откалиброванная модель с Gini=0.3 хуже, чем
    плохо откалиброванная с Gini=0.7 (калибровку можно поправить).
    """
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = float(np.max(np.abs(tpr - fpr)))

    return {
        "method": name,
        "auc_roc": round(auc, 4),
        "gini":    round(gini, 4),
        "ks_stat": round(ks, 4),
    }


# ---------------------------------------------------------------------------
# Population Stability Index (PSI)
# ---------------------------------------------------------------------------

def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Population Stability Index — индекс стабильности популяции.

    Измеряет сдвиг распределения скоров между двумя выборками:
        expected — эталонное распределение (обычно обучающая выборка)
        actual   — проверяемое распределение (калибровочная или тестовая)

    Формула: PSI = Σ (actual_% − expected_%) × ln(actual_% / expected_%)

    Интерпретация:
        PSI < 0.10  — распределение стабильно, изменений нет
        0.10–0.25   — умеренный сдвиг, требует мониторинга
        > 0.25      — значительный сдвиг, модель могла устареть

    Returns:
        dict с ключами: psi_value, verdict, bin_details (DataFrame)
    """
    # Строим бины по квантилям expected
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    eps = 1e-7
    rows = []
    psi_value = 0.0

    for i in range(len(breakpoints) - 1):
        exp_pct = np.mean((expected >= breakpoints[i]) & (expected < breakpoints[i + 1]))
        act_pct = np.mean((actual   >= breakpoints[i]) & (actual   < breakpoints[i + 1]))

        exp_pct = max(exp_pct, eps)
        act_pct = max(act_pct, eps)

        bucket_psi = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        psi_value += bucket_psi
        rows.append({
            "bin":      i + 1,
            "exp_%":    round(exp_pct * 100, 2),
            "act_%":    round(act_pct * 100, 2),
            "psi_bin":  round(bucket_psi, 5),
        })

    if psi_value < 0.10:
        verdict = "Стабильно (PSI < 0.10)"
    elif psi_value < 0.25:
        verdict = "Умеренный сдвиг (0.10 <= PSI < 0.25) -- мониторинг"
    else:
        verdict = "Значительный сдвиг (PSI >= 0.25) -- переобучка модели"

    return {
        "psi_value":  round(psi_value, 5),
        "verdict":    verdict,
        "bin_details": pd.DataFrame(rows),
    }


# ---------------------------------------------------------------------------
# Bootstrap доверительные интервалы
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_iter: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap доверительный интервал для произвольной метрики.

    Зачем нужно:
        При малом числе дефолтов (~115 в тесте) точечные оценки метрик
        имеют высокую дисперсию. Bootstrap показывает реальную ширину CI:
        если интервалы двух методов перекрываются — их различие незначимо.

    Параметры:
        metric_fn  — функция вида f(y_true, y_prob) → float
        n_iter     — число bootstrap-итераций (1000 достаточно для CI)
        ci         — уровень доверия (0.95 → 95% CI)

    Returns:
        dict с ключами: point_estimate, ci_lower, ci_upper, std
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    bootstrap_scores = []

    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        # Пропускаем итерации без обоих классов (редко, но возможно)
        if len(np.unique(y_b)) < 2:
            continue
        try:
            bootstrap_scores.append(metric_fn(y_b, p_b))
        except Exception:
            continue

    scores = np.array(bootstrap_scores)
    alpha = (1 - ci) / 2

    return {
        "point_estimate": round(metric_fn(y_true, y_prob), 5),
        "ci_lower":       round(float(np.percentile(scores, alpha * 100)), 5),
        "ci_upper":       round(float(np.percentile(scores, (1 - alpha) * 100)), 5),
        "std":            round(float(scores.std()), 5),
        "n_iter":         len(scores),
    }
