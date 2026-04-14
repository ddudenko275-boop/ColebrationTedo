"""
Реализация четырёх методов калибровки PD (Probability of Default).

Все калибраторы следуют единому интерфейсу:
    fit(scores, y)   — обучение на калибровочной выборке
    predict(scores)  — получение откалиброванных вероятностей

Дополнительно для сплайна:
    SplineCalibratorCV — подбирает оптимальный параметр сглаживания по CV
    spline_smoothing_analysis() — анализ влияния параметра на качество
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss
from scipy.interpolate import UnivariateSpline
from betacal import BetaCalibration


# ---------------------------------------------------------------------------
# Базовые калибраторы
# ---------------------------------------------------------------------------

class LogitCalibrator:
    """
    Логит-калибровка (Platt Scaling).

    Обучает логистическую регрессию на скорах модели:
        P(default) = sigmoid(a * score + b)

    Предполагает монотонную, приблизительно линейную (в логит-пространстве)
    зависимость. Устойчива к малым выборкам.
    """

    def __init__(self):
        self._model = LogisticRegression(solver="lbfgs", max_iter=500)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "LogitCalibrator":
        self._model.fit(scores.reshape(-1, 1), y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(scores.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """
    Изотоническая регрессия.

    Непараметрический метод — ищет монотонно возрастающую ступенчатую функцию.
    Не предполагает форму зависимости, но склонна к переобучению на малых выборках.
    """

    def __init__(self):
        self._model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._model.fit(scores, y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._model.predict(scores)


class BetaCalibrator:
    """
    Бета-калибровка (Kull et al., 2017).

    Моделирует преобразование скоров через бета-распределение:
        P(default) = sigmoid(a * log(score) + b * log(1 - score) + c)

    Особенно подходит для скоров из [0, 1]. Более гибкая, чем логит-калибровка,
    сохраняет граничные значения 0 и 1.
    """

    def __init__(self):
        self._model = BetaCalibration(parameters="abm")

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "BetaCalibrator":
        self._model.fit(scores.reshape(-1, 1), y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._model.predict(scores.reshape(-1, 1))


class SplineCalibrator:
    """
    Сплайн-калибровка.

    Кубический сглаживающий сплайн аппроксимирует нелинейную зависимость
    между скорами и реальными вероятностями дефолта.

    Параметр smoothing_factor (s):
        s=None  — автоматический выбор scipy (часто переобучается)
        s=0     — интерполяция точек (переобучение)
        s>0     — сглаживание (рекомендуется подбирать через SplineCalibratorCV)

    Преимущество перед логитом: улавливает нелинейности в зоне низких PD (0-0.3),
    что критично для банковских портфелей.
    """

    def __init__(self, smoothing_factor: float = None):
        self.smoothing_factor = smoothing_factor
        self._spline = None
        self._score_min = None
        self._score_max = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "SplineCalibrator":
        self._score_min = scores.min()
        self._score_max = scores.max()

        sort_idx = np.argsort(scores)
        sorted_scores = scores[sort_idx]
        sorted_y = y[sort_idx].astype(float)

        self._spline = UnivariateSpline(
            sorted_scores,
            sorted_y,
            k=3,
            s=self.smoothing_factor,
            ext="const",
        )
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        probs = self._spline(scores)
        return np.clip(probs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Сплайн с кросс-валидационным подбором параметра сглаживания
# ---------------------------------------------------------------------------

class SplineCalibratorCV:
    """
    Сплайн-калибровка с автоматическим подбором параметра сглаживания (CV).

    Перебирает сетку значений smoothing_factor и выбирает значение,
    минимизирующее Brier Score на кросс-валидации по калибровочной выборке.

    Атрибуты после fit():
        best_s_     — оптимальный параметр сглаживания
        cv_results_ — DataFrame с результатами перебора
    """

    def __init__(self, s_grid: np.ndarray = None, n_folds: int = 5):
        if s_grid is None:
            s_grid = np.logspace(-4, 0, 20)
        self.s_grid = s_grid
        self.n_folds = n_folds
        self.best_s_ = None
        self.cv_results_ = None
        self._best_calibrator = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "SplineCalibratorCV":
        import pandas as pd

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        results = []

        for s in self.s_grid:
            fold_scores = []
            for train_idx, val_idx in kf.split(scores):
                cal = SplineCalibrator(smoothing_factor=s)
                try:
                    cal.fit(scores[train_idx], y[train_idx])
                    preds = cal.predict(scores[val_idx])
                    fold_scores.append(brier_score_loss(y[val_idx], preds))
                except Exception:
                    fold_scores.append(np.inf)
            results.append({"s": s, "brier_cv": np.mean(fold_scores)})

        self.cv_results_ = pd.DataFrame(results)
        self.best_s_ = self.cv_results_.loc[
            self.cv_results_["brier_cv"].idxmin(), "s"
        ]

        # Обучаем финальную модель на всех данных с лучшим s
        self._best_calibrator = SplineCalibrator(smoothing_factor=self.best_s_)
        self._best_calibrator.fit(scores, y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._best_calibrator.predict(scores)


# ---------------------------------------------------------------------------
# Анализ сглаживания сплайна
# ---------------------------------------------------------------------------

def spline_smoothing_analysis(
    scores_calib: np.ndarray,
    y_calib: np.ndarray,
    scores_test: np.ndarray,
    y_test: np.ndarray,
    s_grid: np.ndarray = None,
):
    """
    Анализ влияния параметра сглаживания сплайна на качество калибровки.

    Обучает сплайн с разными значениями s на калибровочной выборке,
    оценивает Brier Score на тестовой (OOT) выборке.

    Returns:
        DataFrame с колонками ['s', 'brier_test', 'brier_calib']
    """
    import pandas as pd
    from src.metrics import brier_score

    if s_grid is None:
        s_grid = np.logspace(-4, 0, 30)

    results = []
    for s in s_grid:
        cal = SplineCalibrator(smoothing_factor=s)
        try:
            cal.fit(scores_calib, y_calib)
            preds_test  = cal.predict(scores_test)
            preds_calib = cal.predict(scores_calib)
            results.append({
                "s":            s,
                "brier_test":   brier_score(y_test, preds_test),
                "brier_calib":  brier_score(y_calib, preds_calib),
            })
        except Exception:
            pass

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Утилита
# ---------------------------------------------------------------------------

def get_all_calibrators() -> dict:
    """Возвращает словарь базовых калибраторов (без CV-сплайна)."""
    return {
        "Логит":                  LogitCalibrator(),
        "Изотоническая регрессия": IsotonicCalibrator(),
        "Бета-калибровка":        BetaCalibrator(),
        "Сплайн (CV)":            SplineCalibratorCV(),
    }
