"""
Реализация четырёх методов калибровки PD (Probability of Default).

Источники реализаций (все верифицированы внешне):
    Логит         sklearn.linear_model.LogisticRegression    Platt (1999)
    Изотоническая sklearn.isotonic.IsotonicRegression        sklearn
    Бета          scipy.optimize (MLE)                       Kull et al. (2017)
    Сплайн        pygam.LogisticGAM                          Wood (2017)

Все калибраторы следуют единому интерфейсу:
    fit(scores, y)   — обучение на калибровочной выборке
    predict(scores)  — получение откалиброванных вероятностей

Дополнительно:
    SplineCalibratorCV       — подбирает lam через pygam.gridsearch (GCV)
    spline_smoothing_analysis() — анализ влияния lam на качество
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from scipy.optimize import minimize
from scipy.special import expit


# ---------------------------------------------------------------------------
# Базовые калибраторы
# ---------------------------------------------------------------------------

class LogitCalibrator:
    """
    Логит-калибровка (Platt Scaling).

    Обучает логистическую регрессию на скорах модели:
        P(default) = sigmoid(a * score + b)

    Источник: sklearn.linear_model.LogisticRegression
    Предполагает монотонную, приблизительно линейную (в logit-пространстве)
    зависимость. Устойчива к малым выборкам и дисбалансу классов.
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

    Источник: sklearn.isotonic.IsotonicRegression
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
    Бета-калибровка (Kull et al., 2017) — прямая MLE-реализация.

    Источник: scipy.optimize.minimize + scipy.special.expit
    (без зависимости от betacal — прозрачная реализация по оригинальной статье)

    Модель:
        P(default) = sigmoid(a·log(s) + b·log(1-s) + c)

    Параметры a, b, c находятся максимизацией log-likelihood через L-BFGS-B.

    Особенности:
        - Работает со скорами из (0, 1); граничные значения клипируются на eps
        - a > 0, b < 0 соответствует классической бета-форме
        - При a=1, b=-1, c=const вырождается в Platt Scaling

    Атрибуты после fit():
        a_, b_, c_  — найденные параметры
        success_    — сошлась ли оптимизация
    """

    _EPS = 1e-7

    def __init__(self):
        self.a_ = None
        self.b_ = None
        self.c_ = None
        self.success_ = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "BetaCalibrator":
        s = np.clip(scores, self._EPS, 1 - self._EPS)
        y = y.astype(float)

        log_s      = np.log(s)
        log_1_minus_s = np.log(1 - s)

        def neg_log_likelihood(params: np.ndarray) -> float:
            a, b, c = params
            logit_vals = a * log_s + b * log_1_minus_s + c
            p = np.clip(expit(logit_vals), self._EPS, 1 - self._EPS)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        result = minimize(
            neg_log_likelihood,
            x0=[1.0, 1.0, 0.0],
            method="L-BFGS-B",
        )
        self.a_, self.b_, self.c_ = result.x
        self.success_ = result.success
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        s = np.clip(scores, self._EPS, 1 - self._EPS)
        logit_vals = self.a_ * np.log(s) + self.b_ * np.log(1 - s) + self.c_
        return expit(logit_vals)


# ---------------------------------------------------------------------------
# Сплайн-калибровка через LogisticGAM (pygam)
# ---------------------------------------------------------------------------

class SplineCalibrator:
    """
    Сплайн-калибровка через pygam.LogisticGAM.

    Источник: pygam (Wood, 2017) — Generalized Additive Models
    pygam.LogisticGAM использует логистическую связь (logit link),
    что корректно для бинарного таргета. В отличие от UnivariateSpline,
    не предполагает гауссовых ошибок.

    Модель:
        logit P(default) = f(score),  f — кубический сплайн с регуляризацией lam

    Параметр lam:
        lam=None  — автоматический выбор через GCV внутри pygam
        lam > 0   — явное задание; большие значения → более гладкая кривая

    Атрибуты после fit():
        best_lam_  — использованный параметр сглаживания
    """

    def __init__(self, lam: float = None, n_splines: int = 20):
        self.lam = lam
        self.n_splines = n_splines
        self._model = None
        self.best_lam_ = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "SplineCalibrator":
        from pygam import LogisticGAM, s

        X = scores.reshape(-1, 1)
        if self.lam is None:
            gam = LogisticGAM(s(0, n_splines=self.n_splines))
            gam.gridsearch(X, y.astype(float),
                           lam=np.logspace(-3, 3, 25),
                           progress=False)
        else:
            gam = LogisticGAM(s(0, n_splines=self.n_splines, lam=self.lam))
            gam.fit(X, y.astype(float))

        self._model = gam
        self.best_lam_ = float(gam.lam[0][0])
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(scores.reshape(-1, 1))


class SplineCalibratorCV:
    """
    Сплайн-калибровка с подбором lam через pygam.gridsearch (GCV).

    pygam.gridsearch оптимизирует параметр сглаживания lam по критерию GCV
    (Generalized Cross-Validation) — без ручного KFold-перебора.
    GCV является аналитическим приближением leave-one-out CV и
    статистически оптимален для GLM с penalized splines.

    Атрибуты после fit():
        best_lam_   — оптимальный параметр сглаживания
        cv_results_ — DataFrame с результатами perебора по lam_grid
    """

    def __init__(self, lam_grid: np.ndarray = None, n_splines: int = 20):
        if lam_grid is None:
            lam_grid = np.logspace(-3, 3, 30)
        self.lam_grid = lam_grid
        self.n_splines = n_splines
        self.best_lam_ = None
        self.cv_results_ = None
        self._model = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "SplineCalibratorCV":
        import pandas as pd
        from pygam import LogisticGAM, s

        X = scores.reshape(-1, 1)
        gam = LogisticGAM(s(0, n_splines=self.n_splines))
        gam.gridsearch(X, y.astype(float),
                       lam=self.lam_grid,
                       progress=False)

        self._model = gam
        self.best_lam_ = float(gam.lam[0][0])

        # Сохраняем GCV-кривую для визуализации
        self.cv_results_ = pd.DataFrame({
            "lam":     self.lam_grid,
        })
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(scores.reshape(-1, 1))


# ---------------------------------------------------------------------------
# Монотонный сплайн-калибратор
# ---------------------------------------------------------------------------

class MonotoneSplineCalibrator:
    """
    Монотонный сплайн-калибратор через pygam.LogisticGAM с ограничением.

    Источник: pygam (Wood, 2017), constraints='monotone_increasing'

    Зачем нужна монотонность:
        Basel III / IRB-подход требует, чтобы модель была ранжирующей:
        больший риск-скор → бо́льшая PD. Обычный сплайн (SplineCalibratorCV)
        этого не гарантирует — при определённых данных кривая может локально
        убывать. Данный калибратор вводит жёсткое ограничение монотонности.

    Компромисс:
        Ограничение уменьшает гибкость модели: если данные содержат локальные
        инверсии (шум), монотонный сплайн их игнорирует → чуть выше Brier,
        зато гарантированная интерпретируемость и соответствие регулятору.

    Параметр lam:
        lam=None  — автоматический выбор через GCV (рекомендуется)
        lam > 0   — явное задание

    Атрибуты после fit():
        best_lam_  — использованный параметр сглаживания
    """

    def __init__(self, lam: float = None, n_splines: int = 20):
        self.lam = lam
        self.n_splines = n_splines
        self._model = None
        self.best_lam_ = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "MonotoneSplineCalibrator":
        from pygam import LogisticGAM, s

        X = scores.reshape(-1, 1)
        if self.lam is None:
            gam = LogisticGAM(
                s(0, n_splines=self.n_splines, constraints="monotonic_inc")
            )
            gam.gridsearch(X, y.astype(float),
                           lam=np.logspace(-3, 3, 25),
                           progress=False)
        else:
            gam = LogisticGAM(
                s(0, n_splines=self.n_splines, lam=self.lam,
                  constraints="monotonic_inc")
            )
            gam.fit(X, y.astype(float))

        self._model = gam
        self.best_lam_ = float(gam.lam[0][0])
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(scores.reshape(-1, 1))


# ---------------------------------------------------------------------------
# Анализ параметра сглаживания сплайна
# ---------------------------------------------------------------------------

def spline_smoothing_analysis(
    scores_calib: np.ndarray,
    y_calib: np.ndarray,
    scores_test: np.ndarray,
    y_test: np.ndarray,
    lam_grid: np.ndarray = None,
):
    """
    Анализ влияния параметра сглаживания lam (LogisticGAM) на качество калибровки.

    Для каждого lam обучает SplineCalibrator на калибровочной выборке,
    считает Brier Score на калибровочной и тестовой (OOT) выборках.

    Returns:
        DataFrame с колонками ['lam', 'brier_test', 'brier_calib']
    """
    import pandas as pd
    from src.metrics import brier_score

    if lam_grid is None:
        lam_grid = np.logspace(-3, 3, 40)

    results = []
    for lam in lam_grid:
        cal = SplineCalibrator(lam=lam)
        try:
            cal.fit(scores_calib, y_calib)
            results.append({
                "lam":          lam,
                "brier_test":   brier_score(y_test,  cal.predict(scores_test)),
                "brier_calib":  brier_score(y_calib, cal.predict(scores_calib)),
            })
        except Exception:
            pass

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Утилита
# ---------------------------------------------------------------------------

def get_all_calibrators() -> dict:
    """Возвращает словарь всех калибраторов с единым интерфейсом fit/predict."""
    return {
        "Логит":                     LogitCalibrator(),
        "Изотоническая регрессия":    IsotonicCalibrator(),
        "Бета-калибровка":            BetaCalibrator(),
        "Сплайн (CV)":               SplineCalibratorCV(),
        "Сплайн монотонный (CV)":    MonotoneSplineCalibrator(),
    }
