"""
Генерация реалистичных синтетических данных кредитного портфеля банка.

Особенности:
    - Доля дефолтов 3-5% (реалистично для розничного кредитования)
    - Временна́я метка выдачи кредита (2019-2024) для out-of-time валидации
    - Банковские признаки: кредитный рейтинг, LTV, DTI, срок и др.
    - Намеренная нелинейность в зависимости скор → PD (сплайн имеет преимущество)
    - n_samples=30_000 по умолчанию: при 3-5% дефолтов даёт ~600+ событий
      в калибровочной выборке, что необходимо для устойчивой калибровки
"""

import numpy as np
import pandas as pd


def generate_credit_data(n_samples: int = 30000, random_state: int = 42) -> pd.DataFrame:
    """
    Генерирует синтетический кредитный портфель банка.

    Признаки:
        - credit_score:      внутренний кредитный рейтинг клиента (300-850)
        - ltv:               loan-to-value — отношение кредита к залогу (0-1)
        - dti:               debt-to-income — отношение долга к доходу (0-1)
        - employment_years:  стаж работы на текущем месте (лет)
        - loan_amount:       сумма кредита (тыс. руб.)
        - loan_term:         срок кредита (мес.)
        - num_delinquencies: количество просрочек за последние 2 года
        - loan_purpose:      цель кредита (0=ипотека, 1=авто, 2=потреб.)
        - origination_year:  год выдачи кредита (2019-2024)

    Таргет:
        - default: 1 = дефолт в течение 12 мес., 0 = нет дефолта
    """
    rng = np.random.default_rng(random_state)

    # --- Признаки ---
    credit_score    = rng.integers(300, 851, size=n_samples).astype(float)
    ltv             = rng.beta(a=3, b=2, size=n_samples)          # смещено к 0.6
    dti             = rng.beta(a=2, b=4, size=n_samples)          # смещено к 0.3
    employment_years = rng.exponential(scale=5, size=n_samples)
    employment_years = np.clip(employment_years, 0, 40)
    loan_amount     = rng.lognormal(mean=13.5, sigma=0.8, size=n_samples) / 1000  # тыс. руб.
    loan_term       = rng.choice([12, 24, 36, 48, 60, 84, 120], size=n_samples).astype(float)
    num_delinquencies = rng.poisson(lam=0.3, size=n_samples).astype(float)
    loan_purpose    = rng.choice([0, 1, 2], size=n_samples, p=[0.4, 0.25, 0.35])
    origination_year = rng.choice(
        [2019, 2020, 2021, 2022, 2023, 2024],
        size=n_samples,
        p=[0.15, 0.15, 0.20, 0.20, 0.20, 0.10],
    )

    # --- Истинная PD: нелинейная зависимость (кусочная + взаимодействия) ---
    # Нормализуем credit_score в [0, 1]
    cs_norm = (credit_score - 300) / 550

    # Нелинейный эффект кредитного рейтинга (ступенчатый — сплайн справится лучше)
    cs_effect = np.where(
        cs_norm < 0.3, 2.5 - 4.0 * cs_norm,
        np.where(cs_norm < 0.6, 1.5 - 2.5 * cs_norm, 0.5 - 1.0 * cs_norm)
    )

    log_odds = (
        -6.2
        + cs_effect
        + 2.0 * ltv
        + 1.5 * dti
        - 0.05 * employment_years
        + 0.8 * num_delinquencies
        + 0.3 * (loan_purpose == 2).astype(float)   # потреб. кредиты рискованнее
        + 0.0002 * loan_amount
        - 0.003 * loan_term
    )

    # Добавляем макро-шок в 2020 году (COVID) — повышение дефолтности
    log_odds += 0.5 * (origination_year == 2020).astype(float)

    true_prob = 1 / (1 + np.exp(-log_odds))
    default = rng.binomial(n=1, p=true_prob)

    df = pd.DataFrame({
        "credit_score":      credit_score,
        "ltv":               ltv.round(4),
        "dti":               dti.round(4),
        "employment_years":  employment_years.round(1),
        "loan_amount":       loan_amount.round(1),
        "loan_term":         loan_term,
        "num_delinquencies": num_delinquencies,
        "loan_purpose":      loan_purpose,
        "origination_year":  origination_year,
        "default":           default,
    })

    return df


def get_oot_split(df: pd.DataFrame, target_col: str = "default"):
    """
    Out-of-time (OOT) разбивка:
        Train:       2019-2021  (~50%)
        Calibration: 2022-2023  (~40%)
        Test (OOT):  2024       (~10%)

    Такое разбиение имитирует реальную банковскую валидацию:
    модель обучена на прошлом, калибровка на недавнем, тест — на свежих данных.
    """
    feature_cols = [c for c in df.columns if c not in (target_col, "origination_year")]

    train_mask = df["origination_year"].isin([2019, 2020, 2021])
    calib_mask = df["origination_year"].isin([2022, 2023])
    test_mask  = df["origination_year"] == 2024

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]

    X_calib = df.loc[calib_mask, feature_cols]
    y_calib = df.loc[calib_mask, target_col]

    X_test  = df.loc[test_mask, feature_cols]
    y_test  = df.loc[test_mask, target_col]

    return X_train, X_calib, X_test, y_train, y_calib, y_test


if __name__ == "__main__":
    df = generate_credit_data(n_samples=30000)
    print(f"Датасет: {df.shape[0]:,} строк, {df.shape[1]} столбцов")
    print(f"Доля дефолтов: {df['default'].mean():.2%}")
    print(f"\nДефолты по годам:")
    print(df.groupby("origination_year")["default"].agg(["sum", "mean", "count"])
            .rename(columns={"sum": "дефолтов", "mean": "доля", "count": "всего"}))
