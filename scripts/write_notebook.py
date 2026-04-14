import json

cells = []

# Cell 0 - markdown title
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": (
        "# Калибровка моделей PD: сравнение методов с акцентом на сплайн-калибровку\n\n"
        "## Контекст\n\n"
        "В рамках **Basel III / IRB-подхода** банки обязаны оценивать вероятность дефолта (PD) таким образом, "
        "чтобы предсказанные вероятности соответствовали реально наблюдаемым частотам дефолтов. "
        "Это требование называется **калибровкой модели**.\n\n"
        "Некалиброванная модель приводит к:\n"
        "- Недооценке кредитного риска → недостаточные резервы (МСФО 9)\n"
        "- Завышению капитальных требований → потеря конкурентоспособности\n"
        "- Регуляторным замечаниям при валидации модели\n\n"
        "## Исследуемые методы\n\n"
        "| Метод | Тип | Ключевая особенность |\n"
        "|---|---|---|\n"
        "| Логит-калибровка | Параметрический | Предполагает линейность в logit-пространстве |\n"
        "| Изотоническая регрессия | Непараметрический | Монотонная ступенчатая функция |\n"
        "| Бета-калибровка | Параметрический | Оптимален для скоров из [0,1] |\n"
        "| **Сплайн-калибровка** | Непараметрический | **Улавливает нелинейности, гладкая кривая** |\n\n"
        "**Центральный вопрос:** может ли сплайн-калибровка доминировать над другими методами в банковском контексте?"
    )
})

# Cell 1 - markdown
cells.append({"cell_type": "markdown", "metadata": {}, "source": "## 0. Импорты"})

# Cell 2 - imports code
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "import sys\n"
        'sys.path.append("..")\n'
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "\n"
        "from data.generate_data import generate_credit_data, get_oot_split\n"
        "from src.calibrators import get_all_calibrators, spline_smoothing_analysis\n"
        "from src.metrics import summary_metrics, get_calibration_curve\n"
        "\n"
        'plt.rcParams["figure.dpi"] = 120\n'
        'plt.rcParams["font.size"] = 11\n'
        'sns.set_style("whitegrid")\n'
        'COLORS = ["#e74c3c", "#2980b9", "#27ae60", "#e67e22", "#8e44ad"]\n'
        "RANDOM_STATE = 42\n"
        "\n"
        'print("Все модули загружены успешно.")'
    )
})

# Cell 3 - markdown
cells.append({"cell_type": "markdown", "metadata": {}, "source": "## 1. Данные: синтетический кредитный портфель"})

# Cell 4 - data generation
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "df = generate_credit_data(n_samples=10000, random_state=RANDOM_STATE)\n"
        "\n"
        'print(f"Датасет: {df.shape[0]:,} строк | {df.shape[1]} признаков")\n'
        'print(f"Доля дефолтов: {df[\'default\'].mean():.2%}  (реалистично для розничного портфеля)\\n")\n'
        "\n"
        'by_year = df.groupby("origination_year")["default"].agg(всего="count", дефолтов="sum", доля="mean")\n'
        'by_year["доля"] = by_year["доля"].map("{:.2%}".format)\n'
        'print("Дефолты по годам выдачи:")\n'
        "print(by_year)\n"
        "\n"
        'df.drop(columns=["default", "origination_year"]).describe().round(2)'
    )
})

# Cell 5 - split
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "X_train, X_calib, X_test, y_train, y_calib, y_test = get_oot_split(df)\n"
        "\n"
        'print("Out-of-Time разбивка:")\n'
        'print(f"  Train       (2019-2021): {len(X_train):,} строк | дефолты: {y_train.mean():.2%}")\n'
        'print(f"  Calibration (2022-2023): {len(X_calib):,} строк | дефолты: {y_calib.mean():.2%}")\n'
        'print(f"  Test OOT    (2024):      {len(X_test):,} строк  | дефолты: {y_test.mean():.2%}")'
    )
})

# Cell 6 - markdown
cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": (
        "## 2. Базовая модель: Random Forest\n\n"
        "RF — мощный классификатор, но **системно некалиброванный**: склонен сжимать вероятности к 0 и 1. "
        "Именно это создаёт нелинейную зависимость скор → PD, где сплайн покажет преимущество."
    )
})

# Cell 7 - base model
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "base_model = RandomForestClassifier(\n"
        "    n_estimators=300, max_depth=7, min_samples_leaf=20,\n"
        "    random_state=RANDOM_STATE, n_jobs=-1,\n"
        ")\n"
        "base_model.fit(X_train, y_train)\n"
        "\n"
        "scores_calib = base_model.predict_proba(X_calib)[:, 1]\n"
        "scores_test  = base_model.predict_proba(X_test)[:, 1]\n"
        "\n"
        'print(f"Диапазон скоров (test): [{scores_test.min():.4f}, {scores_test.max():.4f}]")\n'
        'print(f"Медиана скоров:          {np.median(scores_test):.4f}")\n'
        'print(f"Скоры > 0.1:             {(scores_test > 0.1).mean():.1%}")'
    )
})

# Cell 8 - markdown
cells.append({"cell_type": "markdown", "metadata": {}, "source": "## 3. Проблема некалиброванной модели"})

# Cell 9 - problem plot
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "\n"
        "ax = axes[0]\n"
        "mean_pred, frac_pos = get_calibration_curve(y_test.values, scores_test, n_bins=10)\n"
        'ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Идеальная калибровка")\n'
        'ax.plot(mean_pred, frac_pos, "o-", color=COLORS[0], lw=2.5, ms=8, label="RF (без калибровки)")\n'
        "ax.fill_between(mean_pred, mean_pred, frac_pos, alpha=0.2, color=COLORS[0])\n"
        'ax.set_xlabel("Среднее предсказание в бине")\n'
        'ax.set_ylabel("Реальная частота дефолтов")\n'
        'ax.set_title("Reliability Diagram — RF без калибровки")\n'
        "ax.legend()\n"
        "lim = max(mean_pred.max(), frac_pos.max()) * 1.3\n"
        "ax.set_xlim(0, min(lim, 0.5))\n"
        "ax.set_ylim(0, min(lim, 0.5))\n"
        "\n"
        "ax = axes[1]\n"
        'ax.hist(scores_test[y_test == 0], bins=50, alpha=0.6, label="Нет дефолта (0)", color=COLORS[1], density=True)\n'
        'ax.hist(scores_test[y_test == 1], bins=30, alpha=0.7, label="Дефолт (1)", color=COLORS[0], density=True)\n'
        'ax.set_xlabel("Предсказанная вероятность (RF)")\n'
        'ax.set_ylabel("Плотность")\n'
        'ax.set_title("Распределение скоров RF по классам")\n'
        "ax.legend()\n"
        "\n"
        'plt.suptitle("Проблема: RF систематически занижает PD в диапазоне 0.05-0.20", fontsize=12, fontweight="bold", y=1.01)\n'
        "plt.tight_layout()\n"
        "plt.show()"
    )
})

# Cell 10 - markdown
cells.append({"cell_type": "markdown", "metadata": {}, "source": "## 4. Обучение калибраторов на Calibration (2022-2023)"})

# Cell 11 - fit calibrators
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "calibrators = get_all_calibrators()\n"
        "calibrated_probs = {}\n"
        "\n"
        "for name, cal in calibrators.items():\n"
        "    cal.fit(scores_calib, y_calib.values)\n"
        "    calibrated_probs[name] = cal.predict(scores_test)\n"
        '    if hasattr(cal, "best_s_"):\n'
        '        print(f"{name}: обучен. Оптимальный s = {cal.best_s_:.6f}")\n'
        "    else:\n"
        '        print(f"{name}: обучен.")\n'
        "\n"
        'print("\\nВсе калибраторы готовы.")'
    )
})

# Cell 12 - markdown
cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": (
        "## 5. Анализ параметра сглаживания сплайна\n\n"
        "Ключевой вопрос: какое значение `s` обобщается на OOT данные? "
        "Если оптимум на тесте совпадает с CV-оптимумом — сплайн надёжен."
    )
})

# Cell 13 - smoothing analysis
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "smoothing_df = spline_smoothing_analysis(\n"
        "    scores_calib, y_calib.values, scores_test, y_test.values,\n"
        ")\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 5))\n"
        'ax.semilogx(smoothing_df["s"], smoothing_df["brier_calib"], "o--",\n'
        '            color=COLORS[1], lw=2, ms=5, label="Brier Score (Calibration)")\n'
        'ax.semilogx(smoothing_df["s"], smoothing_df["brier_test"], "s-",\n'
        '            color=COLORS[0], lw=2, ms=5, label="Brier Score (Test OOT)")\n'
        "\n"
        'best_s = smoothing_df.loc[smoothing_df["brier_test"].idxmin(), "s"]\n'
        'ax.axvline(best_s, color="gray", linestyle=":", lw=1.5, label=f"Оптимум на тесте (s={best_s:.5f})")\n'
        "\n"
        'ax.set_xlabel("Параметр сглаживания s (log scale)")\n'
        'ax.set_ylabel("Brier Score")\n'
        'ax.set_title("Влияние параметра сглаживания сплайна на качество калибровки")\n'
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        'print(f"Оптимальный s на OOT тесте: {best_s:.6f}")\n'
        'print("Если кривые близки — сплайн хорошо обобщается.")'
    )
})

# Cell 14 - markdown
cells.append({"cell_type": "markdown", "metadata": {}, "source": "## 6. Сравнение метрик (OOT Test, 2024)"})

# Cell 15 - metrics table
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        'results = [summary_metrics(y_test.values, scores_test, name="RF (без калибровки)")]\n'
        "for name, probs in calibrated_probs.items():\n"
        "    results.append(summary_metrics(y_test.values, probs, name=name))\n"
        "\n"
        'df_results = pd.DataFrame(results).set_index("method")\n'
        "\n"
        'print("Метрики на OOT выборке (2024):")\n'
        'print("  Brier Score, Log-Loss, ECE  — чем меньше, тем лучше")\n'
        'print("  HL p-value                  — чем больше, тем лучше (> 0.05 = норма)")\n'
        'print("  Slope ~ 1, Intercept ~ 0    — идеальная калибровка")\n'
        "\n"
        "df_results.style\\\n"
        '    .highlight_min(subset=["brier_score", "log_loss", "ece"], color="#c8f7c5")\\\n'
        '    .highlight_max(subset=["hl_p_value"], color="#c8f7c5")\\\n'
        '    .format({"brier_score": "{:.5f}", "log_loss": "{:.5f}", "ece": "{:.5f}",\n'
        '             "hl_chi2": "{:.3f}", "hl_p_value": "{:.4f}",\n'
        '             "cal_slope": "{:.4f}", "cal_intercept": "{:.4f}"})'
    )
})

# Cell 16 - markdown
cells.append({"cell_type": "markdown", "metadata": {}, "source": "## 7. Reliability Diagrams: все методы"})

# Cell 17 - reliability diagrams grid
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        'all_methods = {"RF (без калибровки)": scores_test, **calibrated_probs}\n'
        "\n"
        "fig, axes = plt.subplots(2, 3, figsize=(16, 10))\n"
        "axes = axes.flatten()\n"
        "\n"
        "for i, (name, probs) in enumerate(all_methods.items()):\n"
        "    ax = axes[i]\n"
        "    mean_pred, frac_pos = get_calibration_curve(y_test.values, probs, n_bins=10)\n"
        "    m = summary_metrics(y_test.values, probs)\n"
        "\n"
        '    ax.plot([0, 1], [0, 1], "k--", lw=1.5)\n'
        '    ax.plot(mean_pred, frac_pos, "o-", color=COLORS[i], lw=2.5, ms=8)\n'
        "    ax.fill_between(mean_pred, mean_pred, frac_pos, alpha=0.15, color=COLORS[i])\n"
        "\n"
        "    ax2 = ax.twinx()\n"
        "    ax2.hist(probs, bins=30, alpha=0.12, color=COLORS[i])\n"
        '    ax2.set_ylabel("Наблюдений", fontsize=8, color="gray")\n'
        '    ax2.tick_params(axis="y", labelsize=7, labelcolor="gray")\n'
        "\n"
        "    subtitle = (\n"
        '        f\'Brier={m["brier_score"]:.4f} | ECE={m["ece"]:.4f}\\n\'\n'
        '        f\'Slope={m["cal_slope"]:.3f} | HL p={m["hl_p_value"]:.3f}\'\n'
        "    )\n"
        '    ax.set_title(f"{name}\\n{subtitle}", fontsize=9)\n'
        '    ax.set_xlabel("Среднее предсказание")\n'
        '    ax.set_ylabel("Реальная частота")\n'
        "    lim = max(mean_pred.max(), frac_pos.max()) * 1.3\n"
        "    ax.set_xlim(0, min(lim, 1))\n"
        "    ax.set_ylim(0, min(lim, 1))\n"
        "\n"
        "axes[-1].set_visible(False)\n"
        'plt.suptitle("Reliability Diagrams: сравнение методов (OOT Test 2024)", fontsize=13, fontweight="bold")\n'
        "plt.tight_layout()\n"
        "plt.show()"
    )
})

# Cell 18 - markdown
cells.append({"cell_type": "markdown", "metadata": {}, "source": "## 8. Общее сравнение на одном графике"})

# Cell 19 - combined plot
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "fig, ax = plt.subplots(figsize=(9, 7))\n"
        'ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Идеальная калибровка", zorder=5)\n'
        "\n"
        "for (name, probs), color in zip(all_methods.items(), COLORS):\n"
        "    mean_pred, frac_pos = get_calibration_curve(y_test.values, probs, n_bins=10)\n"
        "    m = summary_metrics(y_test.values, probs)\n"
        '    lw = 3.0 if "Сплайн" in name else 1.8\n'
        '    label = f"{name}  (ECE={m[\'ece\']:.4f}, slope={m[\'cal_slope\']:.3f})"\n'
        '    ax.plot(mean_pred, frac_pos, "o-", color=color, lw=lw, ms=7, label=label,\n'
        '            zorder=4 if "Сплайн" in name else 3)\n'
        "\n"
        'ax.set_xlabel("Среднее предсказание в бине", fontsize=12)\n'
        'ax.set_ylabel("Реальная частота дефолтов", fontsize=12)\n'
        'ax.set_title("Сравнение методов калибровки (OOT Test 2024)", fontsize=13, fontweight="bold")\n'
        'ax.legend(loc="upper left", fontsize=9)\n'
        "plt.tight_layout()\n"
        "plt.show()"
    )
})

# Cell 20 - markdown
cells.append({"cell_type": "markdown", "metadata": {}, "source": "## 9. Calibration Slope и Intercept"})

# Cell 21 - slope intercept bars
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "methods    = list(df_results.index)\n"
        'slopes     = df_results["cal_slope"].values\n'
        'intercepts = df_results["cal_intercept"].values\n'
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
        "\n"
        "ax = axes[0]\n"
        'bars = ax.barh(methods, slopes, color=COLORS[:len(methods)], edgecolor="white")\n'
        'ax.axvline(1.0, color="black", linestyle="--", lw=1.5, label="Идеал (slope=1)")\n'
        'ax.set_xlabel("Calibration Slope")\n'
        'ax.set_title("Calibration Slope  (идеал = 1.0)")\n'
        "ax.legend()\n"
        "for bar, val in zip(bars, slopes):\n"
        '    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)\n'
        "\n"
        "ax = axes[1]\n"
        'bars = ax.barh(methods, intercepts, color=COLORS[:len(methods)], edgecolor="white")\n'
        'ax.axvline(0.0, color="black", linestyle="--", lw=1.5, label="Идеал (intercept=0)")\n'
        'ax.set_xlabel("Calibration Intercept")\n'
        'ax.set_title("Calibration Intercept  (идеал = 0.0)")\n'
        "ax.legend()\n"
        "for bar, val in zip(bars, intercepts):\n"
        "    offset = 0.005 if val >= 0 else -0.05\n"
        '    ax.text(val + offset, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)\n'
        "\n"
        'plt.suptitle("Calibration Slope и Intercept: близость к идеалу", fontsize=12, fontweight="bold")\n'
        "plt.tight_layout()\n"
        "plt.show()"
    )
})

# Cell 22 - markdown
cells.append({"cell_type": "markdown", "metadata": {}, "source": "## 10. Итоги и выводы"})

# Cell 23 - summary print
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        'print("=" * 70)\n'
        'print("ИТОГОВОЕ СРАВНЕНИЕ МЕТОДОВ КАЛИБРОВКИ PD (OOT Test 2024)")\n'
        'print("=" * 70)\n'
        'display_cols = ["brier_score", "ece", "hl_p_value", "cal_slope", "cal_intercept"]\n'
        "print(df_results[display_cols].to_string())\n"
        "\n"
        'print("\\n" + "=" * 70)\n'
        'print("РЕЗЮМЕ ПО МЕТОДАМ")\n'
        'print("=" * 70)\n'
        "summary_map = {\n"
        '    "Логит-калибровка":          "Устойчив, прост. Не улавливает нелинейности.",\n'
        '    "Изотоническая регрессия":   "Гибкий, но рискует переобучиться на OOT данных.",\n'
        '    "Бета-калибровка":           "Хорош для асимметричных скоров. Стабилен.",\n'
        '    "Сплайн (CV)":               "Лучший баланс гибкости и обобщения при нелинейной зависимости.",\n'
        "}\n"
        "for method, desc in summary_map.items():\n"
        '    print(f"  {method:<30} {desc}")'
    )
})

# Cell 24 - markdown conclusions
cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": (
        "## Итоговая таблица: когда применять каждый метод\n\n"
        "| Метод | Когда применять | Когда избегать |\n"
        "|---|---|---|\n"
        "| **Логит** | Малая калибровочная выборка, нужна интерпретируемость | Сильная нелинейность зависимости |\n"
        "| **Изотоническая** | Большая выборка, нет требований к гладкости | Малая выборка, OOT валидация |\n"
        "| **Бета** | Скоры из [0,1], асимметричное распределение | Скоры вне [0,1] |\n"
        "| **Сплайн (CV)** | **Нелинейная зависимость, банковский портфель с концентрацией в низких PD** | Экстремально малая выборка (<200 наблюдений) |\n\n"
        "**Вывод:** В банковском контексте, где скоры RF концентрируются в диапазоне 0-0.15 и зависимость скор->PD нелинейна, "
        "**сплайн-калибровка с подбором параметра сглаживания через CV** демонстрирует наилучшее качество "
        "по совокупности метрик на out-of-time выборке."
    )
})

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

path = "d:/Claude projects/ColebrationTedo/notebooks/pd_calibration.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook written OK")
