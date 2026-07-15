# PD Calibration And Capital Impact Lab

Проект про калибровку Probability of Default (PD) в банковском кредитном риске: сравнение методов калибровки, out-of-time validation, диагностика качества вероятностей и оценка эффекта на Basel-style RWA / capital.

## Что внутри

- `notebooks/pd_calibration.ipynb` - основной исследовательский notebook.
- `notebooks/monte_carlo.ipynb` - Monte Carlo проверка устойчивости калибровок к составу портфеля: whole-model тест калибровки (само-калибровка) / PD / EL / UL / EL+UL / RWA по сценариям, сводка mean/min/max/range, межметодный разброс EL+UL внутри сценария и диагностика по грейдам мастер-шкалы.
- `notebooks/monte_carlo.py` - вычислительный движок Monte Carlo (ресемплинг ~80% без переобучения модели), используется ноутбуком выше.
- `src/calibrators.py` - методы калибровки PD: logit/Platt, isotonic, beta, spline.
- `src/metrics.py` - метрики калибровки, дискриминации и стабильности.
- `src/capital.py` - Basel-style расчет expected loss, unexpected-loss capital и RWA.
- `data/generate_data.py` - генерация синтетического кредитного портфеля.
- `scripts/spline_parameter_search.py` - data-driven подбор параметров сплайн-калибраторов.
- `docs/monte_carlo_scenarios.csv` - полная выгрузка всех 1000 Monte Carlo сценариев (5000 строк) для отдельного просмотра.
- `docs/monte_carlo_cross_method_spread.csv` - межметодный разброс EL + UL внутри каждого сценария (1000 строк): самый дешёвый / дорогой метод, abs и rel разница.
- `docs/references.md` - статьи и регуляторные источники, на которые опирается проект.
- `docs/spline_methodology.md` - разбор методологии сплайн-калибраторов и обоснование переработки 2026-07.
- `docs/calibration_change_report.md` - журнал методологических изменений калибровки.

## Основная идея

Базовая ML-модель может хорошо ранжировать заемщиков, но ее вероятности дефолта не обязательно хорошо откалиброваны. Для банковской практики это важно, потому что PD влияет не только на качество риск-оценки, но и на резервы, RWA и требуемый капитал.

В проекте сравниваются несколько подходов:

- логит-калибровка / Platt scaling;
- изотоническая регрессия;
- beta calibration;
- монотонная spline-калибровка;
- двухшаговая схема logit + monotone spline, вдохновленная практикой PD-калибровки и ICAS-style моделей.

## Методологическая рамка

Статистическое качество оценивается через Brier Score, Log-Loss, ECE, Hosmer-Lemeshow, calibration slope/intercept, AUC/Gini/KS и bootstrap confidence intervals.

Экономический эффект оценивается отдельно: calibrated PD подаются в Basel-style IRB расчет, где считаются expected loss, unexpected-loss capital, RWA и capital impact относительно некалиброванной модели.

Подробный список источников лежит в `docs/references.md`.

## Важное ограничение

Проект использует синтетический портфель и упрощенную IRB-style формулу. Результаты подходят для исследования чувствительности капитала к калибровке PD, но не являются готовым регуляторным расчетом для отчетности банка.
