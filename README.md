# PD Calibration And Capital Impact Lab

Проект про калибровку Probability of Default (PD) в банковском кредитном риске: сравнение методов калибровки, out-of-time validation, диагностика качества вероятностей и оценка эффекта на Basel-style RWA / capital.

## Что внутри

- `notebooks/pd_calibration.ipynb` - основной исследовательский notebook.
- `src/calibrators.py` - методы калибровки PD: logit/Platt, isotonic, beta, spline.
- `src/metrics.py` - метрики калибровки, дискриминации и стабильности.
- `src/capital.py` - Basel-style расчет expected loss, unexpected-loss capital и RWA.
- `data/generate_data.py` - генерация синтетического кредитного портфеля.
- `docs/references.md` - статьи и регуляторные источники, на которые опирается проект.

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
