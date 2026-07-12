# Источники и методологическая база

Этот проект опирается на два типа источников: банковскую методологию PD/RWA и литературу по калибровке вероятностей бинарных классификаторов. Такое разделение важно: качество калибровки оценивается статистически, а эффект на капитал интерпретируется через Basel-style расчет RWA.

## Regulatory Capital / IRB

- Basel Committee on Banking Supervision. **Basel Framework**, разделы `CRE30-CRE36`: IRB-подход, PD/LGD/EAD/M, риск-весовые функции, expected loss и unexpected loss.  
  https://www.bis.org/baselframework/BaselFramework.pdf

- Basel Committee on Banking Supervision. **An Explanatory Note on the Basel II IRB Risk Weight Functions**.  
  https://www.bis.org/bcbs/irbriskweight.htm

- European Banking Authority. **Guidelines on PD estimation, LGD estimation and treatment of defaulted exposures**.  
  https://www.eba.europa.eu/publications-and-media/press-releases/eba-publishes-final-guidelines-estimation-risk-parameters

## Banking PD Calibration And Rating Systems

- Tasche, D. **The art of probability-of-default curve calibration**. 2012/2013.  
  https://arxiv.org/abs/1212.3716

- Tasche, D. **Probability-of-default curve calibration and validation of internal rating systems**. IFC Bulletin, BIS. 2016.  
  https://www.bis.org/ifc/publ/ifcb43_zd.pdf

- Banque de France. **Return on Investment on AI: The Case of Capital Requirement**. Working Paper no. 809.  
  https://www.banque-france.fr/en/publications-and-statistics/publications/return-investment-ai-case-capital-requirement

- Banca d'Italia. **The In-house Credit Assessment System of Banca d'Italia**. Occasional Paper no. 586.  
  https://www.bancaditalia.it/pubblicazioni/qef/2020-0586/index.html

## Probability Calibration Methods

- Gupta et al. **Calibration of Neural Networks using Splines**. 2021.  
  https://www.mensink.nu/pubs/gupta21iclr.pdf

- Kull, Silva Filho, Flach. **Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers**. AISTATS, 2017.  
  https://proceedings.mlr.press/v54/kull17a

- Kull, Silva Filho, Flach. **Beyond Sigmoids: How to obtain well-calibrated probabilities from binary classifiers with beta calibration**. 2017.  
  https://research-information.bris.ac.uk/en/publications/beyond-sigmoids-how-to-obtain-well-calibrated-probabilities-from-

- **Calibration Techniques for Binary Classification Problems: A Comparative Analysis**. 2019.  
  https://www.scitepress.org/Papers/2019/81655/81655.pdf

- **On the appropriateness of Platt scaling in classifier calibration**. 2021.  
  https://doi.org/10.1016/j.is.2020.101641

## Spline Methodology (переработка 2026-07)

Подробный разбор — в `docs/spline_methodology.md`; эти источники обосновывают текущую конструкцию сплайн-калибраторов.

- Jiang, X., Osl, M., Kim, J., Ohno-Machado, L. **Smooth Isotonic Regression: A New Method to Calibrate Predictive Models**. AMIA Summits on Translational Science, 2011. Основа связки isotonic → PCHIP в `MonotoneSplineCalibrator`.  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3248752/

- Lucena, B. **Spline-Based Probability Calibration**. 2018. Современная альтернатива (регуляризованная логрегрессия на сплайн-базисе); реализация — пакет `ml-insights`.  
  https://arxiv.org/abs/1809.07751

- Пуарье, Д. **Эконометрия структурных изменений (с особым акцентом на сплайн-функции)**. М.: Финансы и статистика, 1981 (ориг. 1976). Гл. 7 — статистическое тестирование положения узла; §8.5 — правила Уолда для выбора узлов и предостережение против подбора узлов под желаемый результат.

- Ильясов, Р.Х. **Сплайн-моделирование и анализ взаимосвязей в экономике при возможном наличии точек переключения регрессии**. НТВ СПбГПУ. Экономические науки, 11(4), 2018. Обоснование `knot_diagnostics()`: поиск точек переключения через первую производную сплайна.  
  https://doi.org/10.18721/JE.11412

- **Численные методы для экономических расчетов** (учебное пособие). Формальные определения линейного/параболического/кубического сплайна и понятие дефекта сплайна — обоснование выбора PCHIP (дефект 1, монотонность важнее гладкости второй производной).

## Credit Scoring, Splines, And Class Imbalance

- **Approaches for credit scorecard calibration: An empirical analysis**. 2017.  
  https://doi.org/10.1016/j.knosys.2017.07.034

- **Multinomial Logistic Regression and Spline Regression for Credit Risk Modelling**. 2018.  
  https://doi.org/10.1088/1742-6596/1108/1/012019

- **Credit scoring using neural networks and SURE posterior probability calibration**. 2021.  
  https://arxiv.org/abs/2107.07206

- **Using Platt's scaling for calibration after undersampling: limitations and how to address them**. 2024.  
  https://arxiv.org/abs/2410.18144

- **Deep learning model calibration for improving performance in class-imbalanced medical datasets**. 2022.  
  https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0262838

## Как эти источники используются в проекте

- Basel/EBA задают язык расчета капитала: `PD`, `LGD`, `EAD`, `M`, `K`, `RWA`, expected loss и unexpected loss.
- Tasche и BIS/IFC источники задают банковскую постановку калибровки PD-кривой и внутренней рейтинговой системы.
- Platt, beta, isotonic, spline и GAM-источники обосновывают набор сравниваемых калибраторов.
- Источники по imbalance помогают объяснить, почему дефолты нужно анализировать отдельно от обычных balanced classification задач.
