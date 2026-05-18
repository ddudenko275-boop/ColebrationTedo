# PD Calibration and Capital Impact Lab

This project studies Probability of Default (PD) calibration in banking credit risk. It compares calibration methods, out-of-time validation, probability-quality diagnostics, binomial z-stat checks, and the effect of calibrated PD on Basel-style RWA and capital.

## Contents

- `notebooks/pd_calibration.ipynb` - main research notebook.
- `notebooks/pd_calibration_legacy.ipynb` - archived legacy note that points to the current notebook.
- `src/calibrators.py` - PD calibration methods: logit/Platt, isotonic, beta, monotone spline, and French spline.
- `src/metrics.py` - calibration, discrimination, and stability metrics.
- `src/capital.py` - Basel-style expected loss, unexpected-loss capital, and RWA calculations.
- `data/generate_data.py` - synthetic credit portfolio generation.
- `docs/references.md` - academic and regulatory references used by the project.

## Core Idea

A base ML model can rank borrowers well, but its default probabilities may still be poorly calibrated. In banking practice, this matters because PD affects not only risk-score quality, but also reserves, RWA, and required capital.

The project compares several approaches:

- logit calibration / Platt scaling;
- isotonic regression;
- beta calibration;
- monotone spline calibration;
- a two-stage logit + monotone spline scheme inspired by practical PD calibration and ICAS-style models.

## Methodological Frame

Statistical quality is assessed with Brier Score, Log-Loss, ECE, Hosmer-Lemeshow, calibration slope/intercept, AUC/Gini/KS, bootstrap confidence intervals, and binomial z-stat diagnostics by calibration bins and fixed master-scale buckets.

The economic effect is assessed separately: calibrated PD values are passed into a Basel-style IRB calculation to estimate expected loss, unexpected-loss capital, RWA, and capital impact relative to logit calibration.

The full source list is available in `docs/references.md`.

## Important Limitation

The project uses a synthetic portfolio and a simplified IRB-style formula. Results are suitable for studying capital sensitivity to PD calibration, but they are not a production regulatory reporting calculation for a bank.
