# Project Summary: House Price Prediction — Full Regression Pipeline

## Objective
Apply and compare every major regression technique on a single dataset
(California Housing) to understand not just *which* model performs best,
but *why* — and specifically, when regularization helps and when it doesn't.

## Dataset
California Housing dataset (1990 U.S. Census), ~20,000 rows, 9 numeric
features plus one categorical feature (`ocean_proximity`, one-hot encoded).
Target: `median_house_value` (renamed `Price`).

## Results

| # | Model | RMSE | R² | Key Setting |
|---|---|---|---|---|
| 1 | Simple Linear Regression | 84,976.53 | 0.4720 | feature = `median_income` |
| 2 | Multiple Linear Regression | 69,297.72 | 0.6488 | all 12 features |
| 3 | Polynomial Regression (deg=2) | 84,652.79 | 0.4760 | feature = `median_income` |
| 4 | Ridge Regression | 69,298.76 | 0.6488 | alpha = 10.0 |
| 5 | Lasso Regression | 69,297.74 | 0.6488 | alpha = 1.0, 0 features dropped |
| 6 | Elastic Net | 69,300.22 | 0.6488 | alpha = 0.01, l1_ratio = 0.9 |
| 7 | Logistic Regression (bonus) | — | — | Accuracy = 0.8424 (classification) |
| 8 | Random Forest Regression | **48,757.92** | **0.8262** | n_estimators = 200 |

*(RMSE is in US dollars; lower is better. R² is the proportion of price variance explained; higher is better, max 1.0.)*

## Key Findings

**1. More features help substantially, but only up to a point.**
Moving from one feature (`median_income` alone) to all 12 features cut RMSE
by ~$15,700 and raised R² from 0.47 to 0.65 (Models 1→2). This was the
single largest jump from adding information, until Random Forest.

**2. The income–price relationship is already close to linear.**
Adding a squared term (Polynomial Regression) barely moved the needle
(R² 0.4720 → 0.4760, Model 1 vs. 3) — evidence that curvature isn't the
missing piece for this particular feature.

**3. Regularization (Ridge, Lasso, Elastic Net) provided no measurable
improvement over plain Multiple Linear Regression** (Models 2, 4, 5, 6 all
land at RMSE ≈ 69,300 / R² ≈ 0.6488). This is not a failed experiment — it's
a legitimate finding: this dataset does not suffer from severe
multicollinearity or irrelevant/noisy features, which is precisely the
problem these three techniques are designed to solve. Lasso confirmed this
directly by not dropping a single feature even at its best cross-validated
alpha.

**4. Regularization did fix one specific, real problem: an unstable
coefficient.** Plain Linear Regression assigned `ocean_proximity_ISLAND`
an implausible +$213,653 coefficient — almost certainly an artifact of
that category having very few data points. Both Ridge (+$2,900) and Random
Forest's feature importance (ranked dead last, 0.0004) independently
confirmed this feature carries far less real signal than plain Linear
Regression suggested. This didn't move the *overall* RMSE/R² (island
houses are too rare to matter at that scale), but it's a clean, concrete
example of regularization doing its job correctly.

**5. Random Forest decisively outperformed every linear-family model.**
RMSE dropped by roughly $20,500 and R² rose from ~0.65 to 0.83 — by far the
largest improvement in the whole project. This indicates the true
relationship between location, income, and price involves nonlinearities
and feature interactions (e.g., income likely matters differently in
coastal vs. inland areas) that no linear model — regularized or not — can
represent, no matter how well-tuned.

**6. Reframing the problem as classification worked well.**
Predicting "above vs. below median price" instead of an exact dollar
amount reached 84.2% accuracy, with balanced precision/recall across both
classes (~0.83–0.85 each) — showing the same underlying signal
(`median_income` especially) is strong enough to answer a simpler,
binary version of the question with high reliability.

**7. `median_income` was the single most consistent predictor across
every model in the project** — linear or nonlinear, regularized or not —
making it the most robust, defensible finding of the whole analysis.

## Overall Conclusion
On this dataset, model *family* mattered far more than regularization
*strength*. Every linear approach — however penalized — hit the same
ceiling around R²≈0.65, because the underlying price relationship is
genuinely nonlinear. Ridge, Lasso, and Elastic Net earned their place in
the pipeline not by boosting accuracy, but by demonstrating exactly when
regularization does and doesn't matter, and by correcting one clearly
unstable coefficient. Random Forest's clear win is the project's central
takeaway: capturing nonlinear structure and feature interactions was more
valuable here than any amount of linear regularization.
