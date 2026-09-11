# House Price Prediction — Full Regression Pipeline

A single end-to-end project that walks through every major regression
technique on one real dataset, so each method's purpose is motivated by a
concrete limitation of the one before it — rather than treating each
algorithm as an isolated exercise.

**Dataset:** California Housing (1990 U.S. Census, ~20,000 rows)
**Task:** Predict median house value from location, income, and housing
characteristics.

---

## What's inside

| Model | RMSE | R² | Notes |
|---|---|---|---|
| Simple Linear Regression | 84,976.53 | 0.4720 | baseline, 1 feature (`median_income`) |
| Multiple Linear Regression | 69,297.72 | 0.6488 | all 12 features |
| Polynomial Regression (deg=2) | 84,652.79 | 0.4760 | tests nonlinearity on `median_income` |
| Ridge Regression | 69,298.76 | 0.6488 | L2 regularization, best alpha=10.0 |
| Lasso Regression | 69,297.74 | 0.6488 | L1 regularization, 0 features dropped |
| Elastic Net | 69,300.22 | 0.6488 | L1+L2 blend, best l1_ratio=0.9 |
| Logistic Regression (bonus) | — | Acc = 0.8424 | classification: above/below median price |
| **Random Forest Regression** | **48,757.92** | **0.8262** | nonlinear ensemble benchmark |

Full narrative and interpretation of these results is in
[`project_results_summary.md`](./project_results_summary.md).

---

## Key findings

- **Regularization (Ridge/Lasso/Elastic Net) matched plain Multiple Linear
  Regression almost exactly** — a real finding, not a failure: this
  dataset doesn't have severe multicollinearity or noisy/irrelevant
  features for those techniques to fix. Lasso confirmed this directly by
  not dropping a single feature.
- **Regularization did fix one specific problem:** an unstable
  `ocean_proximity_ISLAND` coefficient that plain Linear Regression
  inflated to +$213,653 (an artifact of that category having very few
  data points). Ridge pulled it down to +$2,900; Random Forest
  independently ranked it as the least important feature overall.
- **Random Forest decisively outperformed every linear model** (RMSE down
  ~$20,500, R² up from ~0.65 to 0.83), showing the true relationship
  between location, income, and price is nonlinear and involves feature
  interactions no linear model — regularized or not — can represent.
- **`median_income` was the most consistent predictor across every single
  model**, linear or nonlinear.

---

## Project structure

```
.
├── house_price_regression_project.py   # full pipeline, runs end-to-end
├── house_price_regression_project.ipynb (optional, if exported from Jupyter)
├── housing.csv                          # dataset
├── project_results_summary.md           # full write-up of findings
├── model_comparison_results.csv         # final RMSE/R2 table
├── rmse_comparison.png                  # bar chart of all models
└── README.md
```

---

## How to run

**1. Clone the repo and install dependencies:**
```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
```

Or install directly:
```bash
pip install scikit-learn pandas numpy matplotlib
```

**2. Run the pipeline:**
```bash
python house_price_regression_project.py
```

This trains and evaluates all 8 models end-to-end, prints results to the
console, and saves `model_comparison_results.csv` and
`rmse_comparison.png`.

Or open `house_price_regression_project.ipynb` in Jupyter to step through
it cell by cell.

---

## Dataset

`housing.csv` is the California Housing dataset (1990 U.S. Census),
popularized via Aurélien Géron's *Hands-On Machine Learning*. Columns:

- `longitude`, `latitude` — geographic coordinates
- `housing_median_age` — median age of houses in the block
- `total_rooms`, `total_bedrooms`, `population`, `households` — block-level counts
- `median_income` — median income of the block (tens of thousands of USD)
- `ocean_proximity` — categorical: distance/relation to the ocean
- `median_house_value` — **target**, renamed `Price` in this project

A handful of rows (~1%) with missing `total_bedrooms` are dropped; the
categorical column is one-hot encoded.

---

## Requirements

```
scikit-learn
pandas
numpy
matplotlib
```

## Possible extensions

- Apply `PolynomialFeatures` to all features (not just one) before
  Ridge/Lasso, to see if regularization becomes more useful with many
  correlated polynomial terms.
- Add engineered ratio features (`rooms_per_household`,
  `bedrooms_per_room`) — a well-known trick for this dataset that often
  helps linear models close some of the gap with Random Forest.
- Try Gradient Boosting (e.g., XGBoost/LightGBM) as a further nonlinear
  benchmark.

## License

Add a license of your choice (e.g., MIT) if you plan to make this repo public.
