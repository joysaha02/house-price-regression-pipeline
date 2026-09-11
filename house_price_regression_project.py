"""
=====================================================================
FULL REGRESSION PIPELINE PROJECT
House Price Prediction — California Housing Dataset
=====================================================================

This single project walks through every major regression technique,
each one motivated by a real limitation of the previous step:

  1. Simple Linear Regression      -> baseline, one feature
  2. Multiple Linear Regression    -> use all features
  3. Polynomial Regression         -> capture nonlinearity
  4. Ridge Regression (L2)         -> handle multicollinearity
  5. Lasso Regression (L1)         -> automatic feature selection
  6. Elastic Net (L1 + L2)         -> best of both, correlated features
  7. Logistic Regression (bonus)   -> classification twist
  8. Random Forest Regression      -> nonlinear ensemble benchmark

Run this as a script, or copy each numbered section into its own
Jupyter notebook cell.
=====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, classification_report
)

RANDOM_STATE = 42
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# =====================================================================
# 0. LOAD & EXPLORE DATA
# =====================================================================
# California Housing dataset (same data as sklearn's fetch_california_housing,
# loaded here from a CSV mirror). Includes one categorical column
# (ocean_proximity) and a handful of missing values in total_bedrooms,
# which is realistic and gives us a small cleaning step to do.
CSV_PATH = "housing.csv"
df = pd.read_csv(CSV_PATH)
df.rename(columns={"median_house_value": "Price"}, inplace=True)

# Drop rows with missing values (only ~1% of rows affected)
df = df.dropna()

# One-hot encode the categorical column so all downstream models
# (which expect numeric input) can use it.
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(df.head())
print("\nShape:", df.shape)
print("\nSummary stats:\n", df.describe())

# Check correlation with target -> tells us which single feature to
# use for the simple linear regression step, and hints at
# multicollinearity among predictors for later (Ridge/Lasso).
corr = df.corr(numeric_only=True)["Price"].sort_values(ascending=False)
print("\nCorrelation with Price:\n", corr)

FEATURES = [c for c in df.columns if c != "Price"]
X = df[FEATURES]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

results = {}  # collects (RMSE, R2, notes) per model for the final table


def evaluate(name, y_true, y_pred, notes=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    results[name] = {"RMSE": rmse, "R2": r2, "Notes": notes}
    print(f"\n[{name}]  RMSE = {rmse:.4f}   R2 = {r2:.4f}   {notes}")
    return rmse, r2


# =====================================================================
# 1. SIMPLE LINEAR REGRESSION (one feature)
# =====================================================================
# Use the single most correlated feature with Price.
best_single_feature = corr.drop("Price").abs().idxmax()
print(f"\nUsing '{best_single_feature}' as the single feature "
      f"(highest correlation with Price).")

X_train_simple = X_train[[best_single_feature]]
X_test_simple = X_test[[best_single_feature]]

simple_lr = LinearRegression()
simple_lr.fit(X_train_simple, y_train)
y_pred_simple = simple_lr.predict(X_test_simple)
evaluate("1. Simple Linear Regression", y_test, y_pred_simple,
          notes=f"feature = {best_single_feature}")

# =====================================================================
# 2. MULTIPLE LINEAR REGRESSION (all features)
# =====================================================================
multi_lr = LinearRegression()
multi_lr.fit(X_train, y_train)
y_pred_multi = multi_lr.predict(X_test)
evaluate("2. Multiple Linear Regression", y_test, y_pred_multi,
          notes="all features")

coef_table = pd.Series(multi_lr.coef_, index=FEATURES).sort_values(
    key=abs, ascending=False
)
print("\nMultiple Linear Regression coefficients (sorted by magnitude):")
print(coef_table)

# =====================================================================
# 3. POLYNOMIAL REGRESSION
# =====================================================================
# Apply polynomial expansion to the same top feature used in step 1,
# to check whether the relationship is actually curved rather than
# linear. Degree 2 is a sensible starting point; higher degrees risk
# overfitting fast on this dataset.
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_simple)
X_test_poly = poly.transform(X_test_simple)

poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)
y_pred_poly = poly_lr.predict(X_test_poly)
evaluate("3. Polynomial Regression (deg=2)", y_test, y_pred_poly,
          notes=f"feature = {best_single_feature}, degree=2")

# =====================================================================
# 4. RIDGE REGRESSION (L2 regularization)
# =====================================================================
# Standardize first -- regularized models are sensitive to feature
# scale, unlike plain OLS.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_cv = GridSearchCV(
    Ridge(random_state=RANDOM_STATE),
    param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    scoring="neg_mean_squared_error",
    cv=5,
)
ridge_cv.fit(X_train_scaled, y_train)
best_ridge = ridge_cv.best_estimator_
y_pred_ridge = best_ridge.predict(X_test_scaled)
evaluate("4. Ridge Regression", y_test, y_pred_ridge,
          notes=f"best alpha={ridge_cv.best_params_['alpha']}")

# =====================================================================
# 5. LASSO REGRESSION (L1 regularization / feature selection)
# =====================================================================
lasso_cv = GridSearchCV(
    Lasso(random_state=RANDOM_STATE, max_iter=10000),
    param_grid={"alpha": [0.001, 0.01, 0.1, 1.0]},
    scoring="neg_mean_squared_error",
    cv=5,
)
lasso_cv.fit(X_train_scaled, y_train)
best_lasso = lasso_cv.best_estimator_
y_pred_lasso = best_lasso.predict(X_test_scaled)

dropped = [f for f, c in zip(FEATURES, best_lasso.coef_) if c == 0]
evaluate("5. Lasso Regression", y_test, y_pred_lasso,
          notes=f"best alpha={lasso_cv.best_params_['alpha']}, "
                f"dropped features: {dropped or 'none'}")

# =====================================================================
# 6. ELASTIC NET (L1 + L2 combined)
# =====================================================================
elastic_cv = GridSearchCV(
    ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
    param_grid={
        "alpha": [0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.5, 0.9],
    },
    scoring="neg_mean_squared_error",
    cv=5,
)
elastic_cv.fit(X_train_scaled, y_train)
best_elastic = elastic_cv.best_estimator_
y_pred_elastic = best_elastic.predict(X_test_scaled)
evaluate("6. Elastic Net", y_test, y_pred_elastic,
          notes=f"best params={elastic_cv.best_params_}")

# =====================================================================
# 7. LOGISTIC REGRESSION (bonus classification twist)
# =====================================================================
# Reframe the problem: is this house's price ABOVE the median?
# Same features, different question -> natural bridge into
# classification without switching datasets.
median_price = y.median()
y_train_class = (y_train > median_price).astype(int)
y_test_class = (y_test > median_price).astype(int)

log_reg = LogisticRegression(max_iter=5000)
log_reg.fit(X_train_scaled, y_train_class)
y_pred_class = log_reg.predict(X_test_scaled)

acc = accuracy_score(y_test_class, y_pred_class)
print(f"\n[7. Logistic Regression]  Accuracy = {acc:.4f}")
print(classification_report(y_test_class, y_pred_class,
                             target_names=["Below median", "Above median"]))
results["7. Logistic Regression (classification)"] = {
    "RMSE": np.nan, "R2": np.nan, "Notes": f"Accuracy={acc:.4f}"
}

# =====================================================================
# 8. RANDOM FOREST REGRESSION (nonlinear ensemble benchmark)
# =====================================================================
rf = RandomForestRegressor(
    n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train, y_train)  # tree models don't need scaling
y_pred_rf = rf.predict(X_test)
evaluate("8. Random Forest Regression", y_test, y_pred_rf,
          notes="n_estimators=200")

importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(
    ascending=False
)
print("\nRandom Forest feature importances:")
print(importances)

# =====================================================================
# FINAL COMPARISON TABLE
# =====================================================================
print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)
results_df = pd.DataFrame(results).T
print(results_df)

results_df.to_csv("/home/claude/model_comparison_results.csv")

# =====================================================================
# OPTIONAL: quick bar chart of RMSE across regression models
# (excludes the classification-only Logistic Regression row)
# =====================================================================
plot_df = results_df.drop(index="7. Logistic Regression (classification)")
plot_df["RMSE"] = plot_df["RMSE"].astype(float)

plt.figure(figsize=(9, 5))
plt.bar(plot_df.index, plot_df["RMSE"], color="steelblue")
plt.xticks(rotation=30, ha="right")
plt.ylabel("RMSE (lower is better)")
plt.title("Model Comparison: RMSE on Test Set")
plt.tight_layout()
plt.savefig("/home/claude/rmse_comparison.png", dpi=150)
print("\nSaved chart to rmse_comparison.png")
print("Saved results table to model_comparison_results.csv")
