# Week 5 - Machine Learning and Model Evaluation
# YuvaIntern Data Analytics Internship

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

DATA_PATH = "dataset/Sample - Superstore.csv"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, encoding="latin1")

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# 3. Data Preprocessing
# --------------------------------------------------

print("\nChecking missing values...")

print(df.isnull().sum())


# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")


# Create useful date-based features
df["Order_Year"] = df["Order Date"].dt.year
df["Order_Month"] = df["Order Date"].dt.month


# Remove rows where target is missing
df = df.dropna(subset=["Sales"])


# --------------------------------------------------
# 4. Feature Selection
# --------------------------------------------------

features = [
    "Ship Mode",
    "Segment",
    "Region",
    "Category",
    "Sub-Category",
    "Quantity",
    "Discount",
    "Order_Year",
    "Order_Month"
]

target = "Sales"

X = df[features]
y = df[target]


print("\nSelected Features:")
print(features)

print("\nTarget:")
print(target)


# --------------------------------------------------
# 5. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 6. Preprocessing Pipeline
# --------------------------------------------------

categorical_features = [
    "Ship Mode",
    "Segment",
    "Region",
    "Category",
    "Sub-Category"
]

numeric_features = [
    "Quantity",
    "Discount",
    "Order_Year",
    "Order_Month"
]


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# --------------------------------------------------
# 7. Linear Regression Model
# --------------------------------------------------

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)


print("\nTraining Linear Regression...")

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)


# --------------------------------------------------
# 8. Linear Regression Evaluation
# --------------------------------------------------

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, linear_predictions)


print("\nLinear Regression Results")
print("-------------------------")
print("MAE :", linear_mae)
print("MSE :", linear_mse)
print("RMSE:", linear_rmse)
print("R2  :", linear_r2)


# --------------------------------------------------
# 9. Random Forest Model
# --------------------------------------------------

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


print("\nTraining Random Forest...")

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)


# --------------------------------------------------
# 10. Random Forest Evaluation
# --------------------------------------------------

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_predictions)


print("\nRandom Forest Results")
print("---------------------")
print("MAE :", rf_mae)
print("MSE :", rf_mse)
print("RMSE:", rf_rmse)
print("R2  :", rf_r2)


# --------------------------------------------------
# 11. Cross Validation
# --------------------------------------------------

print("\nPerforming 5-Fold Cross Validation...")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_scores = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)


print("\nCross Validation R2 Scores:")
print(cv_scores)

print("Mean CV R2:", cv_scores.mean())


# --------------------------------------------------
# 12. Hyperparameter Tuning
# --------------------------------------------------

print("\nStarting Grid Search...")


param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}


grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X_train, y_train)


print("\nBest Parameters:")
print(grid_search.best_params_)


print("\nBest CV RMSE:")
print(-grid_search.best_score_)


# --------------------------------------------------
# 13. Tuned Model Evaluation
# --------------------------------------------------

best_model = grid_search.best_estimator_

tuned_predictions = best_model.predict(X_test)


tuned_mae = mean_absolute_error(
    y_test,
    tuned_predictions
)

tuned_mse = mean_squared_error(
    y_test,
    tuned_predictions
)

tuned_rmse = np.sqrt(tuned_mse)

tuned_r2 = r2_score(
    y_test,
    tuned_predictions
)


print("\nTuned Random Forest Results")
print("---------------------------")
print("MAE :", tuned_mae)
print("MSE :", tuned_mse)
print("RMSE:", tuned_rmse)
print("R2  :", tuned_r2)


# --------------------------------------------------
# 14. Model Comparison
# --------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Tuned Random Forest"
    ],
    "MAE": [
        linear_mae,
        rf_mae,
        tuned_mae
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse,
        tuned_rmse
    ],
    "R2": [
        linear_r2,
        rf_r2,
        tuned_r2
    ]
})


print("\nModel Comparison:")
print(comparison)


comparison.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)


# --------------------------------------------------
# 15. Actual vs Predicted Plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    tuned_predictions,
    alpha=0.5
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted.png"
    ),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 16. Residual Plot
# --------------------------------------------------

residuals = y_test - tuned_predictions

plt.figure(figsize=(8, 6))

plt.scatter(
    tuned_predictions,
    residuals,
    alpha=0.5
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "residual_plot.png"
    ),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 17. Model Comparison Visualization
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.barplot(
    data=comparison,
    x="Model",
    y="RMSE"
)

plt.title("Model Comparison - RMSE")
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.png"
    ),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 18. Save Tuning Results
# --------------------------------------------------

tuning_results = pd.DataFrame(
    grid_search.cv_results_
)

tuning_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "tuning_results.csv"
    ),
    index=False
)


# --------------------------------------------------
# 19. Save Predictions
# --------------------------------------------------

prediction_output = pd.DataFrame({
    "Actual_Sales": y_test.values,
    "Predicted_Sales": tuned_predictions
})

prediction_output.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "predictions.csv"
    ),
    index=False
)


print("\n-----------------------------------")
print("Week 5 ML analysis completed!")
print("All outputs saved in the output folder.")
print("-----------------------------------")