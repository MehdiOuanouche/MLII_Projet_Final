# script.py

# 1/ Import libraries
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# 2/ Load the datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Preprocessing: Remove the target variable from the train dataset
train0 = train.drop("loan_status", axis=1)
combined = pd.concat([test, train0], axis=0)

# 3/ Data Cleaning
# Remove individuals older than 65 years
combined = combined[combined['person_age'] <= 65]

# Fill missing values in numerical columns with column mean
combined = combined.fillna(combined.mean(numeric_only=True))

# Remove duplicates
combined = combined.drop_duplicates()

# 4/ Define features and target
features = combined[['person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
                     'loan_intent', 'loan_grade', 'loan_int_rate', 'loan_percent_income',
                     'cb_person_default_on_file', 'cb_person_cred_hist_length']]

# Assuming 'loan_amnt' is the target variable
target = combined['loan_amnt']

# 5/ Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 6/ Preprocessing pipeline
numerical_features = ['person_age', 'person_income', 'person_emp_length',
                      'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 7/ Model training and evaluation
def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{model_name} - Mean Squared Error: {mse:.4f}')
    print(f'{model_name} - R² Score: {r2:.4f}')
    return mse, r2

# Initialize models
models = [
    (LinearRegression(), "Linear Regression (Not Optimized)"),
    (Ridge(alpha=1.0), "Ridge Regression (Not Optimized)"),
    (RandomForestRegressor(n_estimators=100, random_state=42), "Random Forest Regressor (Not Optimized)"),
    (HistGradientBoostingRegressor(random_state=42), "HistGradient Boosting Regressor (Not Optimized)"),
    (XGBRegressor(random_state=42), "XGBoost Regressor (Not Optimized)")
]

# Train and evaluate each model
results = []
for model, name in models:
    mse, r2 = train_and_evaluate(model, name, X_train, X_test, y_train, y_test)
    results.append((name, mse, r2))

# Find the best model based on R^2 score
best_model_info = max(results, key=lambda x: x[2])  # x[2] is the R^2 score

# Optimize the best model
optimized_model = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=3, min_samples_leaf=2, random_state=42)
optimized_name = f"{best_model_info[0]} (Optimized)"
mse_optimized, r2_optimized = train_and_evaluate(optimized_model, optimized_name, X_train, X_test, y_train, y_test)
results.append((optimized_name, mse_optimized, r2_optimized))

# 8/ Write metrics to a file
os.makedirs('out', exist_ok=True)
with open('out/score.txt', 'w') as f:
    for name, mse, r2 in results:
        f.write(f'{name} - Mean Squared Error: {mse:.4f}\n')
        f.write(f'{name} - R² Score: {r2:.4f}\n')
    f.write(f'\nBest Model: {best_model_info[0]} with MSE: {best_model_info[1]:.4f} and R² Score: {best_model_info[2]:.4f}\n')
    f.write(f'Optimized Model: {optimized_name} with MSE: {mse_optimized:.4f} and R² Score: {r2_optimized:.4f}\n')

print("Metrics for all models have been written to out/score.txt.")
