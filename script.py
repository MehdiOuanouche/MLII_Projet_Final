# script.py
import pandas as pd
import numpy as np
import os
import optuna
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Ensure output directory exists
output_dir = "out"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Preprocess the data
train = train.drop(columns=["loan_status"], errors='ignore')
combined = pd.concat([test, train], axis=0)

# Data cleaning
combined = combined[combined['person_age'] <= 65]
combined = combined.fillna(combined.mean(numeric_only=True))
combined = combined.drop_duplicates()

# Sample data for faster testing
combined_sample = combined.sample(frac=0.1, random_state=42)
target_variable = 'loan_amnt'
X = combined_sample.drop(columns=[target_variable])
y = combined_sample[target_variable]

# Define numerical and categorical features
numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Function for each model's Optuna optimization
def optimize_model(model_name, objective_func):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_func, n_trials=5)  # Adjust trials as needed
    print(f"{model_name} optimized parameters: {study.best_params}")
    return study.best_params

# Define optimization functions
def objective_random_forest(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 5)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='r2')
    return np.mean(cv_scores)

def objective_hist_gradient_boosting(trial):
    max_iter = trial.suggest_int('max_iter', 50, 200)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 30)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        random_state=42
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='r2')
    return np.mean(cv_scores)

def objective_xgboost(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 150)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='r2')
    return np.mean(cv_scores)

# Perform optimization and get best parameters
best_params_rf = optimize_model("Random Forest", objective_random_forest)
best_params_hist_gb = optimize_model("HistGradientBoosting", objective_hist_gradient_boosting)
best_params_xgb = optimize_model("XGBoost", objective_xgboost)

# Define base and optimized models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Bayesian Ridge Regression": BayesianRidge(),
    "Random Forest Regressor (Base)": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
    "HistGradient Boosting Regressor (Base)": HistGradientBoostingRegressor(max_iter=50, random_state=42),
    "XGBoost Regressor (Base)": XGBRegressor(n_estimators=50, max_depth=3, random_state=42),
    "Random Forest Regressor (Optimized)": RandomForestRegressor(**best_params_rf, random_state=42),
    "HistGradient Boosting Regressor (Optimized)": HistGradientBoostingRegressor(**best_params_hist_gb, random_state=42),
    "XGBoost Regressor (Optimized)": XGBRegressor(**best_params_xgb, random_state=42)
}

# Evaluate all models and track the best one
best_model_name = None
best_r2_score = -np.inf

with open(os.path.join(output_dir, "score.txt"), "w") as f:
    for model_name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        
        # Train and evaluate the model
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Write metrics to file
        f.write(f"{model_name}:\n")
        f.write(f"  Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"  R² Score: {r2:.4f}\n")
        f.write("-" * 40 + "\n")
        
        # Track the best model
        if r2 > best_r2_score:
            best_r2_score = r2
            best_model_name = model_name

    # Write the best model's name and R² score at the end of the file
    f.write(f"\nBest Model: {best_model_name}\n")
    f.write(f"Best Model R² Score: {best_r2_score:.4f}\n")

print("Metrics for all models and the best model selection have been written to out/score.txt")
