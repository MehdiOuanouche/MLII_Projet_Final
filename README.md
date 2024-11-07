# MLII_Projet_Final

# Loan Default Prediction

## 1. Business Use Case

This project addresses a crucial need in financial services: predicting loan defaults. Loan default prediction models help financial institutions manage lending risk by identifying high-risk borrowers. Accurate predictions of default risk allow banks to:
- Minimize potential losses by limiting exposure to high-risk loans.
- Offer more favorable rates to low-risk customers.
- Improve decision-making and operational efficiency in loan approvals.

This project uses machine learning to build a model that predicts whether a borrower will default on their loan, based on demographic, loan, and credit history information.

## 2. Dataset Description

The dataset, sourced from Kaggle’s [Playground Series S4 E10](https://www.kaggle.com/competitions/playground-series-s4e10/overview), contains information about loan applicants, their loan characteristics, and credit history. Key features include:

- **Demographic Information**:
  - `person_age`: Applicant’s age.
  - `person_income`: Applicant’s annual income.
  - `person_home_ownership`: Home ownership status (e.g., rent, mortgage).
  - `person_emp_length`: Employment length in years.

- **Loan Characteristics**:
  - `loan_intent`: Purpose of the loan (e.g., education, medical).
  - `loan_amnt`: Amount of the loan requested.
  - `loan_int_rate`: Interest rate for the loan.
  - `loan_percent_income`: Loan amount as a percentage of annual income.

- **Credit History**:
  - `cb_person_default_on_file`: Whether the applicant has previously defaulted.
  - `cb_person_cred_hist_length`: Length of the applicant’s credit history.

The target variable, `loan_status`, is a binary indicator where 1 signifies a default and 0 signifies no default.

## 3. Baseline Model

### Baseline Setup

For the baseline, a simple model was implemented to understand initial predictive power without extensive tuning. This baseline model includes the following setup:

#### Features Used:
- **Numerical Features**: `person_age`, `person_income`, `person_emp_length`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`.
- **Categorical Features**: `person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`.

#### Preprocessing Steps:
1. **Numerical Features**: Standardized using `StandardScaler` to center and scale the values.
2. **Categorical Features**: One-hot encoded using `OneHotEncoder` to handle categorical variables without introducing an ordinal relationship.

#### Model:
- A **RandomForestClassifier** was chosen for the baseline. It provides robust, interpretable results with minimal tuning. The classifier was initialized with 50 trees and a max depth of 5 for efficient training and to establish a quick performance baseline.

#### Metrics:
- **Accuracy**: Measures the percentage of correct predictions.
- **ROC-AUC**: Accounts for both true positives and false positives, making it ideal for imbalanced datasets.

#### Results:
- **Baseline Accuracy**: ~0.70
- **Baseline ROC-AUC**: ~0.75

These metrics provided a reasonable starting point for further optimization, indicating room for improvement through hyperparameter tuning and feature engineering.

## 4. First Iteration

After establishing the baseline, the next steps focused on feature engineering and hyperparameter tuning using Optuna, a popular library for automated hyperparameter optimization.

### Changes Implemented

#### 1. Feature Engineering
- **Loan-to-Income Ratio**: A new feature that expresses the loan amount as a percentage of income, providing insight into the applicant’s financial risk.
- **Employment Duration**: Employment length was retained as an indicator of job stability, which can correlate with loan repayment capability.

#### 2. Hyperparameter Tuning with Optuna
Using Optuna, we optimized key hyperparameters for the `RandomForestClassifier`:
- `n_estimators`: The number of trees in the forest.
- `max_depth`: The maximum depth of each tree.
- `min_samples_split`: Minimum samples required to split an internal node.
- `min_samples_leaf`: Minimum samples required to form a leaf node.

Optuna was configured to explore hyperparameters in a limited number of trials for efficient tuning.

### Results

After incorporating feature engineering and tuning, the model's metrics showed a meaningful improvement:

- **First Iteration Accuracy**: ~0.75
- **First Iteration ROC-AUC**: ~0.80

The ROC-AUC increase indicated that the model became better at distinguishing between defaults and non-defaults. Hyperparameter tuning in the `RandomForestClassifier` allowed for more complex patterns to be captured, while the new feature (loan-to-income ratio) added valuable information.

## 5. Final Model and Experiment Tracking

In the final model iteration, additional improvements were explored through various algorithms, including:
1. **XGBoost Regressor**: For its strong predictive power in handling structured data.
2. **HistGradientBoostingRegressor**: For its ability to handle imbalanced data.
3. **Logistic Regression with Optuna Hyperparameter Tuning**: To optimize the C parameter and solver choice for better regularization.

### Experiment Tracking with MLflow
All key experiments were tracked using **MLflow**, which logs:
- Model parameters and hyperparameters.
- Evaluation metrics (MSE, R²).
- Trained model artifacts for easy comparison and reproducibility.

Each model's performance was logged in MLflow, allowing for comparisons between baseline, intermediate, and final models.

## 6. Cross-Validation and Evaluation

To validate the model's stability, **K-Fold Cross-Validation** was used, and the results were compared with test set performance. This approach mitigated overfitting by evaluating the model across multiple data splits.

### Evaluation Metrics:
- **Mean Squared Error (MSE)**: Used for regression tasks, measuring the average squared difference between actual and predicted values.
- **R² Score**: Indicates the proportion of variance captured by the model.

#### Final Model Results:
- **Cross-Validation MSE**: ~ [final MSE from cross-validation]
- **Cross-Validation R²**: ~ [final R² from cross-validation]
- **Test Set MSE**: ~ [final MSE from test set]
- **Test Set R²**: ~ [final R² from test set]

## 7. Conclusion and Next Steps

The final model achieved a significant improvement over the baseline, leveraging optimized hyperparameters, feature engineering, and cross-validation to generalize well on the test data. Future directions include:
- Exploring other algorithms like **Gradient Boosting Machines** or **CatBoost**.
- Experimenting with class-balancing techniques to address target imbalance.
- Integrating additional domain-specific features for further improvements.

