"""
Breast Cancer Classification - Model Training Script

This script:
1. Loads the Breast Cancer Wisconsin Diagnostic dataset.
2. Creates a stratified 80:20 train-test split.
3. Trains six classification models.
4. Evaluates each model using Accuracy, AUC, Precision, Recall, F1 and MCC.
5. Saves the trained models in the model/ directory.
6. Saves the held-out test data and model results for the Streamlit app.

Run:
    python train_models.py
"""

import os
import pickle
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)

# sklearn's breast cancer dataset uses:
# 0 = malignant, 1 = benign
# Assignment requirement:
# 1 = malignant, 0 = benign
y = pd.Series((data.target == 0).astype(int), name="target")


# ---------------------------------------------------------
# Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# ---------------------------------------------------------
# Models
#
# Pipelines are used for models that require scaling.
# This keeps preprocessing together with the model and
# avoids having to separately save a scaler.
# ---------------------------------------------------------
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),

    "decision_tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),

    "naive_bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}


# ---------------------------------------------------------
# Train, evaluate and save models
# ---------------------------------------------------------
results = []

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    result = {
        "ML Model Name": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    results.append(result)

    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print(f"Saved model: {model_path}")


# ---------------------------------------------------------
# Save model evaluation results
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(BASE_DIR, "model_results.csv"),
    index=False
)


# ---------------------------------------------------------
# Save held-out test data
# ---------------------------------------------------------
test_data = X_test.copy()
test_data["target"] = y_test.values

test_data.to_csv(
    os.path.join(BASE_DIR, "test_data.csv"),
    index=False
)


# ---------------------------------------------------------
# Save feature names
# ---------------------------------------------------------
with open(os.path.join(BASE_DIR, "feature_names.txt"), "w") as file:
    for feature in data.feature_names:
        file.write(feature + "\n")


# ---------------------------------------------------------
# Print final results
# ---------------------------------------------------------
print("\nModel Evaluation Results")
print("=" * 80)
print(results_df.to_string(index=False))

print("\nTraining completed successfully.")
print(f"Models saved in: {MODEL_DIR}")
print(f"Results saved in: {os.path.join(BASE_DIR, 'model_results.csv')}")
print(f"Test data saved in: {os.path.join(BASE_DIR, 'test_data.csv')}")
