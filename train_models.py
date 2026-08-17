# train_models.py

import os
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load Breast Cancer Wisconsin Dataset
# ---------------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

# sklearn dataset:
# 0 = malignant
# 1 = benign
#
# Assignment requirement:
# 1 = malignant
# 0 = benign
y = 1 - y


# ---------------------------------------------------------
# Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# Define sklearn Models
# ---------------------------------------------------------

models = {

    "logistic_regression": LogisticRegression(
        max_iter=5000,
        random_state=42
    ),

    "decision_tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "knn": KNeighborsClassifier(
        n_neighbors=7
    ),

    "naive_bayes": GaussianNB(),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "svm": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )
}


# ---------------------------------------------------------
# Train Models and Save as Pickle Files
# ---------------------------------------------------------

for model_name, model in models.items():

    print(f"Training {model_name}...")

    # Models that require feature scaling
    if model_name in [
        "logistic_regression",
        "knn",
        "naive_bayes",
        "svm"
    ]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)

    # Save trained sklearn model
    model_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.pkl"
    )

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print(f"Saved -> {model_path}")


# ---------------------------------------------------------
# Save Test Data
# ---------------------------------------------------------

# Save original, unscaled test features.
# app.py can apply scaling when required.
test_data_path = os.path.join(
    BASE_DIR,
    "test_data.csv"
)

import pandas as pd

test_df = pd.DataFrame(
    X_test,
    columns=data.feature_names
)

test_df["target"] = y_test

test_df.to_csv(
    test_data_path,
    index=False
)


print("\n-----------------------------------------")
print("Training completed successfully.")
print("-----------------------------------------")
print(f"Models saved in: {MODEL_DIR}")
print(f"Test data saved at: {test_data_path}")
