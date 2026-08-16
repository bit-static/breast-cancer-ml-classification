import os
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "SVM": "svm.pkl",
}

FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

st.title("🩺 Breast Cancer Classification")
st.write(
    "Interactive evaluation of classification models on the "
    "Breast Cancer Wisconsin (Diagnostic) test data."
)

with st.sidebar:
    st.header("Model Selection")
    selected_model = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])
    st.caption("The uploaded CSV should contain the 30 feature columns and a Diagnosis column.")
    st.caption("Diagnosis: 1 = Malignant, 0 = Benign")

if uploaded_file is None:
    st.info("Upload the supplied test_data.csv file to evaluate the selected model.")
    st.stop()

df = pd.read_csv(uploaded_file)

missing_features = [c for c in FEATURE_NAMES if c not in df.columns]
if missing_features:
    st.error(f"Missing feature columns: {missing_features}")
    st.stop()

if "Diagnosis" not in df.columns:
    st.error("The CSV must contain a 'Diagnosis' column to calculate evaluation metrics.")
    st.stop()

X = df[FEATURE_NAMES]
y = df["Diagnosis"].astype(int)

model_path = os.path.join(MODEL_DIR, MODEL_FILES[selected_model])
with open(model_path, "rb") as f:
    model = pickle.load(f)

pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y, pred),
    "AUC": roc_auc_score(y, prob),
    "Precision": precision_score(y, pred, zero_division=0),
    "Recall": recall_score(y, pred, zero_division=0),
    "F1 Score": f1_score(y, pred, zero_division=0),
    "MCC": matthews_corrcoef(y, pred),
}

st.subheader(f"Evaluation Metrics — {selected_model}")
cols = st.columns(6)
for col, (metric, value) in zip(cols, metrics.items()):
    col.metric(metric, f"{value:.4f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1], ["Benign", "Malignant"])
    ax.set_yticks([0, 1], ["Benign", "Malignant"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y, pred, target_names=["Benign", "Malignant"], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4))

st.subheader("Prediction Preview")
preview = X.copy()
preview["Actual Diagnosis"] = y.values
preview["Predicted Diagnosis"] = pred
preview["Prediction"] = preview["Predicted Diagnosis"].map({0: "Benign", 1: "Malignant"})
st.dataframe(preview.head(20))
