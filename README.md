# Breast Cancer Classification — Machine Learning Assignment 2

## 1. Problem Statement

The objective is to implement multiple classification algorithms on the Breast Cancer Wisconsin (Diagnostic) dataset, compare their performance using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC), and deploy an interactive Streamlit application for evaluation on test data.

## 2. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository  
**UCI Dataset ID:** 17  
**Instances:** 569  
**Features:** 30 real-valued features  
**Task:** Binary classification

The 30 features describe characteristics of cell nuclei computed from digitized fine needle aspirate (FNA) images. The original diagnosis labels are malignant (M) and benign (B).

For this project, the target is encoded as:
- `1` = Malignant
- `0` = Benign

The assignment requires a minimum of 500 instances and 12 features, which this dataset satisfies.

## 3. GitHub Repository Link

**Add your GitHub repository URL here after creating the repository.**

## 4. Models Used

The assignment PDF says “6 ML models” but explicitly lists five required models. To avoid ambiguity, this implementation contains the five listed models plus **SVM as an additional sixth classifier**.

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)
6. Support Vector Machine (SVM)

### Evaluation Results

The following results were obtained using an 80:20 stratified train-test split with `random_state=42`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest | 0.9737 | 0.9944 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| SVM | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |


### Observations

- **Logistic Regression:** Provides a strong baseline and performs well after feature standardization. It is suitable when the class boundary is approximately linear.
- **Decision Tree:** Captures non-linear relationships and is easy to interpret, but a single tree can be more sensitive to the training data.
- **kNN:** Performs well after scaling because distance-based classification is sensitive to feature magnitudes. Its prediction process is computationally heavier than the linear models.
- **Naive Bayes:** Provides a fast probabilistic baseline. Its performance is influenced by the independence assumption among features.
- **Random Forest:** Combines many decision trees and generally provides robust performance by reducing the variance associated with a single tree.
- **SVM:** Uses a non-linear RBF kernel and can model complex decision boundaries after feature scaling.

### Overall Winner

Based on the **highest F1/MCC and strong AUC and accuracy on the held-out test set**, the overall winner in this experiment is **SVM**, based on the highest mean across the six reported metrics. Do not claim a winner based only on accuracy.

## 5. Streamlit Application

The application provides:

- CSV test-data upload
- Model selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Prediction preview

## 6. Project Structure

```text
breast_cancer_ml_assignment/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_results.csv
├── feature_names.txt
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── svm.pkl
```

## 7. How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload `test_data.csv` and select a model.

## 8. Streamlit Community Cloud

After pushing this folder to GitHub:

1. Sign in to Streamlit Community Cloud using GitHub.
2. Create a new app.
3. Select this repository and the `main` branch.
4. Select `app.py`.
5. Deploy the application.
6. Copy the live application URL into the final assignment PDF.

## 9. Test Data

`test_data.csv` contains the held-out 20% test set used for the reported evaluation results.

## 10. Reproducibility

- Train-test split: 80:20
- Stratification: enabled
- Random state: 42
- kNN neighbors: 7
- Decision Tree maximum depth: 5
- Random Forest estimators: 300
- SVM: RBF kernel
