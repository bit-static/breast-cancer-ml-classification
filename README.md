# Breast Cancer Classification — Machine Learning Assignment 2

## 1. Problem Statement

The main objective of this assignment is to implement and compare different machine learning classification algorithms using the Breast Cancer Wisconsin (Diagnostic) dataset.

Six classification models have been implemented and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier
6. Support Vector Machine (SVM)

The models are compared using the following evaluation metrics:

* Accuracy
* AUC
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

Along with the model implementation, a Streamlit application has also been developed. The application allows the user to upload test data, select a machine learning model and view the prediction results and evaluation metrics.

---

## 2. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)
**Source:** UCI Machine Learning Repository
**UCI Dataset ID:** 17

The dataset contains **569 instances and 30 numerical features**. These features represent different characteristics of cell nuclei obtained from digitized Fine Needle Aspirate (FNA) images.

The problem is a binary classification problem where the objective is to classify a breast tumor as either malignant or benign.

For this project, the target variable has been encoded as:

* `1` → Malignant
* `0` → Benign

The dataset satisfies the minimum requirements given in the assignment, as it contains more than 500 instances and more than 12 features.

---

## 3. GitHub Repository

The complete project code, trained models, test data and Streamlit application are available in the following GitHub repository:

**GitHub Repository:**
https://github.com/bit-static/breast-cancer-ml-classification

---

## 4. Models Used

The assignment specifies six machine learning models, with five models explicitly mentioned in the given requirements. I have implemented the five specified models and included **SVM as the sixth model**.

### 4.1 Logistic Regression

Logistic Regression is used as a baseline classification model. Since the features have different ranges, feature standardization is applied before training the model.

It performed well on this dataset and provides a good baseline for comparison with the other classifiers.

### 4.2 Decision Tree Classifier

Decision Tree is a non-linear model which splits the data based on feature values. It is easy to understand and interpret.

In this implementation, the maximum depth of the tree is restricted to avoid unnecessary overfitting.

### 4.3 K-Nearest Neighbors (kNN)

kNN classifies a new sample based on the classes of its nearest training samples.

Since kNN is a distance-based algorithm, the features are standardized before applying the model. A value of **7** has been used for the number of neighbors.

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm. It assumes that the features are conditionally independent given the class.

It is computationally fast and has been included as a simple probabilistic model for comparison.

### 4.5 Random Forest

Random Forest is an ensemble learning algorithm which combines multiple decision trees.

It generally performs better than a single decision tree because the predictions from multiple trees help reduce variance and improve generalization.

In this project, **300 trees** are used in the Random Forest model.

### 4.6 Support Vector Machine (SVM)

SVM is used with an **RBF (Radial Basis Function) kernel**. The RBF kernel allows the model to learn non-linear decision boundaries.

Feature scaling is performed before training the SVM because SVM is sensitive to the scale of input features.

---

## 5. Model Evaluation Results

The dataset was divided into training and testing data using an **80:20 stratified train-test split** with `random_state=42`.

The obtained results are given below.

| ML Model Name       | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| ------------------- | -------: | -----: | --------: | -----: | -------: | -----: |
| Logistic Regression |   0.9649 | 0.9960 |    0.9750 | 0.9286 |   0.9512 | 0.9245 |
| Decision Tree       |   0.9211 | 0.9448 |    0.9459 | 0.8333 |   0.8861 | 0.8299 |
| kNN                 |   0.9561 | 0.9825 |    0.9744 | 0.9048 |   0.9383 | 0.9058 |
| Naive Bayes         |   0.9211 | 0.9891 |    0.9231 | 0.8571 |   0.8889 | 0.8292 |
| Random Forest       |   0.9737 | 0.9944 |    1.0000 | 0.9286 |   0.9630 | 0.9442 |
| SVM                 |   0.9737 | 0.9947 |    1.0000 | 0.9286 |   0.9630 | 0.9442 |

---

## 6. Observations

### Logistic Regression

Logistic Regression gives good performance on this dataset, with an accuracy of **96.49%** and an AUC of **0.9960**. It is a good baseline model and performs particularly well after standardizing the features.

### Decision Tree

The Decision Tree gives an accuracy of **92.11%**. Although it can capture non-linear relationships between the features, its performance is lower compared to Random Forest and SVM in this experiment.

### kNN

kNN achieves an accuracy of **95.61%**. The model performs well after feature scaling. Since the algorithm calculates distances between samples, scaling the features is important for getting meaningful results.

### Naive Bayes

Naive Bayes achieves an accuracy of **92.11%**. It has a relatively high AUC of **0.9891**, although its overall classification performance is lower than Logistic Regression, Random Forest and SVM.

### Random Forest

Random Forest gives an accuracy of **97.37%**, with a precision of **1.0000** and an F1 Score of **0.9630**. The use of multiple decision trees helps the model perform better than a single Decision Tree.

### SVM

SVM also gives an accuracy of **97.37%** and an F1 Score of **0.9630**. It has an AUC of **0.9947** and MCC of **0.9442**, making it one of the best performing models in this experiment.

---

## 7. Overall Model Comparison

From the results, **Random Forest and SVM have the highest accuracy of 97.37%**.

However, accuracy alone is not used to decide the best model. Other metrics such as AUC, Precision, Recall, F1 Score and MCC are also considered.

SVM has a slightly higher AUC than Random Forest, while both models have the same Precision, Recall, F1 Score and MCC in the obtained results.

Considering the overall performance across the reported metrics, **SVM is selected as the overall best model for this experiment**.

This conclusion is based on the combined performance of the evaluation metrics and not only on accuracy.

---

## 8. Streamlit Application

A Streamlit-based web application has been developed for testing the trained models.

The application provides the following features:

* Upload test data in CSV format
* Select a machine learning model
* Generate predictions on the uploaded data
* Display Accuracy
* Display AUC
* Display Precision
* Display Recall
* Display F1 Score
* Display MCC
* Display confusion matrix
* Display classification report
* Show a preview of the predictions

The trained models are stored separately as `.pkl` files and are loaded by the Streamlit application when required.

---

## 9. Project Structure

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

---

## 10. Running the Project Locally

First, install the required Python libraries using:

```bash
pip install -r requirements.txt
```

After installing the dependencies, run the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in the browser. The test CSV file can then be uploaded and a model can be selected from the dropdown.

---

## 11. Deployment on Streamlit Community Cloud

The project has been maintained in a GitHub repository so that it can be deployed using Streamlit Community Cloud.

Repository:

https://github.com/bit-static/breast-cancer-ml-classification

The basic deployment process is:

1. Open Streamlit Community Cloud and sign in using GitHub.
2. Create a new application.
3. Select the GitHub repository.
4. Select the `main` branch.
5. Select `app.py` as the main application file.
6. Deploy the application.
7. After successful deployment, the generated application URL can be added to the final assignment submission.

---

## 12. Test Data

The `test_data.csv` file contains the held-out test samples from the 80:20 train-test split.

The same test set is used for evaluating the different models so that their performance can be compared under the same conditions.

---

## 13. Reproducibility

The following settings were used during the experiment:

* Train-test split: **80:20**
* Stratified split: **Yes**
* Random state: **42**
* Number of kNN neighbors: **7**
* Decision Tree maximum depth: **5**
* Random Forest estimators: **300**
* SVM kernel: **RBF**

Using the same dataset and these settings should produce the same or very similar results.

---

## 14. Conclusion

In this assignment, six different classification algorithms were implemented on the Breast Cancer Wisconsin (Diagnostic) dataset.

The models showed good classification performance overall. Decision Tree and Naive Bayes had comparatively lower performance, while Logistic Regression, kNN, Random Forest and SVM performed better.

Random Forest and SVM achieved the highest accuracy of **97.37%**. Considering the complete set of evaluation metrics, SVM was selected as the overall best performing model in this experiment.

The trained models were also integrated into a Streamlit application, which provides an interactive way to upload test data, select a model and view its performance and predictions.

**GitHub Repository:**
https://github.com/bit-static/breast-cancer-ml-classification
