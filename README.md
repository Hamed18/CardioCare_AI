# CardioCare AI 🫀

**CardioCare AI** is a machine learning project designed to predict the likelihood of heart disease based on patient health data. This model leverages advanced data preprocessing, visualization, classification, and explainability tools to provide interpretable and reliable predictions for heart disease diagnosis.

---

## 📌 Features

- Binary classification: Detect presence of heart disease
- Exploratory Data Analysis (EDA) using Seaborn & Matplotlib
- Missing value handling and data cleaning
- Feature scaling with MinMaxScaler
- Class imbalance handled using SMOTE
- Machine Learning models:
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM)
- ROC Curve and accuracy comparison across models
- SHAP-based model explainability
- Model saving using `joblib`

---

## 📁 Dataset

The dataset includes 13 clinical features used to predict heart disease, such as:

- `age`
- `sex`
- `cp` (chest pain type)
- `trestbps` (resting blood pressure)
- `chol` (serum cholesterol)
- `fbs` (fasting blood sugar)
- `restecg` (resting ECG results)
- `thalach` (maximum heart rate achieved)
- `exang` (exercise-induced angina)
- `oldpeak`, `slope`, `ca`, `thal`

> Dataset file: `heart.csv` (should be present in the working directory)

---

## 📊 Models Used

The following models are trained, evaluated, and compared:

- **K-Nearest Neighbors**
- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

Model performance is evaluated using:
- Accuracy
- Confusion Matrix
- ROC-AUC Score
- Cross-Validation Score

---

## 📈 Visualizations & Explainability

- Correlation heatmap
- Distribution of features
- SHAP summary plots and dependence plots
- Feature importance (Random Forest)

---

## 🔧 Installation

To run this project locally or in a notebook environment, install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn shap joblib

