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

# Project Summary

This project focused on building and evaluating machine learning models to predict the likelihood of heart disease.

## Key Findings

* Several machine learning models were successfully implemented and evaluated.
* All models demonstrated reasonable predictive performance on the dataset.
* The **Random Forest** model consistently achieved the best performance metrics across our evaluations.
* Feature importance analysis using the Random Forest model highlighted the following features as particularly influential in the prediction:
    * **ca**: Number of major vessels (0-3) colored by fluoroscopy
    * **thal**: Thallium stress test result (3 = normal; 6 = fixed defect; 7 = reversible defect)
    * **oldpeak**: ST depression induced by exercise relative to rest
    * **thalach**: Maximum heart rate achieved
* **SHAP (SHapley Additive exPlanations) analysis** was employed to provide further insights into how individual features contribute to the model's predictions for each instance. This allows for a more granular understanding of the model's decision-making process.

## Future Improvements

To further enhance the project and its potential clinical utility, the following improvements are planned:

* **Hyperparameter Tuning:** Implementing techniques like **GridSearchCV** to systematically search for the optimal hyperparameters for each model. This can lead to significant improvements in model performance.
* **Feature Engineering:** Exploring and creating new features from the existing data that might capture more complex relationships and improve the predictive power of the models.
* **Additional Explainability Tools:** Investigating and integrating other explainability techniques that are particularly relevant and interpretable for clinical practitioners. This will aim to increase trust and understanding in the model's predictions within a medical context.

---

## 🔧 Installation

To run this project locally or in a notebook environment, install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn shap joblib

