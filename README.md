# Churn Prediction Project

## Overview
This project predicts customer churn using an ensemble stacking model combining Random Forest, XGBoost, LightGBM, and Logistic Regression as a meta-classifier. The model is optimized with Optuna hyperparameter tuning and uses SMOTE for class balancing.

## Features
- Data preprocessing: missing value imputation, categorical encoding, scaling, PCA feature extraction
- Data balancing with SMOTE
- Stacking ensemble model with hyperparameter tuning (Optuna)
- Model evaluation with classification report, confusion matrix, and threshold optimization
- Scaling data with RobustScaler

## Getting Started

### Prerequisites
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, lightgbm, optuna, matplotlib, seaborn, joblib

Install dependencies:
```bash
pip install -r requirements.txt
```
Project Structure
├── costumer_churn.ipynb

├── README.md

├── requirements.txt

└── .gitignore

---

## 🚀 How to Run the Project

1. **Clone the repo:**

```bash
git clone https://github.com/moecr7/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```
2.Install dependencies:

```bash
pip install -r requirements.txt
```


Run costumer_churn notebooks in this order

🧠 Model Summary
Final model: StackingClassifier combining RandomForest, XGBoost, LightGBM

Meta-learner: Logistic Regression

Imbalanced data handled with SMOTE

Hyperparameters optimized with Optuna


Results
Achieved Recall score: 0.94

Confusion matrix and classification report available in evaluation notebook

Future Work
Add more feature engineering and data visualization

Experiment with other ensemble methods or neural networks

Deploy model using Flask/Django API

Contact
For questions or collaboration: [moeinbox55@gmail.com]

This project is part of my machine learning portfolio, showcasing data preprocessing, model tuning, and ensemble learning.
