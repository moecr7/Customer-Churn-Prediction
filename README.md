# Churn Prediction Project

## Overview
This project predicts customer churn using an ensemble stacking model combining Random Forest, XGBoost, LightGBM, and Logistic Regression as a meta-classifier. The model is optimized with Optuna hyperparameter tuning and uses SMOTE for class balancing.

## Features
- Data preprocessing: missing value imputation, categorical encoding, scaling, PCA feature extraction
- Data balancing with SMOTE
- Stacking ensemble model with hyperparameter tuning (Optuna)
- Model evaluation with classification report, confusion matrix, and threshold optimization
- Saving/loading model and predictions for reproducibility

## Getting Started

### Prerequisites
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, lightgbm, optuna, matplotlib, seaborn, joblib

Install dependencies:
```bash
pip install -r requirements.txt
```
Project Structure
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
└── y_pred_final.csv
└── stacking_model.pkl

└── import_data_visualization.ipynb
└── preprocessing.ipynb
└── modelling.ipynb
└── evaluating.ipynb

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


Run these notebooks in this order:

import_data_visualization.ipynb — import data and vizualization

preprocessing.ipynb — data cleaning, encoding, PCA

modelling.ipynb — train stacking model with Optuna tuning

evaluating.ipynb — ate model, tune threshold, save predictions and model

🧠 Model Summary
Final model: StackingClassifier combining RandomForest, XGBoost, LightGBM

Meta-learner: Logistic Regression

Imbalanced data handled with SMOTE

Hyperparameters optimized with Optuna

📦 Outputs
Model saved at model/stacking_model.pkl

Data splits and predictions saved in data/

Evaluation metrics and plots generated in notebooks



Results
Achieved Recall score: 0.93

Confusion matrix and classification report available in evaluation notebook

Future Work
Add more feature engineering and data visualization

Experiment with other ensemble methods or neural networks

Deploy model using Flask/Django API

Contact
For questions or collaboration: [moeinbox55@gmail.com]

This project is part of my machine learning portfolio, showcasing data preprocessing, model tuning, and ensemble learning.
