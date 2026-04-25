# Customer Churn Prediction

Predicting customer churn for a telecom company using classic ML classification models.

## Problem Statement
A telecom company wants to identify customers likely to cancel their subscription
before it happens — enabling proactive retention offers. This project builds and 
compares three classification models to predict churn from customer profile data.

## Dataset
- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,032 customers × 21 features
- **Target:** `Churn` (Yes / No) — 26.5% positive class

## Project Structure
customer-churn-prediction/
│
├── data/
│   └── telco_churn.csv
├── notebook/
│   └── churn_prediction.ipynb
├── requirements.txt
└── README.md

## Workflow
1. **EDA** — Distribution analysis, churn rate by contract type, correlation heatmap
2. **Preprocessing** — Label encoding, one-hot encoding, StandardScaler, null handling
3. **Modelling** — Logistic Regression, KNN (tuned K), Naive Bayes
4. **Evaluation** — StratifiedKFold CV, F1, ROC-AUC, Confusion Matrix
5. **Regularization** — L1/L2 tuning on Logistic Regression

## Results

| Model               | Precision | Recall | F1    | ROC-AUC |
|---------------------|-----------|--------|-------|---------|
| Logistic Regression | 0.65      | 0.56   | 0.603 | 0.843   |
| KNN (k=?)           | 0.58      | 0.51   | 0.543 | 0.771   |
| Naive Bayes         | 0.42      | 0.76   | 0.541 | 0.832   |

> Logistic Regression with L2 regularization performed best overall.
> Naive Bayes showed higher recall — useful when minimising missed churners matters most.

## Key Insights
- Month-to-month contract customers churn at ~42% vs ~3% for two-year contracts
- New customers (low tenure) are significantly more likely to churn
- TotalCharges and tenure are highly correlated (0.83)

## Tech Stack
- Python, pandas, matplotlib, seaborn, scikit-learn, Jupyter

## Setup
pip install -r requirements.txt
jupyter notebook notebook/churn_prediction.ipynb

## Skills Demonstrated
Encoding · Normalization · Cross-Validation · Regularization · 
Model Comparison · Imbalanced Classification
