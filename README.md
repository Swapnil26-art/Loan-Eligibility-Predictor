# Loan Eligibility Predictor 🚀

This project predicts whether a loan application should be approved based on applicant details like age, income, education, and credit score. It uses machine learning models — Logistic Regression and Random Forest — and evaluates performance using ROC curves and confusion matrices.

## 📌 Problem Statement

Banks want to determine whether an applicant qualifies for a loan. The objective is to build a classification model that can make this decision based on historical applicant data.

---

## 🎯 Objective

To develop a machine learning model that:
- Takes applicant features (age, income, education, credit score)
- Predicts if a loan should be **approved** or **rejected**
- Can be used in fintech apps or loan assessment systems

---

## 🗃️ Dataset Features

| Feature         | Description                            |
|----------------|----------------------------------------|
| `age`           | Applicant's age                        |
| `income`        | Applicant's income                     |
| `education`     | Graduate / Not Graduate                |
| `credit_score`  | Numeric credit score                   |
| `loan_approved` | Target variable (Yes / No)             |

> You can use `loan_data.csv` with dummy data or your own dataset.

---

## ⚙️ Tech Stack & Libraries

- Python 🐍
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## ✅ Project Workflow

1. **Data Preprocessing**  
   - Handle missing data
   - Encode categorical features (education, loan_approved)

2. **Model Training**
   - Logistic Regression
   - Random Forest Classifier

3. **Model Evaluation**
   - Confusion Matrix
   - Classification Report
   - ROC Curve and AUC Score

4. **Output**
   - Printed evaluation results
   - ROC curve plot for model comparison

---

## 📊 Sample Output

```bash
Logistic Regression:
              precision    recall  f1-score   support

           0       0.50      1.00      0.67         1
           1       1.00      0.33      0.50         3

    accuracy                           0.60         4

Confusion Matrix:
[[1 0]
 [2 1]]
