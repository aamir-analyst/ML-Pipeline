# Loan Delinquent Analysis – Conclusion

## 1. Project Objective

The objective of this project was to analyze loan delinquency data and develop a Machine Learning classification model that can help identify customers who are likely to become delinquent.

## 2. Dataset Overview

- Total Records: 11,548
- Original Features: 7 predictive attributes and 1 target column
- Target Variable: isDelinquent
- Missing Values: 0
- Duplicate Records: 0

## 3. Data Preprocessing

The following preprocessing steps were performed:

- Removed the customer ID column.
- Standardized purpose categories.
- Checked missing values and duplicate records.
- Applied One-Hot Encoding to categorical features.
- Prepared the dataset for Machine Learning.

## 4. Exploratory Data Analysis

EDA revealed that:

- 66.86% of customers were delinquent.
- 33.14% of customers were non-delinquent.
- FICO category showed a notable association with delinquency.
- Customers in the 300-500 FICO category had a higher delinquency rate than customers in the above-500 category.

These findings represent associations in the dataset and do not establish causation.

## 5. Machine Learning Models

The following experiments were performed:

1. Baseline Logistic Regression
2. Balanced Logistic Regression
3. Logistic Regression with Strong Regularization
4. Random Oversampling
5. Random Undersampling

## 6. Cross Validation Results

Five-Fold Stratified Cross Validation produced the following results:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline Logistic Regression | 79.25% | 82.36% | 87.78% | 84.98% | 80.20% |
| Balanced Logistic Regression | 78.33% | 89.32% | 76.77% | 82.55% | 83.52% |
| Oversampling Logistic Regression | 77.95% | 89.41% | 76.02% | 82.14% | 83.50% |

## 7. Final Model Selection

Baseline Logistic Regression was selected as the preferred candidate for the project's primary objective of identifying delinquent customers.

It achieved:

- Accuracy: 79.25%
- Precision: 82.36%
- Recall: 87.78%
- F1-Score: 84.98%
- ROC-AUC: 80.20%

The model achieved the highest delinquent-class recall among the three models evaluated using cross-validation.

Balanced Logistic Regression achieved a higher ROC-AUC and precision, making it another useful candidate depending on the bank's business priorities.

## 8. Conclusion

This project demonstrated the application of Exploratory Data Analysis, data preprocessing, Logistic Regression, regularization, class balancing techniques, and Stratified K-Fold Cross Validation.

The baseline Logistic Regression model showed promising performance in identifying delinquent customers.

The results can help demonstrate how Machine Learning can support loan risk analysis. However, further validation on independent data, business cost analysis, fairness assessment, and threshold optimization would be required before real-world banking deployment.