# Loan Delinquent Analysis

A Machine Learning project to analyze loan delinquency patterns and predict whether a borrower is likely to become delinquent based on loan and customer attributes.

## Project Overview

Loan delinquency is a major challenge for financial institutions because delayed repayments can increase credit risk and financial losses.

This project focuses on analyzing historical loan data and building machine learning models to identify borrowers who are more likely to become delinquent.

The project uses Python, Pandas, Scikit-learn, and Imbalanced-learn to perform data analysis, preprocessing, model training, evaluation, and prediction.

## Business Problem

DRS Bank wants to reduce loan delinquency risk by identifying customers who may fail to repay their loans on time.

The objective is to:

- Analyze historical loan repayment patterns.
- Identify factors associated with loan delinquency.
- Handle imbalanced target data.
- Train and compare multiple machine learning models.
- Evaluate model performance using classification metrics.
- Predict the delinquency risk of new loan applicants.

## Dataset Information

- **Total Records:** 11,548
- **Total Columns:** 8
- **Target Variable:** `isDelinquent`
- **Missing Values:** 0
- **Duplicate Records:** 0

### Dataset Features

| Feature | Description |
|---|---|
| ID | Unique loan/customer identifier |
| isDelinquent | Target variable (1 = Delinquent, 0 = Non-Delinquent) |
| term | Loan repayment term |
| gender | Customer gender |
| purpose | Purpose of the loan |
| home_ownership | Customer home ownership status |
| age | Customer age group |
| FICO | FICO credit score category |

### Target Distribution

| Class | Records | Percentage |
|---|---:|---:|
| Delinquent (1) | 7,721 | 66.86% |
| Non-Delinquent (0) | 3,827 | 33.14% |

The dataset has an imbalanced target distribution, so class imbalance was considered during model development.

## Exploratory Data Analysis (EDA)

The following analysis was performed:

- Target variable distribution.
- Age group analysis.
- FICO score analysis.
- Loan term analysis.
- Gender analysis.
- Loan purpose analysis.
- Home ownership analysis.
- Correlation analysis.

### Key Observations

- Approximately 66.86% of records belong to the delinquent class.
- The 36-month loan category contains most of the records.
- The 300–500 FICO category has a higher observed delinquency rate than the above-500 category.
- Customer and loan characteristics show different delinquency patterns.

These observations describe associations in the dataset and do not establish causation.

## Data Preprocessing

The following preprocessing steps were performed:

1. Checked missing values.
2. Checked duplicate records.
3. Removed the unnecessary `ID` column.
4. Standardized loan purpose categories.
5. Applied One-Hot Encoding to categorical features.
6. Separated independent features and target variable.
7. Split data into training and testing sets.

### Train-Test Split

- Training Data: 80%
- Testing Data: 20%
- Random State: 42
- Stratification: Enabled

## Machine Learning Models

Five model configurations were trained and compared:

1. Baseline Logistic Regression
2. Balanced Logistic Regression
3. Strong Regularization Logistic Regression
4. Random Oversampling + Logistic Regression
5. Random Undersampling + Logistic Regression

### Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Since identifying delinquent borrowers is an important business objective, special attention was given to Recall for class 1.

## Model Comparison Results

Results on the held-out test dataset:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline Logistic Regression | 0.7892 | 0.8313 | 0.8588 | 0.8449 | 0.7888 |
| Balanced Logistic Regression | 0.7784 | 0.8811 | 0.7727 | 0.8233 | 0.8280 |
| Strong Regularization | 0.7528 | 0.8852 | 0.7241 | 0.7966 | 0.8232 |
| Oversampling | 0.7779 | 0.8810 | 0.7720 | 0.8229 | 0.8271 |
| Undersampling | 0.7766 | 0.8819 | 0.7688 | 0.8215 | 0.8272 |

### Selected Model

**Baseline Logistic Regression**

The baseline model was selected because it achieved the highest recall for the delinquent class among the five tested configurations.

- Accuracy: 78.92%
- Precision: 83.13%
- Recall: 85.88%
- F1-Score: 84.49%
- ROC-AUC: 78.88%

The selected model prioritizes identifying more delinquent borrowers. However, this also results in some non-delinquent borrowers being incorrectly classified as delinquent.

## Cross-Validation

Stratified 5-Fold Cross-Validation was performed on the training dataset.

The following configurations were evaluated:

- Baseline Logistic Regression
- Balanced Logistic Regression
- Oversampling + Logistic Regression

Cross-validation was used to assess model performance across multiple training-validation splits.

## Confusion Matrix

The selected model produced the following confusion matrix:

```text
[[497, 269],
 [218, 1326]]
```

Interpretation:

- True Negative: 497
- False Positive: 269
- False Negative: 218
- True Positive: 1326

Here, class 1 represents delinquent borrowers.

## ROC Curve

The ROC curve was generated to evaluate the model's ability to distinguish between delinquent and non-delinquent borrowers.

The ROC-AUC score of the selected model is **0.7888**.

## Project Structure

```text
Loan_Delinquent_Analysis/
│
├── data/
│   ├── raw/
│   │   └── Loan_Delinquent_Dataset (1).csv
│   │
│   └── processed/
│       └── loan_delinquent_processed.csv
│
├── notebooks/
│   ├── 01_data_overview.ipynb
│   └── 02_eda.ipynb
│
├── src/
│   ├── data/
│   │   └── load_data.py
│   │
│   ├── eda/
│   │   └── eda.py
│   │
│   ├── preprocessing/
│   │   └── preprocess.py
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_model.py
│   │   └── final_evaluation.py
│   │
│   └── pipeline/
│       └── main_pipeline.py
│
├── models/
│   ├── loan_delinquency_model.pkl
│   └── model_metadata.json
│
├── reports/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   └── roc_curve.png
│   │
│   ├── conclusion.md
│   ├── model_comparison.csv
│   ├── cross_validation_results.csv
│   ├── final_evaluation.txt
│   ├── pipeline_model_comparison.csv
│   ├── pipeline_cross_validation.csv
│   └── pipeline_final_evaluation.txt
│
├── logs/
│   └── pipeline.log
│
├── README.md
└── requirements.txt
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib
- Jupyter Notebook
- VS Code
- Git & GitHub

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd Loan_Delinquent_Analysis
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

Run the complete machine learning pipeline:

```bash
python -m src.pipeline.main_pipeline
```

Run individual modules:

```bash
python -m src.data.load_data
```

```bash
python -m src.preprocessing.preprocess
```

```bash
python -m src.models.train_model
```

```bash
python -m src.evaluation.evaluate_model
```

## Make Predictions

To predict whether a new borrower is likely to become delinquent:

```bash
python -m src.models.predict
```

The program asks for borrower details and returns:

- Probability of being non-delinquent.
- Probability of being delinquent.
- Final predicted loan status.

## Future Improvements

- Experiment with Random Forest, XGBoost, and other classification algorithms.
- Apply leakage-safe preprocessing using Scikit-learn Pipelines.
- Perform hyperparameter tuning.
- Optimize the classification threshold based on business costs.
- Add model explainability using SHAP.
- Develop an interactive dashboard using Streamlit.
- Deploy the model using FastAPI.

## Disclaimer

This project is developed for educational and analytical purposes. Model predictions are not financial advice and should not be used as the sole basis for real-world loan approval or rejection decisions.

## Author## 

**Aamir**

B.Sc. Data Science & Artificial Intelligence Student

- GitHub: [aamir-analyst](https://github.com/aamir-analyst)
- LinkedIn: [Aamir](https://www.linkedin.com/in/aamir-53635738b/)