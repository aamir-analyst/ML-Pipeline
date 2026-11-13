# 🍷 Wine Quality Prediction

### Machine Learning-Based Premium Wine Classification

A Machine Learning project that predicts whether a red wine is **Premium or Non-Premium** based on its physicochemical properties.

The project uses data analysis, preprocessing, feature engineering, multiple classification algorithms, hyperparameter tuning, and model evaluation to identify important factors associated with premium wine quality.

---

## 1. Project Overview

The wine industry traditionally relies on manual wine tasting to evaluate quality. This process can be time-consuming, expensive, and subjective.

This project aims to develop a Machine Learning solution that can classify wine quality using measurable physicochemical attributes.

The project is based on a business case study involving a red wine company called **Moonshine**, which wants to identify premium-quality wines more efficiently.

The classification model uses 11 physicochemical features to predict whether a wine belongs to the Premium or Non-Premium category.

---

## 2. Business Problem

The company wants to understand:

- Which physicochemical properties are associated with premium-quality wine?
- Can Machine Learning identify premium wines?
- Which classification algorithm performs best?
- How can model performance be evaluated?

### Project Objectives

1. Perform Exploratory Data Analysis (EDA).
2. Clean and preprocess the dataset.
3. Engineer a binary classification target.
4. Train multiple Machine Learning models.
5. Evaluate and compare model performance.
6. Perform hyperparameter tuning.
7. Identify important features.
8. Select and save the best-performing model.

---

## 3. Dataset Description

The project uses a Red Wine Quality dataset.

### Dataset Information

| Property | Description |
|---|---|
| Dataset | Red Wine Quality |
| Original Records | 1,599 |
| Records After Duplicate Removal | 1,359 |
| Total Columns | 12 |
| Input Features | 11 |
| Target | Quality |
| Problem Type | Binary Classification |

### Input Features

| Feature | Description |
|---|---|
| Fixed Acidity | Fixed acids present in wine |
| Volatile Acidity | Volatile acids present in wine |
| Citric Acid | Citric acid content |
| Residual Sugar | Sugar remaining after fermentation |
| Chlorides | Salt content |
| Free Sulfur Dioxide | Free sulfur dioxide concentration |
| Total Sulfur Dioxide | Total sulfur dioxide concentration |
| Density | Density of wine |
| pH | Acidity level |
| Sulphates | Sulphate content |
| Alcohol | Alcohol percentage |

### Target Engineering

The original `quality` column contains wine quality scores.

For this project, the target was converted into a binary classification problem.

```python
df["is_premium"] = (df["quality"] >= 7).astype(int)
```

Classification rules:

| Quality Score | Category | Target |
|---|---|---|
| Below 7 | Non-Premium | 0 |
| 7 or Above | Premium | 1 |

### Target Distribution

After duplicate removal:

| Category | Records |
|---|---:|
| Non-Premium | 1,175 |
| Premium | 184 |
| Total | 1,359 |

The dataset is imbalanced, with significantly fewer premium wine samples.

---

## 4. Technology Stack

### Programming Language
- Python 3

### Data Analysis
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn

### Model Persistence
- Joblib

### Development Tools
- VS Code
- Jupyter Notebook
- Git
- GitHub

### Testing
- Pytest

---

## 5. Project Architecture

```text
Wine-Quality-Prediction/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   └── winequality.csv
│   │
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   └── 03_Model_Experiments.ipynb
│
├── src/
│   ├── data/
│   │   └── data_loader.py
│   │
│   ├── eda/
│   │   └── eda.py
│   │
│   ├── preprocessing/
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── tune.py
│   │   └── feature_importance.py
│   │
│   ├── pipeline/
│   │   └── predict.py
│   │
│   └── utils/
│       └── logger.py
│
├── models/
│   └── best_wine_quality_model.joblib
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── logs/
│
├── tests/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the dataset and identify patterns.

### EDA Activities

- Dataset structure and statistical summary
- Missing value analysis
- Duplicate record analysis
- Target distribution analysis
- Premium vs Non-Premium distribution
- Correlation analysis
- Feature distribution analysis
- Skewness analysis
- Outlier detection

### Key Findings

1. The original dataset contained 1,599 records.
2. After removing duplicates, 1,359 records remained.
3. No missing values were found.
4. The dataset contains an imbalanced target distribution.
5. Alcohol, sulphates, and citric acid showed positive correlation with the original quality score.
6. Volatile acidity showed negative correlation with the original quality score.
7. Outliers were detected but retained because unusual physicochemical values may represent valid wine samples.

### Generated EDA Visualizations

The pipeline generates:

- Quality Distribution
- Premium Distribution
- Feature Histograms
- Correlation Heatmap
- Outlier Boxplots
- Alcohol vs Quality
- Volatile Acidity vs Quality

Visualizations are saved in:

```text
reports/figures/
```

---

## 7. Data Preprocessing

The following preprocessing operations were performed:

1. Dataset loading.
2. Duplicate removal.
3. Missing value analysis.
4. Missing value treatment.
5. Outlier detection.
6. Binary target creation.
7. Feature and target separation.
8. Train-test splitting.

### Train-Test Split

| Dataset | Records |
|---|---:|
| Training Data | 1,087 |
| Testing Data | 272 |
| Total | 1,359 |

The dataset was divided using an 80:20 split with stratification to preserve class proportions.

---

## 8. Machine Learning Models

Five classification algorithms were trained and evaluated.

| Model | Type |
|---|---|
| Logistic Regression | Linear Classifier |
| Decision Tree | Tree-Based Classifier |
| Random Forest | Ensemble Learning |
| Gradient Boosting | Ensemble Learning |
| Support Vector Machine (SVM) | Kernel-Based Classifier |

### Model Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross-Validation F1 Score

Because the dataset is imbalanced, F1 Score and Recall are particularly important for evaluating premium wine classification.

---

## 9. Hyperparameter Tuning

GridSearchCV was used to tune:

- Random Forest
- Support Vector Machine (SVM)

### Best Parameters: Random Forest

```python
{
    "max_depth": 10,
    "min_samples_leaf": 2,
    "n_estimators": 200
}
```

### Best Parameters: SVM

```python
{
    "model__C": 1,
    "model__gamma": "scale",
    "model__kernel": "rbf"
}
```

The models were tuned using F1 Score as the optimization metric.

---

## 10. Model Comparison

### Tuned Model Results

| Metric | Random Forest | SVM |
|---|---:|---:|
| Accuracy | 86.40% | 80.51% |
| Precision | 50.00% | 39.19% |
| Recall | 62.16% | 78.38% |
| F1 Score | 55.42% | 52.25% |
| ROC-AUC | 88.18% | 88.38% |
| CV F1 Score | 50.23% | 51.46% |

### Best Model Selection

**Selected Model: Support Vector Machine (SVM)**

The SVM model was selected based on its higher Cross-Validation F1 Score.

| Model | CV F1 Score |
|---|---:|
| SVM | 51.46% |
| Random Forest | 50.23% |

Although Random Forest achieved a higher test F1 Score, SVM achieved a higher Cross-Validation F1 Score.

Therefore, SVM was selected according to the project's predefined model selection criterion.

The selected model is saved as:

```text
models/best_wine_quality_model.joblib
```

---

## 11. Feature Importance Analysis

Permutation Feature Importance was used to understand how individual features affect the selected SVM model's F1 Score.

### Feature Importance Ranking

| Rank | Feature | Importance Mean |
|---:|---|---:|
| 1 | Alcohol | 0.102721 |
| 2 | Sulphates | 0.078157 |
| 3 | Volatile Acidity | 0.026181 |
| 4 | Chlorides | 0.024128 |
| 5 | Density | 0.010132 |
| 6 | Free Sulfur Dioxide | 0.008084 |
| 7 | Citric Acid | 0.006504 |
| 8 | Total Sulfur Dioxide | 0.004294 |
| 9 | Residual Sugar | -0.000955 |
| 10 | pH | -0.003966 |
| 11 | Fixed Acidity | -0.031001 |

### Key Findings

- Alcohol was the most important feature for the selected SVM model.
- Sulphates ranked second.
- Volatile Acidity and Chlorides also contributed to predictions.
- Some features showed near-zero or negative permutation importance.

Permutation importance measures the impact of shuffling a feature on model performance. It does not establish a causal relationship.

Generated files:

```text
reports/
├── figures/
│   └── 08_permutation_feature_importance.png
│
└── metrics/
    └── permutation_feature_importance.csv
```

---

## 12. Final Model Performance

The selected SVM model achieved the following results on the test dataset:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 80.51% |
| Precision | 39.19% |
| Recall | 78.38% |
| F1 Score | 52.25% |
| ROC-AUC | 88.38% |
| Cross-Validation F1 | 51.46% |

### Confusion Matrix

```text
[[190  45]
 [  8  29]]
```

Interpretation:

- 190 Non-Premium wines were correctly classified.
- 45 Non-Premium wines were incorrectly classified as Premium.
- 8 Premium wines were incorrectly classified as Non-Premium.
- 29 Premium wines were correctly classified.

The model identified 29 out of 37 actual premium wines in the test dataset.

---

## 13. How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/aamir-analyst/ML-Pipeline.git
```

### Step 2: Navigate to the Project

```bash
cd ML-Pipeline/Wine-Quality-Prediction
```

### Step 3: Create Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate Virtual Environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Complete Pipeline

```bash
python main.py
```

The pipeline will:

1. Load the dataset.
2. Generate EDA reports.
3. Perform preprocessing.
4. Engineer classification features.
5. Train classification models.
6. Evaluate model performance.
7. Tune selected models.
8. Select the best model.
9. Save the trained model.
10. Generate permutation feature importance reports.

### Step 7: Run Predictions

```bash
python -m src.pipeline.predict
```

The prediction module loads the saved model and accepts physicochemical feature values for classification.

---

## 14. Testing

Pytest was used to validate important project components.

Tests cover:

- Dataset loading
- Duplicate removal
- Feature engineering
- Missing value handling
- Train-test splitting
- Model training
- Prediction functionality

Run tests using:

```bash
pytest -v
```

---

## 15. Logging and Reports

The project includes a logging system to track pipeline execution.

Log file:

```text
logs/wine_quality.log
```

Generated reports include:

```text
reports/
├── figures/
└── metrics/
```

These reports contain EDA visualizations, outlier summaries, model comparison results, and feature importance analysis.

---

## 16. Business Conclusion

The project demonstrates that Machine Learning can be used to classify premium and non-premium wines based on physicochemical properties.

The analysis identified Alcohol and Sulphates as the two most important features for the selected SVM model.

The selected SVM achieved:

- 80.51% Accuracy
- 78.38% Premium Class Recall
- 52.25% F1 Score
- 88.38% ROC-AUC

The model successfully identified many premium wine samples, but its precision of 39.19% indicates that it also classified several non-premium wines as premium.

Therefore, the model can serve as a decision-support tool, but it should not replace professional wine quality assessment without further validation and improvement.

### Future Improvements

- Improve premium-class precision.
- Explore additional classification algorithms.
- Perform more extensive hyperparameter optimization.
- Evaluate probability calibration.
- Test the model on independent wine datasets.
- Develop a web-based prediction application.

---

## 17. Author

**Aamir**

B.Sc. Data Science & Artificial Intelligence Student

### Skills Demonstrated

- Python Programming
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Machine Learning
- Ensemble Learning
- Model Evaluation
- Hyperparameter Tuning
- Feature Importance
- Modular Python Development
- Unit Testing
- Git and GitHub

---

## 18. License

This project was developed for educational and learning purposes.