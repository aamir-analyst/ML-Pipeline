# Bike Sharing Prediction using Ensemble Techniques

## Project Overview

This project focuses on analyzing bike-sharing rental patterns and predicting the total number of bikes rented using ensemble machine learning techniques.

The project is based on historical bike-sharing data containing weather conditions, seasonal information, holidays, working days, and hourly rental records.

The objective is to help **Travel Along**, a bike-sharing company, understand customer demand patterns and make better operational decisions, especially during surge hours.

## Business Problem

Travel Along wants to expand its customer base and provide better services at a reasonable cost.

Using historical bike rental data, the goal is to:

- Identify factors affecting bike rental demand.
- Understand seasonal, weather, and hourly rental patterns.
- Predict the total number of bikes rented.
- Provide business recommendations for demand planning.

## Dataset Information

The dataset contains historical bike-sharing records from 2011 and 2012, collected from the Capital Bikeshare system in Washington, D.C., USA.

Two datasets are provided:

| Dataset | Records | Description |
|---|---:|---|
| day.csv | 731 | Daily bike rental records |
| hour.csv | 17,379 | Hourly bike rental records |

### Target Variable

`cnt` – Total number of rental bikes, including both casual and registered users.

### Important Features

| Feature | Description |
|---|---|
| season | Season of the year |
| yr | Year |
| mnth | Month |
| hr | Hour of the day |
| holiday | Whether the day is a holiday |
| weekday | Day of the week |
| workingday | Whether the day is a working day |
| weathersit | Weather situation |
| temp | Normalized temperature |
| atemp | Normalized feeling temperature |
| hum | Normalized humidity |
| windspeed | Normalized wind speed |

## Project Workflow

1. Data Loading
2. Dataset Overview
3. Data Quality Checks
4. Exploratory Data Analysis (EDA)
5. Data Preparation
6. Train-Test Split
7. Ensemble Model Training
8. Model Evaluation
9. Feature Importance Analysis
10. Business Recommendations

## Exploratory Data Analysis

EDA was performed on both daily and hourly datasets.

### Data Quality

- No missing values found.
- No duplicate records found.
- Target variable validation completed successfully.
- Verified that `cnt = casual + registered`.

### Visualizations

The following visualizations were performed:

- Daily and hourly rental distribution.
- Season vs bike rentals.
- Weather situation vs bike rentals.
- Working day vs bike rentals.
- Weekday vs bike rentals.
- Temperature, humidity, and windspeed analysis.
- Hourly demand analysis.
- Holiday vs bike rentals.

### Key EDA Findings

- Fall season showed comparatively higher rental demand.
- Clear weather conditions were associated with higher rentals.
- Temperature showed a positive relationship with rental demand.
- Morning and evening hours showed higher bike rental activity.
- Evening peak demand was observed around 5 PM.
- Working days showed higher demand during certain hours.

## Data Preparation

The hourly dataset was selected for model training.

The following columns were removed:

| Column | Reason |
|---|---|
| instant | Record identifier |
| dteday | Date column |
| casual | Target leakage |
| registered | Target leakage |

The target variable is `cnt`.

### Input Features

12 features were used for model training.

### Train-Test Split

A chronological 80:20 split was applied without shuffling.

| Dataset | Records |
|---|---:|
| Training | 13,903 |
| Testing | 3,476 |
| Total | 17,379 |

## Machine Learning Models

Three ensemble regression techniques were implemented.

### 1. Bagging

Bagging Regressor with Decision Tree estimators.

### 2. Boosting

Gradient Boosting Regressor.

### 3. Stacking

Stacking Regressor combining Bagging and Gradient Boosting models with Linear Regression as the final estimator.

## Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Model Comparison Results

| Model | MAE | MSE | RMSE | R² Score |
|---|---:|---:|---:|---:|
| Bagging | 44.99 | 4876.48 | 69.83 | 0.8997 |
| Stacking | 52.11 | 6020.22 | 77.59 | 0.8762 |
| Boosting | 73.26 | 11629.60 | 107.84 | 0.7608 |

### Best Performing Model

**Bagging Regressor**

- MAE: 44.99
- RMSE: 69.83
- R² Score: 0.8997

Bagging achieved the best performance among the three evaluated models.

The trained best model is saved in:

`artifacts/best_model.joblib`

## Feature Importance Analysis

Feature importance was analyzed using the selected Bagging model.

| Feature | Importance |
|---|---:|
| hr | 60.64% |
| temp | 9.39% |
| atemp | 7.33% |
| yr | 6.88% |
| workingday | 5.65% |
| hum | 2.81% |
| weathersit | 1.94% |
| mnth | 1.38% |
| season | 1.33% |
| weekday | 1.31% |
| windspeed | 1.08% |
| holiday | 0.25% |

The hour of the day was the most important feature for predicting bike rental demand.

## Business Recommendations

Based on the EDA and model results:

1. Maintain additional bike availability during morning and evening peak hours.
2. Use weather conditions and temperature patterns for demand planning.
3. Adjust operational planning according to working-day demand.
4. Optimize bike availability during low-demand hours.
5. Use predicted rental counts to support operational decisions during surge hours.

## Project Structure

```text
```text
Project-04-Bike-Sharing-Prediction/
│
├── artifacts/
│   └── best_model.joblib
│
├── data/
│   ├── raw/
│   │   ├── day.csv
│   │   ├── hour.csv
│   │   └── Readme.txt
│   │
│   └── processed/
│
├── logs/
│
├── reports/
│   ├── figures/
│   │   ├── target_distribution.png
│   │   ├── season_vs_rentals.png
│   │   ├── weather_vs_rentals.png
│   │   ├── workingday_vs_rentals.png
│   │   ├── weekday_vs_rentals.png
│   │   ├── weather_factors_vs_rentals.png
│   │   ├── hourly_demand.png
│   │   ├── holiday_vs_rentals.png
│   │   └── feature_importance.png
│   │
│   ├── model_evaluation_results.csv
│   └── feature_importance.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   └── visualization.py
│
├── main.py
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/aamir-analyst/ML-Pipeline.git
```

### 2. Navigate to the Project Folder

```bash
cd ML-Pipeline/Project-04-Bike-Sharing-Prediction
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 4. Run the Project

```bash
python main.py
```

The project will execute data loading, quality checks, EDA, data preparation, model training, evaluation, and feature importance analysis.

## Conclusion

This project demonstrates how ensemble machine learning techniques can be used to analyze and predict bike rental demand.

Among the three evaluated models, Bagging Regressor achieved the best performance with an R² score of approximately 89.97%.

The analysis also identified the hour of the day as the most influential feature in predicting bike rental demand.

These findings can help Travel Along improve demand planning and make better operational decisions.