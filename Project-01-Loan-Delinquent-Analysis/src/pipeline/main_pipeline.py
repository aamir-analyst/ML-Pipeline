import sys
import time
import logging
import joblib
import json

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    make_scorer,
    recall_score
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Loan_Delinquent_Dataset (1).csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "loan_delinquent_processed.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"

FIGURE_DIR = REPORT_DIR / "figures"

LOG_DIR = PROJECT_ROOT / "logs"


for directory in [
    MODEL_DIR,
    REPORT_DIR,
    FIGURE_DIR,
    LOG_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOGGING CONFIGURATION
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "pipeline.log",
            encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# ==================================================
# 3. DATA LOADING
# ==================================================

def load_data():

    logger.info("STEP 1: Loading Dataset")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    logger.info(f"Dataset Shape: {df.shape}")

    logger.info("Dataset Loaded Successfully")

    return df


# ==================================================
# 4. DATA UNDERSTANDING
# ==================================================

def data_overview(df):

    logger.info("STEP 2: Data Understanding")

    logger.info(f"Rows: {df.shape[0]}")

    logger.info(f"Columns: {df.shape[1]}")

    logger.info("\nMissing Values:\n%s", df.isnull().sum())

    logger.info(
        "Duplicate Rows: %s",
        df.duplicated().sum()
    )

    logger.info(
        "Target Distribution:\n%s",
        df["isDelinquent"].value_counts()
    )


# ==================================================
# 5. DATA CLEANING
# ==================================================

def clean_data(df):

    logger.info("STEP 3: Data Cleaning")

    data = df.copy()

    data.drop(
        columns=["ID"],
        inplace=True
    )

    data["purpose"] = (
        data["purpose"]
        .str.strip()
        .str.title()
    )

    logger.info("Data Cleaning Completed")

    return data


# ==================================================
# 6. CATEGORICAL ENCODING
# ==================================================

def encode_data(df):

    logger.info("STEP 4: Categorical Encoding")

    data = pd.get_dummies(
        df,
        columns=[
            "term",
            "gender",
            "purpose",
            "home_ownership",
            "age",
            "FICO"
        ],
        drop_first=True,
        dtype=int
    )

    logger.info(f"Encoded Dataset Shape: {data.shape}")

    return data


# ==================================================
# 7. SAVE PROCESSED DATA
# ==================================================

def save_processed_data(df):

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    logger.info("Processed Dataset Saved")


# ==================================================
# 8. FEATURES AND TARGET
# ==================================================

def prepare_data(df):

    logger.info("STEP 5: Preparing Features and Target")

    X = df.drop(columns=["isDelinquent"])

    y = df["isDelinquent"]

    return X, y


# ==================================================
# 9. TRAIN TEST SPLIT
# ==================================================

def split_data(X, y):

    logger.info("STEP 6: Train Test Split")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    logger.info(f"Training Shape: {X_train.shape}")

    logger.info(f"Testing Shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ==================================================
# 10. MODEL TRAINING
# ==================================================

def train_models(X_train, y_train):

    logger.info("STEP 7: Model Training Experiments")

    models = {}

    # Baseline

    models["Baseline Logistic Regression"] = (
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
    )

    # Balanced

    models["Balanced Logistic Regression"] = (
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    )

    # Strong Regularization

    models["Strong Regularization"] = (
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    )

    # Oversampling

    models["Oversampling"] = ImbPipeline([
        (
            "sampler",
            RandomOverSampler(random_state=42)
        ),
        (
            "model",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        )
    ])

    # Undersampling

    models["Undersampling"] = ImbPipeline([
        (
            "sampler",
            RandomUnderSampler(random_state=42)
        ),
        (
            "model",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        )
    ])

    for name, model in models.items():

        model.fit(X_train, y_train)

        logger.info(f"{name} Training Completed")

    return models


# ==================================================
# 11. MODEL EVALUATION
# ==================================================

def evaluate_models(models, X_test, y_test):

    logger.info("STEP 8: Model Evaluation")

    results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        results.append({

            "Model": name,

            "Accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            "Precision": report["1"]["precision"],

            "Recall": report["1"]["recall"],

            "F1 Score": report["1"]["f1-score"],

            "ROC-AUC": roc_auc_score(
                y_test,
                y_prob
            )

        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "pipeline_model_comparison.csv",
        index=False
    )

    logger.info(
        "\nModel Comparison:\n%s",
        results_df.to_string(index=False)
    )

    return results_df


# ==================================================
# 12. STRATIFIED K-FOLD CROSS VALIDATION
# ==================================================

def cross_validation(X_train, y_train):

    logger.info("STEP 9: 5-Fold Cross Validation")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = {

        "Baseline": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ),

        "Balanced": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),

        "Oversampling": ImbPipeline([
            (
                "sampler",
                RandomOverSampler(random_state=42)
            ),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42
                )
            )
        ])

    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": make_scorer(
            recall_score,
            pos_label=1
        ),
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results = []

    for name, model in models.items():

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results.append({

            "Model": name,

            "Accuracy": scores["test_accuracy"].mean(),

            "Precision": scores["test_precision"].mean(),

            "Recall": scores["test_recall"].mean(),

            "F1 Score": scores["test_f1"].mean(),

            "ROC-AUC": scores["test_roc_auc"].mean()

        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "pipeline_cross_validation.csv",
        index=False
    )

    logger.info(
        "\nCross Validation Results:\n%s",
        results_df.to_string(index=False)
    )

    return results_df


# ==================================================
# 13. FINAL MODEL SELECTION AND EVALUATION
# ==================================================

def final_evaluation(models, X_test, y_test):

    logger.info("STEP 10: Final Model Evaluation")

    # Baseline selected based on the project's
    # primary objective: delinquent-class recall.

    final_model = models["Baseline Logistic Regression"]

    y_pred = final_model.predict(X_test)

    y_prob = final_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"Final Accuracy: {accuracy:.4f}")

    logger.info(f"Final ROC-AUC: {roc_auc:.4f}")

    logger.info("\n%s", report)

    logger.info("\nConfusion Matrix:\n%s", cm)

    # Save Confusion Matrix

    fig, ax = plt.subplots(figsize=(7, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=[
            "Non-Delinquent",
            "Delinquent"
        ],
        cmap="Blues",
        ax=ax
    )

    ax.set_title("Loan Delinquency Confusion Matrix")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close(fig)

    # Save ROC Curve

    fig, ax = plt.subplots(figsize=(7, 5))

    RocCurveDisplay.from_estimator(
        final_model,
        X_test,
        y_test,
        ax=ax
    )

    ax.set_title("Loan Delinquency ROC Curve")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "roc_curve.png",
        dpi=300
    )

    plt.close(fig)

    # Save Evaluation Report

    with open(
        REPORT_DIR / "pipeline_final_evaluation.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("LOAN DELINQUENT ANALYSIS\n\n")

        file.write(f"Accuracy: {accuracy:.4f}\n")

        file.write(f"ROC-AUC: {roc_auc:.4f}\n\n")

        file.write(report)

    return final_model


# ==================================================
# 14. SAVE FINAL MODEL
# ==================================================

def save_final_model(model, feature_columns):

    logger.info("STEP 11: Saving Final Model")

    model_path = MODEL_DIR / "loan_delinquency_model.pkl"

    metadata = {

        "model": "Logistic Regression",

        "target": "isDelinquent",

        "features": feature_columns,

        "random_state": 42

    }

    joblib.dump(
        model,
        model_path
    )

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    logger.info("Final Model Saved Successfully")


# ==================================================
# 15. MAIN PIPELINE
# ==================================================

def main():

    start_time = time.time()

    print("\n")
    print("=" * 55)
    print("       LOAN DELINQUENT ANALYSIS PIPELINE")
    print("=" * 55)

    try:

        # Step 1: Load Dataset

        df = load_data()

        # Step 2: Data Overview

        data_overview(df)

        # Step 3: Data Cleaning

        cleaned_df = clean_data(df)

        # Step 4: Encoding

        encoded_df = encode_data(cleaned_df)

        # Save Processed Dataset

        save_processed_data(encoded_df)

        # Step 5: Features and Target

        X, y = prepare_data(encoded_df)

        # Step 6: Train Test Split

        X_train, X_test, y_train, y_test = split_data(X, y)

        # Step 7: Model Training

        models = train_models(X_train, y_train)

        # Step 8: Model Comparison

        evaluate_models(models, X_test, y_test)

        # Step 9: Cross Validation

        cross_validation(X_train, y_train)

        # Step 10: Final Evaluation

        final_model = final_evaluation(
            models,
            X_test,
            y_test
        )

        # Step 11: Save Model

        save_final_model(
            final_model,
            X.columns.tolist()
        )

        elapsed_time = time.time() - start_time

        print("\n")
        print("=" * 55)
        print("       PIPELINE EXECUTED SUCCESSFULLY!")
        print("=" * 55)

        print(f"\nTotal Execution Time: {elapsed_time:.2f} seconds")

        print("\nGenerated Files:")

        print("- Processed Dataset")

        print("- Model Comparison")

        print("- Cross Validation Results")

        print("- Final Evaluation Report")

        print("- Confusion Matrix")

        print("- ROC Curve")

        print("- Trained Model (.pkl)")

        print("\nPipeline Completed Successfully!")

    except Exception:

        logger.exception("PIPELINE EXECUTION FAILED")

        raise


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    main()