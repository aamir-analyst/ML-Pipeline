import pandas as pd
import joblib
import json

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# -----------------------------------
# 1. Project Paths
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "loan_delinquent_processed.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"

FIGURE_DIR = REPORT_DIR / "figures"


# Create directories

MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------
# 2. Load Dataset
# -----------------------------------

def load_data():

    df = pd.read_csv(DATA_PATH)

    print("\nDataset Loaded Successfully!")

    print("Dataset Shape:", df.shape)

    return df


# -----------------------------------
# 3. Separate Features and Target
# -----------------------------------

def prepare_data(df):

    X = df.drop(columns=["isDelinquent"])

    y = df["isDelinquent"]

    return X, y


# -----------------------------------
# 4. Train-Test Split
# -----------------------------------

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    print("\nTraining Data:", X_train.shape)

    print("Testing Data:", X_test.shape)

    return X_train, X_test, y_train, y_test


# -----------------------------------
# 5. Train Final Model
# -----------------------------------

def train_final_model(X_train, y_train):

    model = LogisticRegression(

        C=1.0,

        max_iter=1000,

        random_state=42

    )

    model.fit(X_train, y_train)

    print("\nFinal Model Training Completed!")

    return model


# -----------------------------------
# 6. Evaluate Final Model
# -----------------------------------

def evaluate_final_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    print("\n========== FINAL MODEL EVALUATION ==========")

    print("\nAccuracy:", round(accuracy, 4))

    print("\nROC-AUC:", round(roc_auc, 4))

    print("\nClassification Report:")

    print(report)

    print("\nConfusion Matrix:")

    print(cm)

    return y_pred, y_prob, accuracy, roc_auc, report, cm


# -----------------------------------
# 7. Confusion Matrix Visualization
# -----------------------------------

def save_confusion_matrix(y_test, y_pred):

    fig, ax = plt.subplots(figsize=(7, 5))

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=[
            "Non-Delinquent",
            "Delinquent"
        ],
        cmap="Blues",
        ax=ax
    )

    ax.set_title("Loan Delinquency - Confusion Matrix")

    fig.tight_layout()

    output_path = FIGURE_DIR / "confusion_matrix.png"

    fig.savefig(output_path, dpi=300)

    plt.close(fig)

    print("\nConfusion Matrix Saved:", output_path)


# -----------------------------------
# 8. ROC Curve Visualization
# -----------------------------------

def save_roc_curve(model, X_test, y_test):

    fig, ax = plt.subplots(figsize=(7, 5))

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax
    )

    ax.set_title("Loan Delinquency - ROC Curve")

    fig.tight_layout()

    output_path = FIGURE_DIR / "roc_curve.png"

    fig.savefig(output_path, dpi=300)

    plt.close(fig)

    print("\nROC Curve Saved:", output_path)


# -----------------------------------
# 9. Save Final Model
# -----------------------------------

def save_model(model, feature_columns):

    model_path = MODEL_DIR / "loan_delinquency_model.pkl"

    joblib.dump(model, model_path)

    # Save feature information for future predictions

    metadata = {

        "target": "isDelinquent",

        "features": feature_columns,

        "model": "Logistic Regression",

        "random_state": 42

    }

    metadata_path = MODEL_DIR / "model_metadata.json"

    with open(metadata_path, "w") as file:

        json.dump(metadata, file, indent=4)

    print("\nFinal Model Saved Successfully!")

    print("Model:", model_path)

    print("Metadata:", metadata_path)


# -----------------------------------
# 10. Save Evaluation Report
# -----------------------------------

def save_evaluation_report(
    accuracy,
    roc_auc,
    report
):

    report_path = REPORT_DIR / "final_evaluation.txt"

    with open(report_path, "w") as file:

        file.write("LOAN DELINQUENT ANALYSIS\n")

        file.write("=======================\n\n")

        file.write(f"Accuracy: {accuracy:.4f}\n")

        file.write(f"ROC-AUC: {roc_auc:.4f}\n\n")

        file.write("Classification Report:\n")

        file.write(report)

    print("\nEvaluation Report Saved:", report_path)


# -----------------------------------
# 11. Main Execution
# -----------------------------------

if __name__ == "__main__":

    # Load Dataset

    df = load_data()

    # Prepare Features and Target

    X, y = prepare_data(df)

    # Train-Test Split

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train Model

    model = train_final_model(
        X_train,
        y_train
    )

    # Evaluate Model

    (
        y_pred,
        y_prob,
        accuracy,
        roc_auc,
        report,
        cm
    ) = evaluate_final_model(
        model,
        X_test,
        y_test
    )

    # Save Confusion Matrix

    save_confusion_matrix(
        y_test,
        y_pred
    )

    # Save ROC Curve

    save_roc_curve(
        model,
        X_test,
        y_test
    )

    # Save Model and Metadata

    save_model(
        model,
        X.columns.tolist()
    )

    # Save Evaluation Report

    save_evaluation_report(
        accuracy,
        roc_auc,
        report
    )

    print("\n====================================")

    print("FINAL MODEL PIPELINE COMPLETED!")

    print("====================================")