import logging
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)

from src.preprocessing.prepare_data import load_and_split_data, preprocess_data


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "models"

REPORT_PATH = PROJECT_ROOT / "reports"

FIGURE_PATH = REPORT_PATH / "figures"

FIGURE_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 2. LOAD TRAINED MODELS
# =====================================================

def load_trained_models():

    logging.info("Loading trained models...")

    models = {}

    for model_file in MODEL_PATH.glob("*.joblib"):

        model_name = model_file.stem

        models[model_name] = joblib.load(
            model_file
        )

        logging.info(
            f"Loaded model: {model_name}"
        )

    if not models:

        raise FileNotFoundError(
            "No trained models found."
        )

    return models


# =====================================================
# 3. EVALUATE MODELS
# =====================================================

def evaluate_models(
    models,
    X_test,
    y_test
):

    logging.info("Starting Model Evaluation...")

    results = []

    for model_name, model in models.items():

        logging.info(
            f"Evaluating {model_name}..."
        )

        y_pred = model.predict(X_test)

        y_probability = model.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

        results.append({

            "Model": model_name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1,

            "ROC AUC": roc_auc

        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    )

    return results_df


# =====================================================
# 4. CONFUSION MATRIX
# =====================================================

def plot_confusion_matrices(
    models,
    X_test,
    y_test
):

    logging.info("Generating Confusion Matrices...")

    for model_name, model in models.items():

        y_pred = model.predict(X_test)

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(
            f"Confusion Matrix - {model_name}"
        )

        plt.xlabel("Predicted Label")

        plt.ylabel("Actual Label")

        plt.tight_layout()

        plt.savefig(
            FIGURE_PATH /
            f"{model_name}_confusion_matrix.png"
        )

        plt.close()


# =====================================================
# 5. ROC CURVE
# =====================================================

def plot_roc_curves(
    models,
    X_test,
    y_test
):

    logging.info("Generating ROC Curves...")

    plt.figure(figsize=(9, 7))

    for model_name, model in models.items():

        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            name=model_name
        )

    plt.title("ROC Curve Comparison")

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH / "roc_curve_comparison.png"
    )

    plt.close()


# =====================================================
# 6. SAVE RESULTS
# =====================================================

def save_results(results_df):

    output_path = (
        REPORT_PATH / "model_evaluation_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    logging.info(
        f"Evaluation results saved: {output_path}"
    )


# =====================================================
# 7. COMPLETE EVALUATION PIPELINE
# =====================================================

def run_evaluation():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    try:

        logging.info("Starting Evaluation Pipeline...")

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = load_and_split_data()

        (
            preprocessor,
            X_train_processed,
            X_test_processed
        ) = preprocess_data(
            X_train,
            X_test
        )

        models = load_trained_models()

        results_df = evaluate_models(
            models,
            X_test_processed,
            y_test
        )

        print("\n========== MODEL EVALUATION RESULTS ==========")

        print(
            results_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        best_model = results_df.iloc[0]

        print("\n========== BEST BASELINE MODEL ==========")

        print("Model:", best_model["Model"])

        print("F1 Score:", round(best_model["F1 Score"], 4))

        plot_confusion_matrices(
            models,
            X_test_processed,
            y_test
        )

        plot_roc_curves(
            models,
            X_test_processed,
            y_test
        )

        save_results(results_df)

        print("\nEvaluation completed successfully!")

    except Exception:

        logging.exception(
            "Evaluation Pipeline Failed."
        )

        raise


# =====================================================
# 8. EXECUTION
# =====================================================

if __name__ == "__main__":

    run_evaluation()