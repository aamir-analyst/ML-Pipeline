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

from src.preprocessing.prepare_data import load_and_split_data


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_tuned_model.joblib"

REPORT_PATH = PROJECT_ROOT / "reports"

FIGURE_PATH = REPORT_PATH / "figures"

FIGURE_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 2. LOAD TUNED MODEL
# =====================================================

def load_best_model():

    logging.info("Loading best tuned model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    logging.info("Tuned model loaded successfully.")

    return model


# =====================================================
# 3. EVALUATE MODEL
# =====================================================

def evaluate_final_model(model, X_test, y_test):

    logging.info("Evaluating final model...")

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    results = {

        "Accuracy": accuracy_score(y_test, y_pred),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "ROC AUC": roc_auc_score(
            y_test,
            y_probability
        )

    }

    return results, y_pred, y_probability


# =====================================================
# 4. CONFUSION MATRIX
# =====================================================

def save_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Final Model Confusion Matrix")

    plt.xlabel("Predicted Label")

    plt.ylabel("Actual Label")

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH / "final_confusion_matrix.png"
    )

    plt.close()


# =====================================================
# 5. ROC CURVE
# =====================================================

def save_roc_curve(model, X_test, y_test):

    fig, ax = plt.subplots(figsize=(8, 6))

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax
    )

    ax.set_title("Final Model ROC Curve")

    fig.tight_layout()

    fig.savefig(
        FIGURE_PATH / "final_roc_curve.png"
    )

    plt.close(fig)


# =====================================================
# 6. SAVE RESULTS
# =====================================================

def save_results(results):

    results_df = pd.DataFrame(
        [results]
    )

    results_df.to_csv(
        REPORT_PATH / "final_model_results.csv",
        index=False
    )


# =====================================================
# 7. MAIN EXECUTION
# =====================================================

def run_final_evaluation():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    try:

        logging.info("Starting Final Model Evaluation...")

        X_train, X_test, y_train, y_test = load_and_split_data()

        model = load_best_model()

        results, y_pred, y_probability = evaluate_final_model(
            model,
            X_test,
            y_test
        )

        print("\n========== FINAL MODEL RESULTS ==========")

        for metric, value in results.items():

            print(f"{metric}: {value:.4f}")

        print("\n========== CLASSIFICATION REPORT ==========")

        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        save_confusion_matrix(
            y_test,
            y_pred
        )

        save_roc_curve(
            model,
            X_test,
            y_test
        )

        save_results(results)

        logging.info(
            "Final Model Evaluation Completed Successfully."
        )

        print("\nFinal results and charts saved successfully.")

    except Exception:

        logging.exception(
            "Final Evaluation Failed."
        )

        raise


# =====================================================
# 8. EXECUTION
# =====================================================

if __name__ == "__main__":

    run_final_evaluation()