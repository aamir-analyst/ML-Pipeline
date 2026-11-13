import logging
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.preprocessing.prepare_data import (
    load_and_split_data,
    preprocess_data
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = PROJECT_ROOT / "reports"

ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "regularization"

REPORT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

ARTIFACT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 2. DEFINE REGULARIZATION MODELS
# =====================================================

def get_regularization_models():

    models = {

        "Logistic Regression L1": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            max_iter=2000,
            random_state=42
        ),

        "Logistic Regression L2": LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=1.0,
            max_iter=2000,
            random_state=42
        ),

        "Logistic Regression Balanced": LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            C=1.0,
            max_iter=2000,
            random_state=42
        )

    }

    return models


# =====================================================
# 3. EVALUATE MODEL
# =====================================================

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    results = {

        "Model": model_name,

        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

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

    return results


# =====================================================
# 4. COUNT L1 ZERO COEFFICIENTS
# =====================================================

def count_zero_coefficients(model):

    zero_count = (
        model.coef_ == 0
    ).sum()

    total_count = model.coef_.size

    return int(zero_count), int(total_count)


# =====================================================
# 5. RUN REGULARIZATION EXPERIMENTS
# =====================================================

def run_regularization():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    try:

        logging.info(
            "Starting Regularization Experiments..."
        )

        # Load original split

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = load_and_split_data()

        # Fit preprocessing only on training data

        (
            preprocessor,
            X_train_processed,
            X_test_processed
        ) = preprocess_data(
            X_train,
            X_test
        )

        models = get_regularization_models()

        all_results = []

        for model_name, model in models.items():

            logging.info(
                f"Training {model_name}..."
            )

            # Train

            model.fit(
                X_train_processed,
                y_train
            )

            # Evaluate

            results = evaluate_model(

                model_name,

                model,

                X_test_processed,

                y_test

            )

            all_results.append(results)

            # Save model

            file_name = (
                model_name.lower()
                .replace(" ", "_")
                + ".joblib"
            )

            joblib.dump(
                model,
                ARTIFACT_PATH / file_name
            )

            logging.info(
                f"{model_name} completed successfully."
            )

            # L1 coefficient analysis

            if model_name == "Logistic Regression L1":

                zero_count, total_count = count_zero_coefficients(
                    model
                )

                print(
                    "\nL1 Regularization Feature Analysis:"
                )

                print(
                    f"Zero Coefficients: {zero_count}"
                )

                print(
                    f"Total Coefficients: {total_count}"
                )

                print(
                    f"Zero Coefficient Percentage: "
                    f"{(zero_count / total_count) * 100:.2f}%"
                )

        # Create results table

        results_df = pd.DataFrame(
            all_results
        )

        results_df = results_df.sort_values(
            by="F1 Score",
            ascending=False
        )

        print(
            "\n========== REGULARIZATION RESULTS =========="
        )

        print(
            results_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        # Save results

        output_path = (
            REPORT_PATH / "regularization_results.csv"
        )

        results_df.to_csv(
            output_path,
            index=False
        )

        logging.info(
            f"Results saved: {output_path}"
        )

        print(
            "\nREGULARIZATION EXPERIMENTS COMPLETED!"
        )

    except Exception:

        logging.exception(
            "Regularization Experiments Failed."
        )

        raise


# =====================================================
# 6. EXECUTION
# =====================================================

if __name__ == "__main__":

    run_regularization()