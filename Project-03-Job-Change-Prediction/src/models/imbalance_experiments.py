import logging
from pathlib import Path

import joblib
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.preprocessing.prepare_data import load_and_split_data
from src.preprocessing.preprocessing_pipeline import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = PROJECT_ROOT / "reports"

ARTIFACT_PATH = PROJECT_ROOT / "artifacts"

REPORT_PATH.mkdir(parents=True, exist_ok=True)

ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. CREATE PREPROCESSOR
# =====================================================

def create_preprocessor():

    numerical_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),

            (
                "scaler",
                StandardScaler()
            )

        ]

    )

    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )

        ]

    )

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES
            ),

            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )

        ],

        remainder="drop"

    )

    return preprocessor


# =====================================================
# 3. CREATE EXPERIMENT PIPELINE
# =====================================================

def create_experiment_pipeline(sampler=None):

    steps = [

        (
            "preprocessor",
            create_preprocessor()
        )

    ]

    if sampler is not None:

        steps.append(
            ("sampler", sampler)
        )

    steps.append(

        (
            "model",
            GradientBoostingClassifier(
                random_state=42
            )
        )

    )

    pipeline = ImbPipeline(
        steps=steps
    )

    return pipeline


# =====================================================
# 4. EVALUATE EXPERIMENT
# =====================================================

def evaluate_experiment(
    experiment_name,
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test
):

    logging.info(
        f"Starting experiment: {experiment_name}"
    )

    # Training

    pipeline.fit(
        X_train,
        y_train
    )

    # Prediction

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    # Metrics

    results = {

        "Experiment": experiment_name,

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

    logging.info(
        f"Experiment completed: {experiment_name}"
    )

    return results, pipeline


# =====================================================
# 5. RUN ALL EXPERIMENTS
# =====================================================

def run_experiments():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    try:

        logging.info(
            "Loading original training and testing data..."
        )

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = load_and_split_data()

        print("\nOriginal Training Distribution:")

        print(y_train.value_counts())

        experiments = {

            "Baseline": None,

            "Random OverSampler": RandomOverSampler(
                random_state=42
            ),

            "Random UnderSampler": RandomUnderSampler(
                random_state=42
            )

        }

        all_results = []

        trained_pipelines = {}

        for experiment_name, sampler in experiments.items():

            pipeline = create_experiment_pipeline(
                sampler
            )

            results, trained_pipeline = evaluate_experiment(

                experiment_name,

                pipeline,

                X_train,

                X_test,

                y_train,

                y_test

            )

            all_results.append(results)

            trained_pipelines[experiment_name] = trained_pipeline

            print(
                f"\nCompleted: {experiment_name}"
            )

        # Results DataFrame

        results_df = pd.DataFrame(
            all_results
        )

        results_df = results_df.sort_values(
            by="F1 Score",
            ascending=False
        )

        print("\n========== IMBALANCE EXPERIMENT RESULTS ==========")

        print(
            results_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        # Save results

        output_path = (
            REPORT_PATH / "imbalance_experiment_results.csv"
        )

        results_df.to_csv(
            output_path,
            index=False
        )

        logging.info(
            f"Results saved: {output_path}"
        )

        # Save trained pipelines

        for experiment_name, pipeline in trained_pipelines.items():

            file_name = (
                experiment_name.lower()
                .replace(" ", "_")
                + "_pipeline.joblib"
            )

            joblib.dump(
                pipeline,
                ARTIFACT_PATH / file_name
            )

        print("\nAll experiment pipelines saved successfully.")

        print("\nIMBALANCE EXPERIMENTS COMPLETED!")

    except Exception:

        logging.exception(
            "Imbalance experiments failed."
        )

        raise


# =====================================================
# 6. EXECUTION
# =====================================================

if __name__ == "__main__":

    run_experiments()