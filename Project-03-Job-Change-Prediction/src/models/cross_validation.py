import logging
from pathlib import Path

import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate

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

REPORT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

N_SPLITS = 5

RANDOM_STATE = 42


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

    return ColumnTransformer(

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

        ]

    )


# =====================================================
# 3. CREATE MODEL PIPELINE
# =====================================================

def create_model_pipeline():

    pipeline = ImbPipeline(

        steps=[

            (
                "preprocessor",
                create_preprocessor()
            ),

            (
                "sampler",
                RandomOverSampler(
                    random_state=RANDOM_STATE
                )
            ),

            (
                "model",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE
                )
            )

        ]

    )

    return pipeline


# =====================================================
# 4. RUN CROSS VALIDATION
# =====================================================

def run_cross_validation():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    try:

        logging.info(
            "Starting Stratified K-Fold Cross Validation..."
        )

        # Load training data only.
        # Final test set remains untouched.

        X_train, X_test, y_train, y_test = load_and_split_data()

        cv = StratifiedKFold(

            n_splits=N_SPLITS,

            shuffle=True,

            random_state=RANDOM_STATE

        )

        pipeline = create_model_pipeline()

        scoring = {

            "accuracy": "accuracy",

            "precision": "precision",

            "recall": "recall",

            "f1": "f1",

            "roc_auc": "roc_auc"

        }

        logging.info(
            "Executing 5-Fold Cross Validation..."
        )

        cv_results = cross_validate(

            pipeline,

            X_train,

            y_train,

            cv=cv,

            scoring=scoring,

            n_jobs=1,

            return_train_score=False

        )

        # Prepare results

        results = {

            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC AUC"
            ],

            "Mean Score": [

                cv_results["test_accuracy"].mean(),

                cv_results["test_precision"].mean(),

                cv_results["test_recall"].mean(),

                cv_results["test_f1"].mean(),

                cv_results["test_roc_auc"].mean()

            ],

            "Standard Deviation": [

                cv_results["test_accuracy"].std(),

                cv_results["test_precision"].std(),

                cv_results["test_recall"].std(),

                cv_results["test_f1"].std(),

                cv_results["test_roc_auc"].std()

            ]

        }

        results_df = pd.DataFrame(results)

        print("\n========== 5-FOLD CROSS VALIDATION RESULTS ==========")

        print(

            results_df.to_string(

                index=False,

                float_format=lambda x: f"{x:.4f}"

            )

        )

        # Save results

        output_path = (
            REPORT_PATH / "cross_validation_results.csv"
        )

        results_df.to_csv(

            output_path,

            index=False

        )

        logging.info(
            f"Cross validation results saved: {output_path}"
        )

        print("\nCROSS VALIDATION COMPLETED SUCCESSFULLY!")

    except Exception:

        logging.exception(
            "Cross Validation Failed."
        )

        raise


# =====================================================
# 5. EXECUTION
# =====================================================

if __name__ == "__main__":

    run_cross_validation()