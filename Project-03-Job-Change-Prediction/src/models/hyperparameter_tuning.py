import logging
from pathlib import Path

import joblib
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV
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

ARTIFACT_PATH = PROJECT_ROOT / "artifacts"

REPORT_PATH = PROJECT_ROOT / "reports"

ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)

REPORT_PATH.mkdir(parents=True, exist_ok=True)


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
# 3. CREATE PIPELINE
# =====================================================

def create_pipeline():

    pipeline = ImbPipeline(

        steps=[

            (
                "preprocessor",
                create_preprocessor()
            ),

            (
                "sampler",
                RandomOverSampler(
                    random_state=42
                )
            ),

            (
                "model",
                GradientBoostingClassifier(
                    random_state=42
                )
            )

        ]

    )

    return pipeline


# =====================================================
# 4. HYPERPARAMETER SEARCH
# =====================================================

def run_hyperparameter_tuning():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    try:

        logging.info(
            "Starting Hyperparameter Tuning..."
        )

        X_train, X_test, y_train, y_test = load_and_split_data()

        pipeline = create_pipeline()

        parameter_grid = {

            "model__n_estimators": [
                50,
                100,
                150,
                200
            ],

            "model__learning_rate": [
                0.01,
                0.05,
                0.1,
                0.2
            ],

            "model__max_depth": [
                2,
                3,
                4,
                5
            ],

            "model__min_samples_split": [
                2,
                5,
                10
            ],

            "model__min_samples_leaf": [
                1,
                2,
                4
            ]

        }

        cv = StratifiedKFold(

            n_splits=5,

            shuffle=True,

            random_state=42

        )

        search = RandomizedSearchCV(

            estimator=pipeline,

            param_distributions=parameter_grid,

            n_iter=15,

            scoring="f1",

            cv=cv,

            random_state=42,

            n_jobs=1,

            verbose=2,

            refit=True

        )

        logging.info(
            "Executing RandomizedSearchCV..."
        )

        search.fit(
            X_train,
            y_train
        )

        print("\n========== HYPERPARAMETER TUNING RESULTS ==========")

        print(
            "\nBest Cross Validation F1 Score:",
            round(search.best_score_, 4)
        )

        print("\nBest Parameters:")

        for parameter, value in search.best_params_.items():

            print(f"{parameter}: {value}")

        # Save best model

        model_path = (
            ARTIFACT_PATH / "best_tuned_model.joblib"
        )

        joblib.dump(
            search.best_estimator_,
            model_path
        )

        # Save results

        results_df = pd.DataFrame(
            search.cv_results_
        )

        results_df = results_df.sort_values(
            by="rank_test_score"
        )

        results_df.to_csv(
            REPORT_PATH / "hyperparameter_tuning_results.csv",
            index=False
        )

        logging.info(
            f"Best model saved: {model_path}"
        )

        print("\nBest Model Saved Successfully!")

        print("\nHYPERPARAMETER TUNING COMPLETED!")

    except Exception:

        logging.exception(
            "Hyperparameter Tuning Failed."
        )

        raise


# =====================================================
# 5. EXECUTION
# =====================================================

if __name__ == "__main__":

    run_hyperparameter_tuning()