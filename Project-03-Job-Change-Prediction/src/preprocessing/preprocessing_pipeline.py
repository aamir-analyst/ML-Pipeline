import logging
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_data.csv"
)


# =====================================================
# 2. FEATURE CONFIGURATION
# =====================================================

TARGET_COLUMN = "target"

NUMERICAL_FEATURES = [
    "city_development_index",
    "training_hours"
]

CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "relevent_experience",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "experience",
    "company_size",
    "company_type",
    "last_new_job"
]


# =====================================================
# 3. LOAD CLEANED DATA
# =====================================================

def load_cleaned_data():

    logging.info("Loading cleaned dataset...")

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    logging.info(
        f"Dataset loaded successfully: {df.shape}"
    )

    return df


# =====================================================
# 4. CREATE NUMERICAL PIPELINE
# =====================================================

def create_numerical_pipeline():

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

    return numerical_pipeline


# =====================================================
# 5. CREATE CATEGORICAL PIPELINE
# =====================================================

def create_categorical_pipeline():

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

    return categorical_pipeline


# =====================================================
# 6. CREATE PREPROCESSOR
# =====================================================

def create_preprocessor():

    logging.info("Creating preprocessing transformer...")

    numerical_pipeline = create_numerical_pipeline()

    categorical_pipeline = create_categorical_pipeline()

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

        remainder="drop",

        verbose_feature_names_out=False

    )

    logging.info(
        "Preprocessing transformer created successfully."
    )

    return preprocessor


# =====================================================
# 7. SPLIT FEATURES AND TARGET
# =====================================================

def split_features_target(df):

    logging.info("Separating features and target...")

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN].astype(int)

    return X, y


# =====================================================
# 8. EXECUTION
# =====================================================

if __name__ == "__main__":

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    try:

        df = load_cleaned_data()

        X, y = split_features_target(df)

        preprocessor = create_preprocessor()

        print("\n========== PREPROCESSING CONFIGURATION ==========")

        print("\nNumerical Features:")

        print(NUMERICAL_FEATURES)

        print("\nCategorical Features:")

        print(CATEGORICAL_FEATURES)

        print("\nOriginal Feature Shape:")

        print(X.shape)

        print("\nTarget Shape:")

        print(y.shape)

        print("\nPreprocessor Created Successfully!")

        print("\nIMPORTANT:")

        print(
            "Preprocessor has NOT been fitted yet."
        )

        print(
            "It will be fitted only on training data."
        )

    except Exception:

        logging.exception(
            "Preprocessing configuration failed."
        )

        raise