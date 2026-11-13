import logging
from pathlib import Path

import joblib
import pandas as pd

from scipy import sparse

from src.preprocessing.preprocessing_pipeline import (
    create_preprocessor,
    split_features_target
)

from src.preprocessing.data_splitter import (
    split_data
)

from src.preprocessing.preprocessing_pipeline import (
    load_cleaned_data
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

ARTIFACTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 2. LOAD AND SPLIT DATA
# =====================================================

def load_and_split_data():

    logging.info("Loading cleaned dataset...")

    df = load_cleaned_data()

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    return X_train, X_test, y_train, y_test


# =====================================================
# 3. FIT AND TRANSFORM DATA
# =====================================================

def preprocess_data(
    X_train,
    X_test
):

    logging.info("Creating preprocessor...")

    preprocessor = create_preprocessor()

    logging.info("Fitting preprocessor on training data...")

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    logging.info("Transforming testing data...")

    X_test_processed = preprocessor.transform(
        X_test
    )

    logging.info("Preprocessing completed.")

    return (
        preprocessor,
        X_train_processed,
        X_test_processed
    )


# =====================================================
# 4. DISPLAY RESULTS
# =====================================================

def display_results(
    X_train_processed,
    X_test_processed,
    y_train,
    y_test
):

    print("\n========== PREPROCESSING RESULTS ==========")

    print(
        "\nTraining Data Shape:",
        X_train_processed.shape
    )

    print(
        "Testing Data Shape:",
        X_test_processed.shape
    )

    print(
        "Training Target Shape:",
        y_train.shape
    )

    print(
        "Testing Target Shape:",
        y_test.shape
    )

    print(
        "\nTraining Data Format:",
        type(X_train_processed)
    )

    print(
        "Testing Data Format:",
        type(X_test_processed)
    )


# =====================================================
# 5. SAVE PREPROCESSOR
# =====================================================

def save_preprocessor(preprocessor):

    output_path = ARTIFACTS_PATH / "preprocessor.joblib"

    joblib.dump(
        preprocessor,
        output_path
    )

    logging.info(
        f"Preprocessor saved: {output_path}"
    )


# =====================================================
# 6. SAVE PROCESSED ARRAYS
# =====================================================

def save_processed_data(
    X_train_processed,
    X_test_processed,
    y_train,
    y_test
):

    logging.info("Saving transformed datasets...")

    # Save sparse matrices efficiently

    sparse.save_npz(
        PROCESSED_PATH / "X_train_processed.npz",
        sparse.csr_matrix(X_train_processed)
    )

    sparse.save_npz(
        PROCESSED_PATH / "X_test_processed.npz",
        sparse.csr_matrix(X_test_processed)
    )

    y_train.to_csv(
        PROCESSED_PATH / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        PROCESSED_PATH / "y_test.csv",
        index=False
    )

    logging.info("Processed datasets saved successfully.")


# =====================================================
# 7. COMPLETE EXECUTION
# =====================================================

def run_preprocessing():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    try:

        logging.info("Starting Preprocessing Pipeline...")

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

        display_results(
            X_train_processed,
            X_test_processed,
            y_train,
            y_test
        )

        save_preprocessor(preprocessor)

        save_processed_data(
            X_train_processed,
            X_test_processed,
            y_train,
            y_test
        )

        print("\n====================================")

        print("PREPROCESSING COMPLETED SUCCESSFULLY")

        print("====================================")

    except Exception:

        logging.exception(
            "Preprocessing Pipeline Failed."
        )

        raise


# =====================================================
# 8. MAIN
# =====================================================

if __name__ == "__main__":

    run_preprocessing()