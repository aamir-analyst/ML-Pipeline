import logging
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from src.preprocessing.preprocessing_pipeline import (
    load_cleaned_data,
    split_features_target
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_SIZE = 0.20

RANDOM_STATE = 42

SPLIT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "splits"

SPLIT_DATA_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 2. TRAIN TEST SPLIT
# =====================================================

def split_data(X, y):

    logging.info("Starting Train-Test Split...")

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    logging.info("Train-Test Split Completed.")

    return X_train, X_test, y_train, y_test


# =====================================================
# 3. DISPLAY SPLIT INFORMATION
# =====================================================

def display_split_information(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n========== DATA SPLIT SUMMARY ==========")

    print("\nTraining Features:", X_train.shape)

    print("Testing Features:", X_test.shape)

    print("Training Target:", y_train.shape)

    print("Testing Target:", y_test.shape)

    print("\nTraining Target Distribution:")

    print(y_train.value_counts(normalize=True) * 100)

    print("\nTesting Target Distribution:")

    print(y_test.value_counts(normalize=True) * 100)


# =====================================================
# 4. SAVE SPLIT DATA
# =====================================================

def save_split_data(
    X_train,
    X_test,
    y_train,
    y_test
):

    logging.info("Saving split datasets...")

    X_train.to_csv(
        SPLIT_DATA_PATH / "X_train.csv",
        index=False
    )

    X_test.to_csv(
        SPLIT_DATA_PATH / "X_test.csv",
        index=False
    )

    y_train.to_csv(
        SPLIT_DATA_PATH / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        SPLIT_DATA_PATH / "y_test.csv",
        index=False
    )

    logging.info("Split datasets saved successfully.")


# =====================================================
# 5. EXECUTION
# =====================================================

def run_data_split():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    try:

        df = load_cleaned_data()

        X, y = split_features_target(df)

        X_train, X_test, y_train, y_test = split_data(
            X,
            y
        )

        display_split_information(
            X_train,
            X_test,
            y_train,
            y_test
        )

        save_split_data(
            X_train,
            X_test,
            y_train,
            y_test
        )

        print("\nTrain-Test Split Completed Successfully!")

    except Exception:

        logging.exception("Train-Test Split Failed.")

        raise


if __name__ == "__main__":

    run_data_split()