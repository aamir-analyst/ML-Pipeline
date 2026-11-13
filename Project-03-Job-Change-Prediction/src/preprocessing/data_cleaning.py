import logging
from pathlib import Path

import pandas as pd

from src.data.data_loader import load_data


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
)

PROCESSED_DATA_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 2. REQUIRED COLUMNS
# =====================================================

REQUIRED_COLUMNS = [

    "enrollee_id",
    "city",
    "city_development_index",
    "gender",
    "relevent_experience",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "experience",
    "company_size",
    "company_type",
    "last_new_job",
    "training_hours",
    "target"

]


# =====================================================
# 3. DATA VALIDATION
# =====================================================

def validate_data(df):

    """
    Validate dataset structure and target variable.
    """

    logging.info("Starting Data Validation...")

    # Check required columns

    missing_columns = [

        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns

    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Check target variable

    if df["target"].isnull().any():

        raise ValueError(
            "Target column contains missing values."
        )

    valid_targets = {0, 1}

    actual_targets = set(df["target"].unique())

    if not actual_targets.issubset(valid_targets):

        raise ValueError(
            f"Invalid target values found: {actual_targets}"
        )

    logging.info("Data Validation Completed Successfully.")


# =====================================================
# 4. HANDLE DUPLICATES
# =====================================================

def handle_duplicates(df):

    """
    Remove duplicate records.
    """

    logging.info("Checking Duplicate Records...")

    initial_rows = len(df)

    duplicate_count = df.duplicated().sum()

    print("\n========== DUPLICATE ANALYSIS ==========")

    print(f"Duplicate Records: {duplicate_count}")

    if duplicate_count > 0:

        df = df.drop_duplicates()

        logging.info(
            f"Removed {duplicate_count} duplicate records."
        )

    else:

        logging.info("No duplicate records found.")

    final_rows = len(df)

    print(f"Rows Before Cleaning: {initial_rows}")

    print(f"Rows After Cleaning: {final_rows}")

    return df


# =====================================================
# 5. STRING CLEANING
# =====================================================

def clean_string_columns(df):

    """
    Remove unnecessary spaces from string columns.
    Convert empty strings into missing values.
    """

    logging.info("Cleaning String Columns...")

    for column in df.columns:

        if isinstance(df[column].dtype, pd.StringDtype):

            df[column] = (
                df[column]
                .str.strip()
                .replace("", pd.NA)
            )

        elif df[column].dtype == object:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )

    logging.info("String Cleaning Completed.")

    return df


# =====================================================
# 6. REMOVE UNNECESSARY FEATURES
# =====================================================

def remove_unnecessary_features(df):

    """
    Remove identifier columns.
    """

    logging.info("Removing Unnecessary Features...")

    columns_to_drop = [

        "enrollee_id"

    ]

    existing_columns = [

        column
        for column in columns_to_drop
        if column in df.columns

    ]

    df = df.drop(
        columns=existing_columns
    )

    print("\n========== REMOVED FEATURES ==========")

    print(existing_columns)

    logging.info("Unnecessary Features Removed.")

    return df


# =====================================================
# 7. FINAL DATA SUMMARY
# =====================================================

def final_data_summary(df):

    """
    Display final cleaned dataset information.
    """

    print("\n========== CLEANED DATASET SUMMARY ==========")

    print(f"Total Rows: {df.shape[0]}")

    print(f"Total Columns: {df.shape[1]}")

    print("\nColumn Names:")

    print(df.columns.tolist())

    print("\nData Types:")

    print(df.dtypes)

    print("\nMissing Values:")

    print(df.isnull().sum())

    print("\nTarget Distribution:")

    print(df["target"].value_counts())


# =====================================================
# 8. SAVE CLEANED DATASET
# =====================================================

def save_cleaned_data(df):

    """
    Save cleaned dataset.
    """

    output_path = (
        PROCESSED_DATA_PATH / "cleaned_data.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    logging.info(
        f"Cleaned dataset saved at: {output_path}"
    )

    print("\nCleaned Dataset Saved Successfully!")

    print(output_path)


# =====================================================
# 9. COMPLETE DATA CLEANING PIPELINE
# =====================================================

def run_data_cleaning():

    """
    Execute complete data cleaning pipeline.
    """

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    try:

        logging.info("Starting Data Cleaning Pipeline...")

        # Step 1: Load Dataset

        df = load_data()

        # Step 2: Validate Dataset

        validate_data(df)

        # Step 3: Handle Duplicates

        df = handle_duplicates(df)

        # Step 4: Clean String Columns

        df = clean_string_columns(df)

        # Step 5: Remove Unnecessary Features

        df = remove_unnecessary_features(df)

        # Step 6: Final Summary

        final_data_summary(df)

        # Step 7: Save Cleaned Dataset

        save_cleaned_data(df)

        logging.info(
            "Data Cleaning Pipeline Completed Successfully."
        )

        print("\n====================================")

        print("DATA CLEANING COMPLETED SUCCESSFULLY")

        print("====================================")

    except Exception:

        logging.exception(
            "Data Cleaning Pipeline Failed."
        )

        raise


# =====================================================
# 10. EXECUTION
# =====================================================

if __name__ == "__main__":

    run_data_cleaning()