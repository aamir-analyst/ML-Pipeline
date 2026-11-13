import pandas as pd
from pathlib import Path

from src.data.load_data import load_dataset


# -----------------------------------
# 1. Check Missing Values & Duplicates
# -----------------------------------

def check_missing_values(df):

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE ROWS ==========")
    print(df.duplicated().sum())


# -----------------------------------
# 2. Data Cleaning
# -----------------------------------

def clean_data(df):

    data = df.copy()

    # Remove ID column
    data.drop(columns=["ID"], inplace=True)

    # Standardize purpose categories
    data["purpose"] = (
        data["purpose"]
        .str.strip()
        .str.title()
    )

    print("\nData Cleaning Completed!")

    print("\nCleaned Dataset:")
    print(data.head())

    print("\nCleaned Shape:", data.shape)

    return data


# -----------------------------------
# 3. Categorical Encoding
# -----------------------------------

def encode_data(df):

    data = df.copy()

    # One-Hot Encoding
    data = pd.get_dummies(
        data,
        columns=[
            "term",
            "gender",
            "purpose",
            "home_ownership",
            "age",
            "FICO"
        ],
        drop_first=True,
        dtype=int
    )

    print("\n========== ENCODING COMPLETED ==========")

    print("\nEncoded Columns:")
    print(data.columns.tolist())

    print("\nEncoded Dataset:")
    print(data.head())

    print("\nEncoded Shape:", data.shape)

    return data


# -----------------------------------
# 4. Save Processed Dataset
# -----------------------------------

def save_processed_data(df):

    # Project root directory
    project_root = Path(__file__).resolve().parents[2]

    # Processed data directory
    output_dir = project_root / "data" / "processed"

    # Create directory if not exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    output_path = output_dir / "loan_delinquent_processed.csv"

    df.to_csv(output_path, index=False)

    print("\nProcessed Dataset Saved Successfully!")
    print("File Location:", output_path)


# -----------------------------------
# 5. Main Execution
# -----------------------------------

if __name__ == "__main__":

    # Load Dataset
    df = load_dataset()

    print("\n========== ORIGINAL DATASET ==========")
    print("Shape:", df.shape)

    # Check Missing Values
    check_missing_values(df)

    # Clean Data
    cleaned_df = clean_data(df)

    # Encode Data
    encoded_df = encode_data(cleaned_df)

    # Save Processed Data
    save_processed_data(encoded_df)

    print("\n========== PREPROCESSING COMPLETED ==========")