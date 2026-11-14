import pandas as pd
from sklearn.model_selection import train_test_split

def inspect_data_quality(day_df, hour_df):

    datasets = {
        "Daily Dataset": day_df,
        "Hourly Dataset": hour_df
    }

    for name, df in datasets.items():

        print(f"\n========== {name} ==========")

        # Missing Values
        print("\nMissing Values:")
        print(df.isnull().sum())

        # Duplicate Records
        print("\nDuplicate Records:")
        print(df.duplicated().sum())

        # Data Types
        print("\nData Types:")
        print(df.dtypes)

        # Target Variable
        print("\nTarget Variable (cnt):")
        print(df["cnt"].describe())

def validate_target_variable(day_df, hour_df):

    datasets = {
        "Daily Dataset": day_df,
        "Hourly Dataset": hour_df
    }

    for name, df in datasets.items():

        calculated_cnt = df["casual"] + df["registered"]

        mismatch_count = (calculated_cnt != df["cnt"]).sum()

        print(f"\n========== {name} ==========")

        print("Total Records:", len(df))

        print("Target Mismatches:", mismatch_count)

        if mismatch_count == 0:
            print("Target Validation: PASSED")
        else:
            print("Target Validation: FAILED")

def prepare_hourly_data(hour_df):

    # Original dataset ko safe rakhna
    df = hour_df.copy()

    # Date column conversion
    df["dteday"] = pd.to_datetime(df["dteday"])

    # Unnecessary and target leakage columns
    columns_to_drop = [
        "instant",
        "dteday",
        "casual",
        "registered"
    ]

    df = df.drop(columns=columns_to_drop)

    # Separate features and target
    X = df.drop(columns=["cnt"])
    y = df["cnt"]

    print("\n========== DATA PREPARATION ==========")
    print("Features Shape:", X.shape)
    print("Target Shape:", y.shape)

    print("\nRemaining Feature Columns:")
    print(X.columns.tolist())

    print("\nTarget Column: cnt")

    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        shuffle=False
    )

    print("\n========== TRAIN TEST SPLIT ==========")
    print("Training Features:", X_train.shape)
    print("Testing Features:", X_test.shape)
    print("Training Target:", y_train.shape)
    print("Testing Target:", y_test.shape)

    return X_train, X_test, y_train, y_test