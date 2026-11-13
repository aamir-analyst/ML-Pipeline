import pandas as pd

from sklearn.model_selection import train_test_split


# =====================================================
# 1. MISSING VALUE ANALYSIS
# =====================================================

def check_missing_values(df):

    print("\n========== MISSING VALUE ANALYSIS ==========")

    missing_values = df.isnull().sum()

    print(missing_values)

    total_missing = missing_values.sum()

    print("\nTotal Missing Values:", total_missing)

    return missing_values


# =====================================================
# 2. MISSING VALUE TREATMENT
# =====================================================

def handle_missing_values(df):

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        if df[column].isnull().any():

            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )

            print(
                f"{column}: Missing values filled using median."
            )

    print("\nMissing Value Treatment Completed!")

    return df


# =====================================================
# 3. OUTLIER DETECTION USING IQR
# =====================================================

def detect_outliers(df):

    print("\n========== OUTLIER DETECTION ==========")

    # Target column ko exclude karna
    feature_df = df.drop(
        columns=["quality"],
        errors="ignore"
    )

    numeric_columns = feature_df.select_dtypes(
        include="number"
    ).columns

    outlier_summary = {}

    for column in numeric_columns:

        Q1 = df[column].quantile(0.25)

        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_limit = Q1 - (1.5 * IQR)

        upper_limit = Q3 + (1.5 * IQR)

        outlier_count = (
            (df[column] < lower_limit) |
            (df[column] > upper_limit)
        ).sum()

        outlier_summary[column] = int(outlier_count)

        print(
            f"{column}: {outlier_count} outliers"
        )

    return outlier_summary


# =====================================================
# 4. TRAIN TEST SPLIT
# =====================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    print("\n========== TRAIN TEST SPLIT ==========")

    print("Training Data:", X_train.shape)

    print("Testing Data:", X_test.shape)

    return X_train, X_test, y_train, y_test