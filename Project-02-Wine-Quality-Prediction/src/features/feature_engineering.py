import pandas as pd


def create_features(df):

    df = df.copy()

    # Create Binary Target
    df["is_premium"] = (
        df["quality"] >= 7
    ).astype(int)

    # Separate Features and Target
    X = df.drop(
        columns=["quality", "is_premium"]
    )

    y = df["is_premium"]

    print("\nFeature Engineering Completed!")

    print("Features Shape:", X.shape)
    print("Target Shape:", y.shape)

    print("\nTarget Distribution:")
    print(y.value_counts())

    return X, y