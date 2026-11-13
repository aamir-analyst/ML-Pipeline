import pandas as pd
from pathlib import Path


def load_data():

    project_root = Path(__file__).resolve().parents[2]

    data_path = project_root / "data" / "raw" / "winequality.csv"

    df = pd.read_csv(data_path, sep=";")

    print("Dataset Loaded Successfully!")
    print("Original Shape:", df.shape)

    # Remove duplicate records
    df = df.drop_duplicates()

    print("After Removing Duplicates:", df.shape)

    return df