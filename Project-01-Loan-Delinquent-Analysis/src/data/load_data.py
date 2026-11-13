import pandas as pd
from pathlib import Path


def load_dataset():

    project_root = Path(__file__).resolve().parents[2]

    file_path = project_root / "data" / "raw" / "Loan_Delinquent_Dataset (1).csv"

    df = pd.read_csv(file_path)

    return df


if __name__ == "__main__":

    df = load_dataset()

    print("Dataset Loaded Successfully!")

    print("Shape:", df.shape)

    print(df.head())