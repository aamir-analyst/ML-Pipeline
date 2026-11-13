import pandas as pd
import logging
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "jobs_data.csv"


def load_data():

    """
    Load Job Change Prediction dataset.
    """

    try:

        logging.info("Starting dataset loading...")

        # Check dataset existence
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {DATA_PATH}"
            )

        # Load CSV
        df = pd.read_csv(DATA_PATH)

        logging.info("Dataset loaded successfully.")

        logging.info(f"Dataset Shape: {df.shape}")

        return df

    except FileNotFoundError as error:

        logging.error(f"File Error: {error}")
        raise

    except pd.errors.ParserError as error:

        logging.error(f"CSV Parsing Error: {error}")
        raise

    except Exception as error:

        logging.exception(f"Unexpected Error: {error}")
        raise


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    df = load_data()

    print("\nDataset Loaded Successfully!")

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())