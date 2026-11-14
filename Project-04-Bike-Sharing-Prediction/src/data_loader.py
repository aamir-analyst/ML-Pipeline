import pandas as pd
import logging
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = BASE_DIR / "data" / "raw"


def load_datasets():

    try:
        # Dataset paths
        day_path = DATA_DIR / "day.csv"
        hour_path = DATA_DIR / "hour.csv"

        # Check files existence
        if not day_path.exists():
            raise FileNotFoundError(f"Daily dataset not found: {day_path}")

        if not hour_path.exists():
            raise FileNotFoundError(f"Hourly dataset not found: {hour_path}")

        # Load datasets
        day_df = pd.read_csv(day_path)
        hour_df = pd.read_csv(hour_path)

        logging.info("Both datasets loaded successfully.")

        return day_df, hour_df

    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        raise