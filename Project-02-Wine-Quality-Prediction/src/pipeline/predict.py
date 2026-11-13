
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_wine_quality_model.joblib"
)


# Features in training dataset order
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("\nModel Loaded Successfully!")

    return model


def get_wine_input():

    print("\n========== ENTER WINE DETAILS ==========")

    wine_data = {}

    for feature in FEATURES:

        while True:

            try:
                value = input(f"Enter {feature}: ").strip()

                if not value:
                    print("Input cannot be empty. Try again.")
                    continue

                value = float(value)

                if not np.isfinite(value):
                    print("Please enter a valid finite number.")
                    continue

                wine_data[feature] = value

                break

            except ValueError:
                print("Invalid input! Please enter a numeric value.")

    return pd.DataFrame([wine_data], columns=FEATURES)


def predict_wine(model, wine_data):

    prediction = model.predict(wine_data)[0]

    if prediction == 1:
        result = "Premium Wine"
    else:
        result = "Non-Premium Wine"

    print("\n========== PREDICTION RESULT ==========")

    print(f"\nWine Category: {result}")

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(wine_data)[0]

        classes = list(model.classes_)

        premium_index = classes.index(1)

        premium_probability = probabilities[premium_index]

        predicted_index = classes.index(prediction)

        predicted_probability = probabilities[predicted_index]

        print(
            f"\nPremium Probability: "
            f"{premium_probability * 100:.2f}%"
        )

        print(
            f"Predicted Class Probability: "
            f"{predicted_probability * 100:.2f}%"
        )

    else:
        print(
            "\nProbability score is not available "
            "for this model."
        )

    print("\n=======================================")

    return result


def main():

    print("\n========== WINE QUALITY PREDICTION ==========")

    try:

        # Load saved model
        model = load_model()

        # Collect user input
        wine_data = get_wine_input()

        # Predict wine quality category
        predict_wine(model, wine_data)

    except FileNotFoundError as error:

        print(f"\nERROR: {error}")

    except KeyboardInterrupt:

        print("\n\nPrediction cancelled by user.")


if __name__ == "__main__":
    main()