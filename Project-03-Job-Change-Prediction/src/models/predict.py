import logging
from pathlib import Path

import joblib
import pandas as pd


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_tuned_model.joblib"


# =====================================================
# 2. EXPECTED FEATURES
# =====================================================

FEATURE_COLUMNS = [

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
    "training_hours"

]


# =====================================================
# 3. LOAD MODEL
# =====================================================

def load_model():

    logging.info("Loading prediction model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    logging.info("Prediction model loaded successfully.")

    return model


# =====================================================
# 4. VALIDATE INPUT
# =====================================================

def validate_input(candidate_data):

    if not isinstance(candidate_data, dict):

        raise TypeError(
            "Candidate input must be a dictionary."
        )

    missing_features = [

        feature
        for feature in FEATURE_COLUMNS
        if feature not in candidate_data

    ]

    extra_features = [

        feature
        for feature in candidate_data
        if feature not in FEATURE_COLUMNS

    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
        )

    if extra_features:

        raise ValueError(
            f"Unexpected features: {extra_features}"
        )

    if not isinstance(candidate_data["city"], str):

        raise ValueError("City must be a string.")

    if not isinstance(candidate_data["training_hours"], (int, float)):

        raise ValueError("Training hours must be numeric.")

    if not isinstance(candidate_data["city_development_index"], (int, float)):

        raise ValueError("City development index must be numeric.")

    if candidate_data["training_hours"] < 0:

        raise ValueError("Training hours cannot be negative.")

    if not 0 <= candidate_data["city_development_index"] <= 1:

        raise ValueError(
            "City development index must be between 0 and 1."
        )


# =====================================================
# 5. PREDICT CANDIDATE
# =====================================================

def predict_candidate(candidate_data):

    logging.info("Starting candidate prediction...")

    validate_input(candidate_data)

    model = load_model()

    input_df = pd.DataFrame(

        [candidate_data],

        columns=FEATURE_COLUMNS

    )

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    job_change_probability = probability[1]

    if prediction == 1:

        result = "Looking for Job Change"

    else:

        result = "Not Looking for Job Change"

    output = {

        "prediction": int(prediction),

        "result": result,

        "job_change_probability": round(
            float(job_change_probability),
            4
        )

    }

    return output


# =====================================================
# 6. EXECUTION
# =====================================================

if __name__ == "__main__":

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    sample_candidate = {

        "city": "city_103",

        "city_development_index": 0.92,

        "gender": "Male",

        "relevent_experience": "Has relevent experience",

        "enrolled_university": "no_enrollment",

        "education_level": "Graduate",

        "major_discipline": "STEM",

        "experience": "5",

        "company_size": "50-99",

        "company_type": "Pvt Ltd",

        "last_new_job": "1",

        "training_hours": 50

    }

    try:

        result = predict_candidate(
            sample_candidate
        )

        print("\n========== CANDIDATE PREDICTION ==========")

        print("Prediction:", result["prediction"])

        print("Result:", result["result"])

        print(
            "Job Change Probability:",
            result["job_change_probability"]
        )

    except Exception:

        logging.exception(
            "Prediction failed."
        )

        raise