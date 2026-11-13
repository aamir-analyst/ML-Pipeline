import joblib
import pandas as pd

from pathlib import Path


# -----------------------------------
# 1. Load Saved Model
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "loan_delinquency_model.pkl"

METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def load_model():

    model = joblib.load(MODEL_PATH)

    print("\nML Model Loaded Successfully!")

    return model


# -----------------------------------
# 2. Prepare Customer Input
# -----------------------------------

def prepare_input():

    print("\n========== CUSTOMER DETAILS ==========")

    term = input("Loan Term (36 months / 60 months): ").strip()

    gender = input("Gender (Male / Female): ").strip().title()

    purpose = input(
        "Purpose (Car / House / Medical / Other / Personal / Wedding): "
    ).strip().title()

    home = input(
        "Home Ownership (Mortgage / Rent / Own): "
    ).strip().title()

    age = input("Age Group (20-25 / >25): ").strip()

    fico = input("FICO Group (300-500 / >500): ").strip()

    # Validate input

    if term not in ["36 months", "60 months"]:
        raise ValueError("Invalid loan term!")

    if gender not in ["Male", "Female"]:
        raise ValueError("Invalid gender!")

    if purpose not in [
        "Car", "House", "Medical", "Other", "Personal", "Wedding"
    ]:
        raise ValueError("Invalid purpose!")

    if home not in ["Mortgage", "Rent", "Own"]:
        raise ValueError("Invalid home ownership!")

    if age not in ["20-25", ">25"]:
        raise ValueError("Invalid age group!")

    if fico not in ["300-500", ">500"]:
        raise ValueError("Invalid FICO group!")

    # Manually encode features to match training columns

    customer = {
        "term_60 months": int(term == "60 months"),
        "gender_Male": int(gender == "Male"),
        "purpose_House": int(purpose == "House"),
        "purpose_Medical": int(purpose == "Medical"),
        "purpose_Other": int(purpose == "Other"),
        "purpose_Personal": int(purpose == "Personal"),
        "purpose_Wedding": int(purpose == "Wedding"),
        "home_ownership_Own": int(home == "Own"),
        "home_ownership_Rent": int(home == "Rent"),
        "age_>25": int(age == ">25"),
        "FICO_>500": int(fico == ">500")
    }

    return pd.DataFrame([customer])


# -----------------------------------
# 3. Make Prediction
# -----------------------------------

def predict_customer(model, customer):

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0]

    delinquent_probability = probability[1] * 100

    non_delinquent_probability = probability[0] * 100

    print("\n========== PREDICTION RESULT ==========")

    print("Non-Delinquent Probability:",
          round(non_delinquent_probability, 2), "%")

    print("Delinquent Probability:",
          round(delinquent_probability, 2), "%")

    if prediction == 1:

        print("\nResult: LOAN DELINQUENT")

    else:

        print("\nResult: LOAN NON-DELINQUENT")


# -----------------------------------
# 4. Main Execution
# -----------------------------------

if __name__ == "__main__":

    model = load_model()

    customer = prepare_input()

    predict_customer(model, customer)