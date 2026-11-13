import logging
from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier

from src.preprocessing.prepare_data import load_and_split_data, preprocess_data


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "models"

MODEL_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 2. DEFINE MODELS
# =====================================================

def get_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )

    }

    return models


# =====================================================
# 3. TRAIN MODELS
# =====================================================

def train_models(X_train, y_train):

    logging.info("Starting Model Training...")

    models = get_models()

    trained_models = {}

    for model_name, model in models.items():

        logging.info(f"Training {model_name}...")

        model.fit(
            X_train,
            y_train
        )

        trained_models[model_name] = model

        logging.info(
            f"{model_name} trained successfully."
        )

    return trained_models


# =====================================================
# 4. SAVE MODELS
# =====================================================

def save_models(trained_models):

    logging.info("Saving trained models...")

    for model_name, model in trained_models.items():

        file_name = (
            model_name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        output_path = MODEL_PATH / file_name

        joblib.dump(
            model,
            output_path
        )

        logging.info(
            f"Saved {model_name}: {output_path}"
        )


# =====================================================
# 5. COMPLETE TRAINING PIPELINE
# =====================================================

def run_training():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    try:

        logging.info("Loading training data...")

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = load_and_split_data()

        logging.info("Preparing training features...")

        (
            preprocessor,
            X_train_processed,
            X_test_processed
        ) = preprocess_data(
            X_train,
            X_test
        )

        # Save the matching preprocessor

        joblib.dump(
            preprocessor,
            PROJECT_ROOT / "artifacts" / "preprocessor.joblib"
        )

        trained_models = train_models(
            X_train_processed,
            y_train
        )

        save_models(trained_models)

        print("\n====================================")

        print("MODEL TRAINING COMPLETED")

        print("====================================")

        print("\nTotal Models Trained:", len(trained_models))

        print("\nTrained Models:")

        for model_name in trained_models:

            print("-", model_name)

    except Exception:

        logging.exception("Model Training Failed.")

        raise


# =====================================================
# 6. EXECUTION
# =====================================================

if __name__ == "__main__":

    run_training()