import logging
import joblib

from pathlib import Path

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)

from sklearn.linear_model import LinearRegression


# ==========================================
# Project Directories
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = BASE_DIR / "artifacts"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Model Training
# ==========================================

def train_models(X_train, y_train):

    try:

        models = {

            # 1. Bagging Model
            "Bagging": BaggingRegressor(
                estimator=DecisionTreeRegressor(
                    random_state=42
                ),
                n_estimators=50,
                random_state=42,
                n_jobs=-1
            ),

            # 2. Boosting Model
            "Boosting": GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ),

            # 3. Stacking Model
            "Stacking": StackingRegressor(
                estimators=[
                    (
                        "bagging",
                        BaggingRegressor(
                            estimator=DecisionTreeRegressor(
                                random_state=42
                            ),
                            n_estimators=30,
                            random_state=42,
                            n_jobs=-1
                        )
                    ),

                    (
                        "boosting",
                        GradientBoostingRegressor(
                            n_estimators=50,
                            random_state=42
                        )
                    )
                ],

                final_estimator=LinearRegression(),
                cv=5,
                n_jobs=-1
            )
        }

        trained_models = {}

        # Train all models
        for name, model in models.items():

            print(f"\nTraining {name} Model...")

            model.fit(X_train, y_train)

            trained_models[name] = model

            print(f"{name} Model Training Completed.")

        # ==========================================
        # Save Best Model
        # ==========================================

        # Based on previous evaluation results,
        # Bagging achieved the highest R2 score.

        best_model = trained_models["Bagging"]

        model_path = ARTIFACT_DIR / "best_model.joblib"

        joblib.dump(best_model, model_path)

        print("\n========== BEST MODEL ==========")
        print("Selected Model: Bagging")
        print(f"Model saved successfully: {model_path}")

        logging.info("All ensemble models trained successfully.")
        logging.info("Best Bagging model saved successfully.")

        return trained_models

    except Exception as e:

        logging.error(f"Model training failed: {e}")

        raise