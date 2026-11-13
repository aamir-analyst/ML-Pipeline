from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def get_models():

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                class_weight="balanced",
                random_state=42
            ))
        ])
    }

    return models


def train_models(X_train, y_train):

    models = get_models()
    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

        print(f"{name} Training Completed!")

    return trained_models


def save_model(model, model_name):

    project_root = Path(__file__).resolve().parents[2]

    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / model_name

    joblib.dump(model, model_path)

    print(f"\nModel Saved Successfully: {model_path}")