import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import make_scorer, recall_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

def load_data():

    project_root = Path(__file__).resolve().parents[2]

    file_path = (
        project_root
        / "data"
        / "processed"
        / "loan_delinquent_processed.csv"
    )

    return pd.read_csv(file_path)


# -----------------------------------
# 2. Prepare Features
# -----------------------------------

def prepare_data(df):

    X = df.drop(columns=["isDelinquent"])

    y = df["isDelinquent"]

    return X, y


# -----------------------------------
# 3. K-Fold Cross Validation
# -----------------------------------

def perform_cross_validation(X, y):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = {

        "Baseline Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ),

        "Balanced Logistic Regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),

        "Oversampling Logistic Regression": Pipeline([
            ("oversampling", RandomOverSampler(random_state=42)),
            ("model", LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            ))
        ])

    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": make_scorer(
            recall_score,
            pos_label=1
        ),
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results = []

    for name, model in models.items():

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results.append({

            "Model": name,

            "Accuracy": scores["test_accuracy"].mean(),

            "Precision": scores["test_precision"].mean(),

            "Recall": scores["test_recall"].mean(),

            "F1 Score": scores["test_f1"].mean(),

            "ROC-AUC": scores["test_roc_auc"].mean()

        })

        print("\n===================================")

        print(name)

        print("Accuracy:", scores["test_accuracy"].mean())

        print("Precision:", scores["test_precision"].mean())

        print("Recall:", scores["test_recall"].mean())

        print("F1 Score:", scores["test_f1"].mean())

        print("ROC-AUC:", scores["test_roc_auc"].mean())


    results_df = pd.DataFrame(results)

    print("\n========== CROSS VALIDATION COMPARISON ==========")

    print(results_df.to_string(index=False))

    return results_df


# -----------------------------------
# 4. Main
# -----------------------------------

if __name__ == "__main__":

    df = load_data()

    X, y = prepare_data(df)

    # Hold out test set for final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Cross Validation Dataset:", X_train.shape)

    results = perform_cross_validation(
        X_train,
        y_train
    )

    project_root = Path(__file__).resolve().parents[2]

    reports_dir = project_root / "reports"

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        reports_dir / "cross_validation_results.csv",
        index=False
    )

    print("\nCross Validation Completed Successfully!")