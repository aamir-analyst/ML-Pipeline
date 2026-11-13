import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# -----------------------------------
# 1. Load Processed Dataset
# -----------------------------------

def load_processed_data():

    project_root = Path(__file__).resolve().parents[2]

    file_path = (
        project_root
        / "data"
        / "processed"
        / "loan_delinquent_processed.csv"
    )

    df = pd.read_csv(file_path)

    print("\nDataset Loaded Successfully!")

    print("Dataset Shape:", df.shape)

    return df


# -----------------------------------
# 2. Separate Features and Target
# -----------------------------------

def split_features_target(df):

    X = df.drop(columns=["isDelinquent"])

    y = df["isDelinquent"]

    print("\nFeatures Shape:", X.shape)

    print("Target Shape:", y.shape)

    return X, y


# -----------------------------------
# 3. Train-Test Split
# -----------------------------------

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    print("\n========== TRAIN TEST SPLIT ==========")

    print("X_train:", X_train.shape)

    print("X_test:", X_test.shape)

    print("\nTraining Distribution:")

    print(y_train.value_counts())

    print("\nTesting Distribution:")

    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


# -----------------------------------
# 4. Train All Models
# -----------------------------------

def train_models(X_train, y_train):

    models = {}

    # Baseline Logistic Regression

    baseline_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    baseline_model.fit(X_train, y_train)

    models["Baseline Logistic Regression"] = baseline_model

    print("\nBaseline Logistic Regression Training Completed!")


    # Balanced Logistic Regression

    balanced_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    balanced_model.fit(X_train, y_train)

    models["Balanced Logistic Regression"] = balanced_model

    print("Balanced Logistic Regression Training Completed!")


    # Strong Regularization

    regularized_model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    regularized_model.fit(X_train, y_train)

    models["Strong Regularization"] = regularized_model

    print("Strong Regularization Training Completed!")


    # Oversampling

    oversampler = RandomOverSampler(
        random_state=42
    )

    X_over, y_over = oversampler.fit_resample(
        X_train,
        y_train
    )

    print("\n========== OVERSAMPLING ==========")

    print(y_over.value_counts())

    oversampling_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    oversampling_model.fit(X_over, y_over)

    models["Oversampling"] = oversampling_model

    print("Oversampling Model Training Completed!")


    # Undersampling

    undersampler = RandomUnderSampler(
        random_state=42
    )

    X_under, y_under = undersampler.fit_resample(
        X_train,
        y_train
    )

    print("\n========== UNDERSAMPLING ==========")

    print(y_under.value_counts())

    undersampling_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    undersampling_model.fit(X_under, y_under)

    models["Undersampling"] = undersampling_model

    print("Undersampling Model Training Completed!")

    return models


# -----------------------------------
# 5. Evaluate All Models
# -----------------------------------

def evaluate_models(models, X_test, y_test):

    results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        precision = report["1"]["precision"]

        recall = report["1"]["recall"]

        f1 = report["1"]["f1-score"]


        # Store Results

        results.append({

            "Model": name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1,

            "ROC-AUC": roc_auc

        })


        # Print Individual Results

        print("\n========================================")

        print(name)

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        print("\nConfusion Matrix:")

        print(
            confusion_matrix(
                y_test,
                y_pred
            )
        )

        print("\nAccuracy:", round(accuracy, 4))

        print("ROC-AUC:", round(roc_auc, 4))


    # -----------------------------------
    # Model Comparison
    # -----------------------------------

    results_df = pd.DataFrame(results)

    print("\n========== FINAL MODEL COMPARISON ==========")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return results_df


# -----------------------------------
# 6. Save Model Comparison
# -----------------------------------

def save_results(results_df):

    project_root = Path(__file__).resolve().parents[2]

    reports_dir = project_root / "reports"

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = reports_dir / "model_comparison.csv"

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nModel Comparison Saved Successfully!")

    print("File Location:", output_path)


# -----------------------------------
# 7. Main Execution
# -----------------------------------

if __name__ == "__main__":

    # Load Data

    df = load_processed_data()

    # Separate Features and Target

    X, y = split_features_target(df)

    # Train Test Split

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train All Models

    models = train_models(
        X_train,
        y_train
    )

    # Evaluate Models

    results_df = evaluate_models(
        models,
        X_test,
        y_test
    )

    # Save Comparison

    save_results(results_df)

    print("\n========================================")

    print("ALL MODEL EXPERIMENTS COMPLETED!")

    print("========================================")