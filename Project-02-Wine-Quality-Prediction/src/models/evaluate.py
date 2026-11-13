import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)


def evaluate_models(
    models,
    X_train,
    X_test,
    y_train,
    y_test
):

    results = []

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for name, model in models.items():

        print(f"\nEvaluating {name}...")

        y_pred = model.predict(X_test)

        # ROC-AUC without probability=True for SVM
        if hasattr(model, "predict_proba"):

            y_score = model.predict_proba(X_test)[:, 1]

        else:

            y_score = model.decision_function(X_test)

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )

        results.append({

            "Model": name,

            "Accuracy": accuracy_score(
                y_test, y_pred
            ),

            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),

            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),

            "F1 Score": f1_score(
                y_test, y_pred, zero_division=0
            ),

            "ROC-AUC": roc_auc_score(
                y_test, y_score
            ),

            "CV F1 Mean": cv_scores.mean()

        })

        print(classification_report(
            y_test,
            y_pred,
            zero_division=0
        ))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    )

    return results_df