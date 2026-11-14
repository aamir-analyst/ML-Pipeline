import logging
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"


def evaluate_models(trained_models, X_test, y_test):

    try:
        results = []

        for name, model in trained_models.items():

            print(f"\nEvaluating {name} Model...")

            # Predictions
            y_pred = model.predict(X_test)

            # Evaluation Metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            results.append({
                "Model": name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2 Score": r2
            })

            print(f"{name} Evaluation Completed.")

        # Convert results into DataFrame
        results_df = pd.DataFrame(results)

        # Sort models by R2 Score
        results_df = results_df.sort_values(
            by="R2 Score",
            ascending=False
        )

        # Save evaluation results
        save_path = REPORT_DIR / "model_evaluation_results.csv"

        results_df.to_csv(save_path, index=False)

        print("\n========== MODEL EVALUATION RESULTS ==========")
        print(results_df.to_string(index=False))

        print(f"\nEvaluation report saved: {save_path}")

        logging.info("Model evaluation completed successfully.")

        return results_df

    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        raise

def analyze_feature_importance(best_model, feature_names):

    try:
        import matplotlib.pyplot as plt

        # Bagging ke individual decision trees ki importance
        importances = np.mean(
            [
                estimator.feature_importances_
                for estimator in best_model.estimators_
            ],
            axis=0
        )

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        # Save CSV report
        save_path = REPORT_DIR / "feature_importance.csv"
        importance_df.to_csv(save_path, index=False)

        # Save visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )

        ax.set_title("Feature Importance - Bagging Model")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        ax.invert_yaxis()

        plt.tight_layout()

        graph_path = REPORT_DIR / "figures" / "feature_importance.png"

        fig.savefig(graph_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("\n========== FEATURE IMPORTANCE ==========")
        print(importance_df.to_string(index=False))

        print(f"\nFeature importance report saved: {save_path}")
        print(f"Feature importance graph saved: {graph_path}")

        return importance_df

    except Exception as e:
        logging.error(f"Feature importance analysis failed: {e}")
        raise