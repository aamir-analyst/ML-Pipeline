from pathlib import Path
import pandas as pd

from src.data.data_loader import load_data
from src.eda.eda import generate_eda_report

from src.preprocessing.preprocess import (
    check_missing_values,
    handle_missing_values,
    detect_outliers,
    split_data
)

from src.features.feature_engineering import create_features

from src.models.train import train_models, save_model
from src.models.evaluate import evaluate_models
from src.models.tune import tune_models
from src.models.feature_importance import analyze_feature_importance

from src.utils.logger import setup_logger


logger = setup_logger()


def main():

    project_root = Path(__file__).resolve().parent

    metrics_dir = project_root / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("WINE QUALITY PIPELINE STARTED")
    logger.info("=" * 50)

    try:

        # STEP 1: DATA LOADING

        print("\n[STEP 1] DATA LOADING")

        df = load_data()

        logger.info("Dataset loaded successfully: %s", df.shape)

        # STEP 2: EXPLORATORY DATA ANALYSIS

        print("\n[STEP 2] EXPLORATORY DATA ANALYSIS")

        generate_eda_report(df)

        logger.info("EDA completed successfully")

        # STEP 3: MISSING VALUE ANALYSIS

        print("\n[STEP 3] MISSING VALUE ANALYSIS")

        check_missing_values(df)

        # STEP 4: MISSING VALUE TREATMENT

        print("\n[STEP 4] MISSING VALUE TREATMENT")

        df = handle_missing_values(df)

        # STEP 5: OUTLIER DETECTION

        print("\n[STEP 5] OUTLIER DETECTION")

        outlier_summary = detect_outliers(df)

        outlier_df = pd.DataFrame(
            list(outlier_summary.items()),
            columns=["Feature", "Outlier Count"]
        )

        outlier_df.to_csv(
            metrics_dir / "outlier_report.csv",
            index=False
        )

        print("\nOutlier Report Saved!")

        # STEP 6: FEATURE ENGINEERING

        print("\n[STEP 6] FEATURE ENGINEERING")

        X, y = create_features(df)

        # STEP 7: TRAIN TEST SPLIT

        print("\n[STEP 7] TRAIN TEST SPLIT")

        X_train, X_test, y_train, y_test = split_data(X, y)

        # STEP 8: MODEL TRAINING

        print("\n[STEP 8] MODEL TRAINING")

        trained_models = train_models(
            X_train,
            y_train
        )

        # STEP 9: MODEL EVALUATION

        print("\n[STEP 9] MODEL EVALUATION")

        comparison_df = evaluate_models(
            trained_models,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print("\n========== MODEL COMPARISON ==========\n")

        print(comparison_df.to_string(index=False))

        comparison_df.to_csv(
            metrics_dir / "model_comparison.csv",
            index=False
        )

        # STEP 10: HYPERPARAMETER TUNING

        print("\n[STEP 10] HYPERPARAMETER TUNING")

        tuned_models = tune_models(
            X_train,
            y_train
        )

        # STEP 11: TUNED MODEL EVALUATION

        print("\n[STEP 11] TUNED MODEL EVALUATION")

        tuned_results = evaluate_models(
            tuned_models,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print("\n========== TUNED MODEL COMPARISON ==========\n")

        print(tuned_results.to_string(index=False))

        tuned_results.to_csv(
            metrics_dir / "tuned_model_comparison.csv",
            index=False
        )

        # STEP 12: BEST MODEL SELECTION

        print("\n[STEP 12] BEST MODEL SELECTION")

        best_index = tuned_results["CV F1 Mean"].idxmax()

        best_model_name = tuned_results.loc[
            best_index, "Model"
        ]

        best_model = tuned_models[best_model_name]

        best_cv_score = tuned_results.loc[
            best_index, "CV F1 Mean"
        ]

        print("\nModel Selection Criterion: CV F1 Score")

        print(f"Selected Model: {best_model_name}")

        print(f"Cross Validation F1: {best_cv_score:.4f}")

        logger.info(
            "Selected model: %s | CV F1: %.4f",
            best_model_name,
            best_cv_score
        )

        # STEP 13: FINAL MODEL SAVING

        print("\n[STEP 13] MODEL SAVING")

        save_model(
            best_model,
            "best_wine_quality_model.joblib"
        )

        # STEP 14: PERMUTATION FEATURE IMPORTANCE

        print("\n[STEP 14] PERMUTATION FEATURE IMPORTANCE")

        feature_importance_df = analyze_feature_importance(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            feature_names=X_test.columns
        )

        logger.info(
            "Permutation feature importance completed successfully"
        )

        # STEP 15: FINAL SUMMARY

        print("\n==========================================")
        print("WINE QUALITY PIPELINE COMPLETED")
        print("==========================================")

        print("\nGenerated Reports:")

        print("- EDA Summary")
        print("- Outlier Report")
        print("- Model Comparison")
        print("- Tuned Model Comparison")
        print("- Permutation Feature Importance")

        print(f"\nFinal Selected Model: {best_model_name}")

        print(f"Cross Validation F1: {best_cv_score:.4f}")

        print("\nTop 5 Important Features:")

        print(
            feature_importance_df.head(5).to_string(index=False)
        )

        print("\n========== PROJECT COMPLETED ==========")

        logger.info(
            "WINE QUALITY PIPELINE COMPLETED SUCCESSFULLY"
        )

    except Exception:

        logger.exception("WINE QUALITY PIPELINE FAILED")

        raise


if __name__ == "__main__":
    main()