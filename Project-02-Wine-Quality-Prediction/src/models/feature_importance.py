from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


def analyze_feature_importance(
    model,
    X_test,
    y_test,
    feature_names
):
    """
    Calculate model-agnostic permutation feature importance.

    A feature is considered important when shuffling it
    reduces the model's F1 score.
    """

    project_root = Path(__file__).resolve().parents[2]

    metrics_dir = project_root / "reports" / "metrics"
    figures_dir = project_root / "reports" / "figures"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== PERMUTATION FEATURE IMPORTANCE ==========")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=30,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance Mean": result.importances_mean,
        "Importance Std": result.importances_std
    })

    importance_df = importance_df.sort_values(
        by="Importance Mean",
        ascending=False
    ).reset_index(drop=True)

    print("\nFeature Importance Ranking:\n")
    print(importance_df.to_string(index=False))

    # Save feature importance report
    csv_path = metrics_dir / "permutation_feature_importance.csv"

    importance_df.to_csv(csv_path, index=False)

    print(f"\nFeature Importance CSV Saved: {csv_path}")

    # Generate feature importance graph
    plt.figure(figsize=(10, 7))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance Mean"],
        xerr=importance_df["Importance Std"]
    )

    plt.xlabel("Decrease in F1 Score")
    plt.ylabel("Features")
    plt.title("Permutation Feature Importance - SVM")
    plt.gca().invert_yaxis()

    plt.tight_layout()

    graph_path = figures_dir / "08_permutation_feature_importance.png"

    plt.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Feature Importance Graph Saved: {graph_path}")

    print("\n========== FEATURE IMPORTANCE COMPLETED ==========")

    return importance_df