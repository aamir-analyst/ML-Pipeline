# ==========================================
# IMPORTS
# ==========================================

# Data Loading
from src.data_loader import load_datasets

# Data Preprocessing
from src.data_preprocessing import (
    inspect_data_quality,
    validate_target_variable,
    prepare_hourly_data,
    split_data
)

# Data Visualization
from src.visualization import (
    plot_target_distribution,
    plot_season_vs_rentals,
    plot_weather_vs_rentals,
    plot_workingday_vs_rentals,
    plot_weekday_vs_rentals,
    plot_weather_factors,
    plot_hourly_demand,
    plot_holiday_vs_rentals
)

# Model Training
from src.model_training import train_models

# Model Evaluation
from src.model_evaluation import (
    evaluate_models,
    analyze_feature_importance
)


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("\n========================================")
    print("       BIKE SHARING PREDICTION")
    print("========================================")


    # ======================================
    # 1. DATA LOADING
    # ======================================

    print("\n========== DATA LOADING ==========")

    day_df, hour_df = load_datasets()

    print("Daily Dataset Shape:", day_df.shape)
    print("Hourly Dataset Shape:", hour_df.shape)


    # ======================================
    # 2. DATASET OVERVIEW
    # ======================================

    print("\n========== DAILY DATASET OVERVIEW ==========")

    day_df.info()

    print("\nColumn Names:")
    print(day_df.columns.tolist())

    print("\nStatistical Summary:")
    print(day_df.describe())


    print("\n========== HOURLY DATASET OVERVIEW ==========")

    hour_df.info()

    print("\nColumn Names:")
    print(hour_df.columns.tolist())

    print("\nStatistical Summary:")
    print(hour_df.describe())


    # ======================================
    # 3. DATA QUALITY CHECKS
    # ======================================

    print("\n========== DATA QUALITY CHECKS ==========")

    inspect_data_quality(day_df, hour_df)

    validate_target_variable(day_df, hour_df)


    # ======================================
    # 4. EXPLORATORY DATA ANALYSIS
    # ======================================

    print("\n========== EXPLORATORY DATA ANALYSIS ==========")

    plot_target_distribution(day_df, hour_df)

    plot_season_vs_rentals(day_df, hour_df)

    plot_weather_vs_rentals(day_df, hour_df)

    plot_workingday_vs_rentals(day_df, hour_df)

    plot_weekday_vs_rentals(day_df, hour_df)

    plot_weather_factors(day_df, hour_df)

    plot_hourly_demand(hour_df)

    plot_holiday_vs_rentals(day_df, hour_df)


    # ======================================
    # 5. DATA PREPARATION
    # ======================================

    print("\n========== DATA PREPARATION ==========")

    X, y = prepare_hourly_data(hour_df)


    # ======================================
    # 6. TRAIN TEST SPLIT
    # ======================================

    print("\n========== TRAIN TEST SPLIT ==========")

    X_train, X_test, y_train, y_test = split_data(X, y)


    # ======================================
    # 7. MODEL TRAINING
    # ======================================

    print("\n========== MODEL TRAINING ==========")

    trained_models = train_models(X_train, y_train)


    # ======================================
    # 8. MODEL EVALUATION
    # ======================================

    print("\n========== MODEL EVALUATION ==========")

    results_df = evaluate_models(
        trained_models,
        X_test,
        y_test
    )


    # ======================================
    # 9. BEST MODEL SELECTION
    # ======================================

    best_model_name = results_df.iloc[0]["Model"]

    best_model = trained_models[best_model_name]

    print("\n========== BEST MODEL ==========")

    print("Best Model:", best_model_name)


    # ======================================
    # 10. FEATURE IMPORTANCE
    # ======================================

    print("\n========== FEATURE IMPORTANCE ==========")

    feature_importance_df = analyze_feature_importance(
        best_model,
        X_train.columns
    )


    # ======================================
    # PROJECT COMPLETED
    # ======================================

    print("\n========================================")
    print("       PROJECT EXECUTION COMPLETED")
    print("========================================")


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()