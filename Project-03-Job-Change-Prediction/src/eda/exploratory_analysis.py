import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.data_loader import load_data


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIGURE_PATH = PROJECT_ROOT / "reports" / "figures"

FIGURE_PATH.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# =====================================================
# 2. DATASET OVERVIEW
# =====================================================

def dataset_overview(df):

    logging.info("Starting Dataset Overview...")

    print("\n========== DATASET SHAPE ==========")
    print(df.shape)

    print("\n========== FIRST 5 RECORDS ==========")
    print(df.head())

    print("\n========== LAST 5 RECORDS ==========")
    print(df.tail())

    print("\n========== DATASET INFORMATION ==========")
    df.info()

    print("\n========== STATISTICAL SUMMARY ==========")
    print(df.describe(include="all").T)

    print("\n========== COLUMN NAMES ==========")
    print(df.columns.tolist())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    logging.info("Dataset Overview Completed.")


# =====================================================
# 3. MISSING VALUES ANALYSIS
# =====================================================

def missing_values_analysis(df):

    logging.info("Starting Missing Values Analysis...")

    missing = pd.DataFrame({

        "Missing_Count": df.isnull().sum(),

        "Missing_Percentage": df.isnull().mean() * 100

    })

    missing = missing[missing["Missing_Count"] > 0]

    missing = missing.sort_values(
        by="Missing_Percentage",
        ascending=False
    )

    print("\n========== MISSING VALUES ==========")

    if missing.empty:

        print("No Missing Values Found.")

    else:

        print(missing)

        plt.figure(figsize=(12, 6))

        sns.barplot(
            x=missing.index,
            y=missing["Missing_Percentage"]
        )

        plt.title("Missing Values Percentage")

        plt.xlabel("Features")
        plt.ylabel("Missing Percentage (%)")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            FIGURE_PATH / "missing_values.png",
            dpi=300
        )

        plt.close()


# =====================================================
# 4. DUPLICATE ANALYSIS
# =====================================================

def duplicate_analysis(df):

    logging.info("Starting Duplicate Analysis...")

    duplicates = df.duplicated().sum()

    print("\n========== DUPLICATE ANALYSIS ==========")

    print(f"Total Duplicate Rows: {duplicates}")

    logging.info("Duplicate Analysis Completed.")


# =====================================================
# 5. TARGET DISTRIBUTION
# =====================================================

def target_distribution(df):

    logging.info("Starting Target Distribution Analysis...")

    target_counts = df["target"].value_counts(
        dropna=False
    ).sort_index()

    target_percentage = (
        df["target"].value_counts(normalize=True) * 100
    ).sort_index()

    print("\n========== TARGET DISTRIBUTION ==========")

    print(target_counts)

    print("\n========== TARGET PERCENTAGE ==========")

    print(target_percentage.round(2))

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="target",
        hue="target",
        legend=False
    )

    plt.title("Target Variable Distribution")

    plt.xlabel("Job Change Target")
    plt.ylabel("Number of Candidates")

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH / "target_distribution.png",
        dpi=300
    )

    plt.close()

    logging.info("Target Distribution Analysis Completed.")


# =====================================================
# 6. NUMERICAL UNIVARIATE ANALYSIS
# =====================================================

def numerical_univariate_analysis(df):

    logging.info("Starting Numerical Univariate Analysis...")

    numerical_features = [

        "city_development_index",
        "training_hours"

    ]

    for feature in numerical_features:

        print(f"\n========== {feature.upper()} ==========")

        print(df[feature].describe())

        # Histogram

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=feature,
            kde=True,
            bins=30
        )

        plt.title(f"Distribution of {feature}")

        plt.xlabel(feature)
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            FIGURE_PATH / f"{feature}_distribution.png",
            dpi=300
        )

        plt.close()

        # Boxplot

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x=feature
        )

        plt.title(f"Boxplot of {feature}")

        plt.xlabel(feature)

        plt.tight_layout()

        plt.savefig(
            FIGURE_PATH / f"{feature}_boxplot.png",
            dpi=300
        )

        plt.close()

    logging.info("Numerical Univariate Analysis Completed.")


# =====================================================
# 7. CATEGORICAL UNIVARIATE ANALYSIS
# =====================================================

def categorical_univariate_analysis(df):

    logging.info("Starting Categorical Analysis...")

    categorical_features = [

        "gender",
        "relevent_experience",
        "enrolled_university",
        "education_level",
        "major_discipline",
        "experience",
        "company_size",
        "company_type",
        "last_new_job",
        "city"

    ]

    for feature in categorical_features:

        print(f"\n========== {feature.upper()} ==========")

        # Category Frequency

        frequency = df[feature].value_counts(
            dropna=False
        )

        print("\nCategory Frequency:")
        print(frequency)

        # Category Percentage

        percentage = (
            df[feature].value_counts(
                normalize=True,
                dropna=False
            ) * 100
        )

        print("\nCategory Percentage:")
        print(percentage.round(2))

        plot_data = df[feature].fillna("Missing")

        # City: Top 15 Categories

        if feature == "city":

            city_counts = plot_data.value_counts().head(15)

            plt.figure(figsize=(12, 6))

            sns.barplot(
                x=city_counts.index,
                y=city_counts.values
            )

            plt.title("Top 15 Cities")

        else:

            plt.figure(figsize=(10, 6))

            order = plot_data.value_counts().index

            sns.countplot(
                x=plot_data,
                order=order
            )

            plt.title(f"Distribution of {feature}")

            plt.xticks(rotation=45)

        plt.xlabel(feature)
        plt.ylabel("Number of Candidates")

        plt.tight_layout()

        plt.savefig(
            FIGURE_PATH / f"{feature}_distribution.png",
            dpi=300
        )

        plt.close()

    logging.info("Categorical Analysis Completed.")


# =====================================================
# 8. CATEGORICAL VS TARGET ANALYSIS
# =====================================================

def categorical_target_analysis(df):

    logging.info("Starting Categorical vs Target Analysis...")

    categorical_features = [

        "gender",
        "relevent_experience",
        "enrolled_university",
        "education_level",
        "major_discipline",
        "experience",
        "company_size",
        "company_type",
        "last_new_job"

    ]

    for feature in categorical_features:

        print(f"\n========== {feature} VS TARGET ==========")

        # Calculate category-wise percentage

        crosstab = pd.crosstab(
            df[feature].fillna("Missing"),
            df["target"],
            normalize="index"
        ) * 100

        print(crosstab.round(2))

        # Visualization

        plot_df = df.copy()

        plot_df[feature] = plot_df[feature].fillna("Missing")

        plt.figure(figsize=(10, 6))

        sns.countplot(
            data=plot_df,
            x=feature,
            hue="target"
        )

        plt.title(f"{feature} vs Job Change Target")

        plt.xlabel(feature)
        plt.ylabel("Number of Candidates")

        plt.xticks(rotation=45)

        plt.legend(
            title="Target",
            labels=["Not Looking", "Looking"]
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_PATH / f"{feature}_vs_target.png",
            dpi=300
        )

        plt.close()

    logging.info("Categorical Target Analysis Completed.")


# =====================================================
# 9. NUMERICAL VS TARGET ANALYSIS
# =====================================================

def numerical_target_analysis(df):

    logging.info("Starting Numerical vs Target Analysis...")

    numerical_features = [

        "city_development_index",
        "training_hours"

    ]

    for feature in numerical_features:

        print(f"\n========== {feature} VS TARGET ==========")

        # Group-wise Statistics

        summary = df.groupby("target")[feature].agg(
            ["count", "mean", "median", "std", "min", "max"]
        )

        print(summary)

        # Boxplot

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x="target",
            y=feature,
            hue="target",
            legend=False
        )

        plt.title(f"{feature} vs Job Change Target")

        plt.xlabel("Job Change Target")
        plt.ylabel(feature)

        plt.tight_layout()

        plt.savefig(
            FIGURE_PATH / f"{feature}_vs_target_boxplot.png",
            dpi=300
        )

        plt.close()

        # Violin Plot

        plt.figure(figsize=(8, 5))

        sns.violinplot(
            data=df,
            x="target",
            y=feature,
            hue="target",
            legend=False
        )

        plt.title(f"{feature} Distribution by Target")

        plt.tight_layout()

        plt.savefig(
            FIGURE_PATH / f"{feature}_vs_target_violin.png",
            dpi=300
        )

        plt.close()

    logging.info("Numerical vs Target Analysis Completed.")


# =====================================================
# 10. CORRELATION ANALYSIS
# =====================================================

def correlation_analysis(df):

    logging.info("Starting Correlation Analysis...")

    numerical_df = df[

        [
            "city_development_index",
            "training_hours",
            "target"
        ]

    ]

    correlation = numerical_df.corr(
        method="pearson"
    )

    print("\n========== CORRELATION MATRIX ==========")

    print(correlation)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Numerical Feature Correlation")

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH / "correlation_matrix.png",
        dpi=300
    )

    plt.close()

    logging.info("Correlation Analysis Completed.")


# =====================================================
# 11. MAIN EDA PIPELINE
# =====================================================

def run_eda():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    try:

        logging.info("Starting Complete EDA Pipeline...")

        # Load Dataset

        df = load_data()

        # Dataset Overview

        dataset_overview(df)

        # Missing Values

        missing_values_analysis(df)

        # Duplicate Analysis

        duplicate_analysis(df)

        # Target Distribution

        target_distribution(df)

        # Numerical Analysis

        numerical_univariate_analysis(df)

        # Categorical Analysis

        categorical_univariate_analysis(df)

        # Categorical vs Target

        categorical_target_analysis(df)

        # Numerical vs Target

        numerical_target_analysis(df)

        # Correlation Analysis

        correlation_analysis(df)

        logging.info("Complete EDA Pipeline Executed Successfully.")

        print("\n==========================================")
        print("COMPLETE EDA PIPELINE EXECUTED SUCCESSFULLY")
        print("==========================================")

        print(f"\nVisualizations saved at: {FIGURE_PATH}")

    except Exception:

        logging.exception("EDA Pipeline Failed.")

        raise


# =====================================================
# 12. EXECUTION
# =====================================================

if __name__ == "__main__":

    run_eda()