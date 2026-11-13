from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ==========================================
# SAVE GRAPH
# ==========================================

def save_plot(filename):

    save_path = FIGURE_DIR / filename

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Graph Saved: {filename}")


# ==========================================
# 1. WINE QUALITY DISTRIBUTION
# ==========================================

def quality_distribution(df):

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x="quality"
    )

    plt.title("Wine Quality Distribution")
    plt.xlabel("Wine Quality")
    plt.ylabel("Number of Wines")

    save_plot("01_quality_distribution.png")


# ==========================================
# 2. PREMIUM DISTRIBUTION
# ==========================================

def premium_distribution(df):

    data = df.copy()

    data["wine_category"] = (
        data["quality"] >= 7
    ).map({
        True: "Premium",
        False: "Non-Premium"
    })

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=data,
        x="wine_category",
        order=["Non-Premium", "Premium"]
    )

    plt.title("Premium vs Non-Premium Wine")
    plt.xlabel("Wine Category")
    plt.ylabel("Number of Wines")

    save_plot("02_premium_distribution.png")


# ==========================================
# 3. FEATURE HISTOGRAMS
# ==========================================

def feature_histograms(df):

    features = df.drop(
        columns=["quality"],
        errors="ignore"
    ).select_dtypes(include="number")

    features.hist(
        figsize=(16, 12),
        bins=30,
        edgecolor="black"
    )

    plt.suptitle(
        "Distribution of Chemical Features",
        fontsize=16
    )

    save_plot("03_feature_histograms.png")


# ==========================================
# 4. CORRELATION HEATMAP
# ==========================================

def correlation_heatmap(df):

    correlation = df.corr(
        numeric_only=True
    )

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Wine Feature Correlation Heatmap")

    save_plot("04_correlation_heatmap.png")


# ==========================================
# 5. OUTLIER BOXPLOTS
# ==========================================

def outlier_boxplots(df):

    features = df.drop(
        columns=["quality"],
        errors="ignore"
    ).select_dtypes(include="number")

    melted_data = features.melt(
        var_name="Feature",
        value_name="Value"
    )

    plt.figure(figsize=(16, 9))

    sns.boxplot(
        data=melted_data,
        x="Feature",
        y="Value"
    )

    plt.title("Outlier Analysis Using Boxplots")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    save_plot("05_outlier_boxplots.png")


# ==========================================
# 6. ALCOHOL VS QUALITY
# ==========================================

def alcohol_quality(df):

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="quality",
        y="alcohol"
    )

    plt.title("Alcohol Content vs Wine Quality")
    plt.xlabel("Wine Quality")
    plt.ylabel("Alcohol Percentage")

    save_plot("06_alcohol_vs_quality.png")


# ==========================================
# 7. VOLATILE ACIDITY VS QUALITY
# ==========================================

def volatile_acidity_quality(df):

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="quality",
        y="volatile acidity"
    )

    plt.title("Volatile Acidity vs Wine Quality")
    plt.xlabel("Wine Quality")
    plt.ylabel("Volatile Acidity")

    save_plot("07_volatile_acidity_vs_quality.png")


# ==========================================
# 8. EDA SUMMARY REPORT
# ==========================================

def eda_summary(df):

    summary = df.describe().T

    summary["missing_values"] = (
        df.isnull().sum()
    )

    summary["unique_values"] = (
        df.nunique()
    )

    summary["skewness"] = (
        df.skew(numeric_only=True)
    )

    summary_path = (
        METRICS_DIR / "eda_summary.csv"
    )

    summary.to_csv(summary_path)

    print("\nEDA Summary Saved!")


# ==========================================
# MAIN EDA FUNCTION
# ==========================================

def generate_eda_report(df):

    print("\n========== EDA STARTED ==========")

    print("\nDataset Shape:", df.shape)

    print("\nGenerating Graphs...")

    quality_distribution(df)

    premium_distribution(df)

    feature_histograms(df)

    correlation_heatmap(df)

    outlier_boxplots(df)

    alcohol_quality(df)

    volatile_acidity_quality(df)

    eda_summary(df)

    print("\n========== EDA COMPLETED ==========")