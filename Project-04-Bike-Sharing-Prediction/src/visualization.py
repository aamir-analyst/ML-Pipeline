import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Graph storage directory
FIGURE_DIR = BASE_DIR / "reports" / "figures"

# Create directory if it does not exist
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. Target Distribution
# ==========================================

def plot_target_distribution(day_df, hour_df):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.histplot(data=day_df, x="cnt", kde=True, ax=axes[0])
        axes[0].set_title("Daily Bike Rental Distribution")
        axes[0].set_xlabel("Total Bike Rentals")
        axes[0].set_ylabel("Frequency")

        sns.histplot(data=hour_df, x="cnt", kde=True, ax=axes[1])
        axes[1].set_title("Hourly Bike Rental Distribution")
        axes[1].set_xlabel("Total Bike Rentals")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()

        save_path = FIGURE_DIR / "target_distribution.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        print(f"Graph saved: {save_path}")

        plt.close(fig)

    except Exception as e:
        logging.error(f"Error generating target distribution: {e}")
        raise


# ==========================================
# 2. Season vs Rentals
# ==========================================

def plot_season_vs_rentals(day_df, hour_df):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(data=day_df, x="season", y="cnt", ax=axes[0])
        axes[0].set_title("Daily Rentals by Season")
        axes[0].set_xlabel("Season")
        axes[0].set_ylabel("Total Bike Rentals")

        sns.boxplot(data=hour_df, x="season", y="cnt", ax=axes[1])
        axes[1].set_title("Hourly Rentals by Season")
        axes[1].set_xlabel("Season")
        axes[1].set_ylabel("Total Bike Rentals")

        plt.tight_layout()

        save_path = FIGURE_DIR / "season_vs_rentals.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        print(f"Graph saved: {save_path}")

        plt.close(fig)

    except Exception as e:
        logging.error(f"Error generating season graph: {e}")
        raise


# ==========================================
# 3. Weather Situation vs Rentals
# ==========================================

def plot_weather_vs_rentals(day_df, hour_df):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(data=day_df, x="weathersit", y="cnt", ax=axes[0])
        axes[0].set_title("Daily Rentals by Weather")
        axes[0].set_xlabel("Weather Situation")
        axes[0].set_ylabel("Total Bike Rentals")

        sns.boxplot(data=hour_df, x="weathersit", y="cnt", ax=axes[1])
        axes[1].set_title("Hourly Rentals by Weather")
        axes[1].set_xlabel("Weather Situation")
        axes[1].set_ylabel("Total Bike Rentals")

        plt.tight_layout()

        save_path = FIGURE_DIR / "weather_vs_rentals.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        print(f"Graph saved: {save_path}")

        plt.close(fig)

    except Exception as e:
        logging.error(f"Error generating weather graph: {e}")
        raise


# ==========================================
# 4. Working Day vs Rentals
# ==========================================

def plot_workingday_vs_rentals(day_df, hour_df):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(data=day_df, x="workingday", y="cnt", ax=axes[0])
        axes[0].set_title("Daily Rentals by Working Day")
        axes[0].set_xlabel("Working Day (0 = No, 1 = Yes)")
        axes[0].set_ylabel("Total Bike Rentals")

        sns.boxplot(data=hour_df, x="workingday", y="cnt", ax=axes[1])
        axes[1].set_title("Hourly Rentals by Working Day")
        axes[1].set_xlabel("Working Day (0 = No, 1 = Yes)")
        axes[1].set_ylabel("Total Bike Rentals")

        plt.tight_layout()

        save_path = FIGURE_DIR / "workingday_vs_rentals.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        print(f"Graph saved: {save_path}")

        plt.close(fig)

    except Exception as e:
        logging.error(f"Error generating working day graph: {e}")
        raise


# ==========================================
# 5. Weekday vs Rentals
# ==========================================

def plot_weekday_vs_rentals(day_df, hour_df):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(data=day_df, x="weekday", y="cnt", ax=axes[0])
        axes[0].set_title("Daily Rentals by Weekday")
        axes[0].set_xlabel("Weekday (0-6)")
        axes[0].set_ylabel("Total Bike Rentals")

        sns.boxplot(data=hour_df, x="weekday", y="cnt", ax=axes[1])
        axes[1].set_title("Hourly Rentals by Weekday")
        axes[1].set_xlabel("Weekday (0-6)")
        axes[1].set_ylabel("Total Bike Rentals")

        plt.tight_layout()

        save_path = FIGURE_DIR / "weekday_vs_rentals.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        print(f"Graph saved: {save_path}")

        plt.close(fig)

    except Exception as e:
        logging.error(f"Error generating weekday graph: {e}")
        raise


# ==========================================
# 6. Temperature, Humidity, Windspeed
# ==========================================

def plot_weather_factors(day_df, hour_df):
    try:
        features = ["temp", "hum", "windspeed"]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        for i, feature in enumerate(features):

            sns.scatterplot(
                data=day_df,
                x=feature,
                y="cnt",
                alpha=0.5,
                ax=axes[0, i]
            )

            axes[0, i].set_title(f"Daily: {feature} vs Rentals")

            sns.scatterplot(
                data=hour_df,
                x=feature,
                y="cnt",
                alpha=0.3,
                ax=axes[1, i]
            )

            axes[1, i].set_title(f"Hourly: {feature} vs Rentals")

        plt.tight_layout()

        save_path = FIGURE_DIR / "weather_factors_vs_rentals.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        print(f"Graph saved: {save_path}")

        plt.close(fig)

    except Exception as e:
        logging.error(f"Error generating weather factors graph: {e}")
        raise


# ==========================================
# 7. Hourly Demand Analysis
# ==========================================

def plot_hourly_demand(hour_df):
    try:
        fig, ax = plt.subplots(figsize=(12, 5))

        sns.lineplot(
            data=hour_df,
            x="hr",
            y="cnt",
            estimator="mean",
            errorbar=None,
            marker="o",
            ax=ax
        )

        ax.set_title("Average Bike Rentals by Hour")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Bike Rentals")
        ax.set_xticks(range(0, 24))
        ax.grid(True)

        plt.tight_layout()

        save_path = FIGURE_DIR / "hourly_demand.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        print(f"Graph saved: {save_path}")

        plt.close(fig)

    except Exception as e:
        logging.error(f"Error generating hourly demand graph: {e}")
        raise


# ==========================================
# 8. Holiday vs Rentals
# ==========================================

def plot_holiday_vs_rentals(day_df, hour_df):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(data=day_df, x="holiday", y="cnt", ax=axes[0])
        axes[0].set_title("Daily Rentals by Holiday")
        axes[0].set_xlabel("Holiday (0 = No, 1 = Yes)")
        axes[0].set_ylabel("Total Bike Rentals")

        sns.boxplot(data=hour_df, x="holiday", y="cnt", ax=axes[1])
        axes[1].set_title("Hourly Rentals by Holiday")
        axes[1].set_xlabel("Holiday (0 = No, 1 = Yes)")
        axes[1].set_ylabel("Total Bike Rentals")

        plt.tight_layout()

        save_path = FIGURE_DIR / "holiday_vs_rentals.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        print(f"Graph saved: {save_path}")

        plt.close(fig)

    except Exception as e:
        logging.error(f"Error generating holiday graph: {e}")
        raise