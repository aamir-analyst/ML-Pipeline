import logging
from pathlib import Path

import pandas as pd

from src.data.data_loader import load_data


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = PROJECT_ROOT / "reports"

REPORT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 2. LOAD DATA
# =====================================================

def load_project_data():

    df = load_data()

    return df


# =====================================================
# 3. ANALYZE TARGET DISTRIBUTION
# =====================================================

def analyze_target_distribution(df):

    target_distribution = (
        df["target"]
        .value_counts(normalize=True)
        * 100
    )

    return target_distribution


# =====================================================
# 4. ANALYZE CITY DEVELOPMENT INDEX
# =====================================================

def analyze_city_development(df):

    result = df.groupby("target")[
        "city_development_index"
    ].mean()

    return result


# =====================================================
# 5. ANALYZE TRAINING HOURS
# =====================================================

def analyze_training_hours(df):

    result = df.groupby("target")[
        "training_hours"
    ].mean()

    return result


# =====================================================
# 6. ANALYZE EXPERIENCE
# =====================================================

def analyze_experience(df):

    result = pd.crosstab(

        df["experience"],

        df["target"],

        normalize="index"

    ) * 100

    return result.sort_values(
        by=1,
        ascending=False
    )


# =====================================================
# 7. ANALYZE RELEVANT EXPERIENCE
# =====================================================

def analyze_relevant_experience(df):

    result = pd.crosstab(

        df["relevent_experience"],

        df["target"],

        normalize="index"

    ) * 100

    return result


# =====================================================
# 8. GENERATE BUSINESS REPORT
# =====================================================

def generate_business_report():

    logging.info("Generating Business Insights Report...")

    df = load_project_data()

    target_distribution = analyze_target_distribution(df)

    city_development = analyze_city_development(df)

    training_hours = analyze_training_hours(df)

    experience_analysis = analyze_experience(df)

    relevant_experience = analyze_relevant_experience(df)

    report = []

    report.append(
        "JOB CHANGE PREDICTION - BUSINESS INSIGHTS"
    )

    report.append("=" * 60)

    report.append("\n1. TARGET DISTRIBUTION")

    report.append(
        target_distribution.to_string()
    )

    report.append("\n2. CITY DEVELOPMENT INDEX")

    report.append(
        city_development.to_string()
    )

    report.append("\n3. TRAINING HOURS")

    report.append(
        training_hours.to_string()
    )

    report.append("\n4. EXPERIENCE ANALYSIS")

    report.append(
        experience_analysis.to_string()
    )

    report.append("\n5. RELEVANT EXPERIENCE ANALYSIS")

    report.append(
        relevant_experience.to_string()
    )

    report.append("\n6. BUSINESS RECOMMENDATIONS")

    report.append(
        """
1. Prioritize candidates predicted as job seekers
   for recruiter follow-up.

2. Investigate early-career candidate requirements
   and placement support.

3. Use city development patterns to guide further
   regional candidate research.

4. Do not rely on training hours alone when
   assessing job-change intention.

5. Use human verification before making final
   recruitment decisions.

6. Monitor false positives and false negatives
   before deploying the model.
"""
    )

    output_path = (
        REPORT_PATH / "business_insights.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("\n".join(report))

    print("\n========== BUSINESS INSIGHTS ==========")

    print("\n".join(report))

    logging.info(
        f"Business report saved: {output_path}"
    )


# =====================================================
# 9. EXECUTION
# =====================================================

if __name__ == "__main__":

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s - %(levelname)s - %(message)s"

    )

    generate_business_report()