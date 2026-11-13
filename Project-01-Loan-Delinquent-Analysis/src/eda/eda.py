import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def target_distribution(df):

    print("\nTarget Distribution:")
    print(df["isDelinquent"].value_counts())

    print("\nPercentage Distribution:")
    print(df["isDelinquent"].value_counts(normalize=True) * 100)

    plt.figure(figsize=(7, 5))

    sns.countplot(data=df, x="isDelinquent")

    plt.title("Loan Delinquency Distribution")
    plt.xlabel("0 = Non-Delinquent | 1 = Delinquent")
    plt.ylabel("Number of Customers")

    plt.show()



def age_analysis(df):

    age_result = pd.crosstab(
        df["age"],
        df["isDelinquent"],
        normalize="index"
    ) * 100

    print("\nAge-wise Delinquency:")
    print(age_result)

    age_result.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5)
    )

    plt.title("Age-wise Delinquency Percentage")
    plt.ylabel("Percentage")

    plt.show()


def fico_analysis(df):

    fico_result = pd.crosstab(
        df["FICO"],
        df["isDelinquent"],
        normalize="index"
    ) * 100

    print("\nFICO-wise Delinquency:")
    print(fico_result)

    fico_result.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5)
    )

    plt.title("FICO-wise Delinquency Percentage")
    plt.ylabel("Percentage")

    plt.show()


if __name__ == "__main__":

    from src.data.load_data import load_dataset

    df = load_dataset()

    target_distribution(df)

    age_analysis(df)

    fico_analysis(df)