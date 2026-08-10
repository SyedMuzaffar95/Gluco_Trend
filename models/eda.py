import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO


def dataset_info(df):

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    buffer = StringIO()
    df.info(buf=buffer)

    print(buffer.getvalue())


def dataset_description(df):

    print("\n" + "=" * 70)
    print("STATISTICAL DESCRIPTION")
    print("=" * 70)

    print(df.describe())


def missing_values(df):

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    print(df.isnull().sum())


def duplicate_values(df):

    print("\n" + "=" * 70)
    print("DUPLICATE VALUES")
    print("=" * 70)

    print("Duplicate rows:", df.duplicated().sum())


def glucose_statistics(df):

    print("\n" + "=" * 70)
    print("GLUCOSE STATISTICS")
    print("=" * 70)

    glucose = df["glucose"]

    print("Mean:", glucose.mean())
    print("Median:", glucose.median())
    print("Minimum:", glucose.min())
    print("Maximum:", glucose.max())
    print("Standard Deviation:", glucose.std())
    print("Variance:", glucose.var())


def glucose_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="glucose",
        kde=True
    )

    plt.title("Glucose Distribution")
    plt.xlabel("Glucose (mg/dL)")
    plt.ylabel("Frequency")

    plt.tight_layout()


def glucose_boxplot(df):

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        x=df["glucose"]
    )

    plt.title("Glucose Boxplot")
    plt.xlabel("Glucose (mg/dL)")

    plt.tight_layout()


def correlation_heatmap(df):

    numerical_df = df.select_dtypes(
        include=np.number
    )

    correlation = numerical_df.corr()

    print("\n" + "=" * 70)
    print("CORRELATION MATRIX")
    print("=" * 70)

    print(correlation)

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="Greens"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()