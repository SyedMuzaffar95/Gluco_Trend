import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
PLOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "static", "plots"
)
os.makedirs(PLOT_DIR, exist_ok=True)


def dataset_info(df):
    buffer = StringIO()
    df.info(buf=buffer)
    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    print(buffer.getvalue())
    return buffer.getvalue()


def dataset_description(df):
    print("\n" + "=" * 70)
    print("STATISTICAL DESCRIPTION")
    print("=" * 70)
    result = df.describe()
    print(result)
    return result


def missing_values(df):
    result = df.isnull().sum()
    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)
    print(result)
    return result


def duplicate_values(df):
    result = int(df.duplicated().sum())
    print("Duplicate rows:", result)
    return result


def glucose_statistics(df):
    glucose = df["glucose"]
    return {
        "Mean": round(glucose.mean(), 2),
        "Median": round(glucose.median(), 2),
        "Minimum": round(glucose.min(), 2),
        "Maximum": round(glucose.max(), 2),
        "Standard Deviation": round(glucose.std(), 2),
        "Variance": round(glucose.var(), 2),
    }


def glucose_distribution(df):
    path = os.path.join(PLOT_DIR, "glucose_distribution.png")
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="glucose", kde=True)
    plt.title("Glucose Distribution")
    plt.xlabel("Glucose (mg/dL)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return "glucose_distribution.png"


def glucose_boxplot(df):
    path = os.path.join(PLOT_DIR, "glucose_boxplot.png")
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df["glucose"])
    plt.title("Glucose Boxplot")
    plt.xlabel("Glucose (mg/dL)")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return "glucose_boxplot.png"


def correlation_heatmap(df):
    numerical_df = df.select_dtypes(include=np.number)
    correlation = numerical_df.corr()

    path = os.path.join(PLOT_DIR, "correlation_heatmap.png")
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="Greens")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

    return correlation, "correlation_heatmap.png"
