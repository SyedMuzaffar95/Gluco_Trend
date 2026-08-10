from flask import Flask, render_template

from models.load import load_dataset

from models.eda import (
    dataset_info,
    dataset_description,
    missing_values,
    duplicate_values,
    glucose_statistics,
    glucose_distribution,
    glucose_boxplot,
    correlation_heatmap
)

import matplotlib.pyplot as plt


app = Flask(__name__)


# ==========================================================
# EDA EXECUTION
# ==========================================================

def run_eda():

    df = load_dataset()

    print("\n")
    print("=" * 70)
    print("GLUCOBENCH EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    dataset_info(df)

    dataset_description(df)

    missing_values(df)

    duplicate_values(df)

    glucose_statistics(df)

    glucose_distribution(df)

    glucose_boxplot(df)

    correlation_heatmap(df)

    print("\n" + "=" * 70)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)

    plt.show()


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def index():

    return render_template("index.html")


# ==========================================================
# LOAD DATASET PAGE
# ==========================================================

@app.route("/load")
def load():

    df = load_dataset()

    return render_template(
        "load.html",
        data=df.head().to_html(
            classes="data-table",
            index=False
        ),
        rows=df.shape[0],
        columns=df.shape[1]
    )


# ==========================================================
# EDA PAGE
# ==========================================================

@app.route("/eda")
def eda():

    return render_template("eda.html")


# ==========================================================
# START APPLICATION
# ==========================================================

if __name__ == "__main__":

    run_eda()

    app.run(
        debug=True,
        use_reloader=False
    )