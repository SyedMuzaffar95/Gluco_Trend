import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from models.load import load_dataset


PLOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "static", "plots"
)
os.makedirs(PLOT_DIR, exist_ok=True)


def chronological_split(X, y, df, test_size=0.20):
    """Split by time to avoid using future observations for training."""
    timestamps = pd.to_datetime(
        df["timestamp"], dayfirst=True, errors="coerce"
    )
    order = np.argsort(timestamps.values)
    cut = int(len(order) * (1 - test_size))

    train_idx = order[:cut]
    test_idx = order[cut:]

    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )


def _compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "MSE": round(mse, 4),
        "RMSE": round(mse ** 0.5, 4),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


SIMPLE_FEATURE = "glucose_lag_1"


def _generate_simple_linear_plot(X_test, y_test, preds, slope, intercept):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_test[SIMPLE_FEATURE], y_test, alpha=0.4, s=12,
        color="#3498db", label="Actual data"
    )
    line_x = np.linspace(
        X_test[SIMPLE_FEATURE].min(), X_test[SIMPLE_FEATURE].max(), 100
    )
    plt.plot(line_x, intercept + slope * line_x, color="#e74c3c",
             linewidth=2.5, label="Best fit line")
    plt.xlabel("Glucose 15 Minutes Earlier (glucose_lag_1)")
    plt.ylabel("Current Glucose")
    plt.title("Simple Linear Regression - Best Fit Line")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "simple_linear_regression.png")
    plt.savefig(path, dpi=140)
    plt.close()
    return "simple_linear_regression.png"


def simple_linear_regression():
    """Simple Linear Regression: one feature (glucose_lag_1) predicts glucose."""
    df = load_dataset()
    data = df.dropna(subset=[SIMPLE_FEATURE, "glucose"])

    X = data[[SIMPLE_FEATURE]]
    y = data["glucose"]

    X_train, X_test, y_train, y_test = chronological_split(X, y, data)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = _compute_metrics(y_test, preds)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    plot_image = _generate_simple_linear_plot(X_test, y_test, preds, slope, intercept)

    results = pd.DataFrame({
        SIMPLE_FEATURE: X_test[SIMPLE_FEATURE].round(2).values,
        "Actual Glucose": y_test.values.round(2),
        "Predicted Glucose": preds.round(2),
        "Abs Error": np.abs(y_test.values - preds).round(2),
    })

    return metrics, results, slope, intercept, plot_image


def _generate_logistic_plots(X_test, y_test, probs, cm):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im = axes[0].imshow(cm, cmap="Blues")
    classes = ["Normal (<140)", "High (>=140)"]
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(classes, fontsize=9)
    axes[0].set_yticklabels(classes, fontsize=9)
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                         fontsize=16, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix")

    colors = ["#2ecc71", "#e74c3c"]
    labels = ["Normal", "High"]
    for cls in [0, 1]:
        mask = (y_test == cls).values
        axes[1].scatter(
            X_test[SIMPLE_FEATURE][mask], probs[mask],
            alpha=0.4, s=12, color=colors[cls], label=labels[cls]
        )
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Glucose 15 Minutes Earlier (glucose_lag_1)")
    axes[1].set_ylabel("Predicted Probability of High Glucose")
    axes[1].set_title("Logistic Regression - Predicted Probabilities")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "logistic_regression.png")
    plt.savefig(path, dpi=140)
    plt.close(fig)
    return "logistic_regression.png"


def simple_logistic_regression(threshold=140):
    """Logistic Regression classifying glucose as High (>=threshold) vs Normal."""
    df = load_dataset()
    data = df.dropna(subset=[SIMPLE_FEATURE, "glucose"]).copy()
    data["high_glucose"] = (data["glucose"] >= threshold).astype(int)

    X = data[[SIMPLE_FEATURE]]
    y = data["high_glucose"]

    X_train, X_test, y_train, y_test = chronological_split(X, y, data)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression()),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "F1": round(f1_score(y_test, preds), 4),
        "high_cases": int(y.sum()),
        "normal_cases": int(len(y) - y.sum()),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    cm = confusion_matrix(y_test, preds)
    plot_image = _generate_logistic_plots(X_test, y_test.reset_index(drop=True), probs, cm)
    cm = cm.tolist()

    results = pd.DataFrame({
        SIMPLE_FEATURE: X_test[SIMPLE_FEATURE].round(2).values,
        "Actual Glucose": data.loc[X_test.index, "glucose"].round(2).values,
        "Actual Class": np.where(y_test.values == 1, "High", "Normal"),
        "High Probability": probs.round(3),
        "Predicted Class": np.where(preds == 1, "High", "Normal"),
    })

    return metrics, cm, results, plot_image


# =========================================================
# TREE & ENSEMBLE MODELS
# Same logic as the Placement Complete_Code, adapted to the
# glucose project. Target: HighGlucose = 1 if glucose >= 140
# else 0. Features: glucose_lag_1 (previous glucose),
# insulin_basal, carbs.
# =========================================================

TREE_FEATURES = ["glucose_lag_1", "insulin_basal", "carbs"]
CLASS_THRESHOLD = 140


def _tree_data():
    """Build the modeling DataFrame: 3 selected features + HighGlucose target."""
    df = load_dataset()
    data = df.dropna(subset=TREE_FEATURES + ["glucose"]).copy()
    data["HighGlucose"] = (data["glucose"] >= CLASS_THRESHOLD).astype(int)
    return data


def decision_tree(df):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report

    X = df[TREE_FEATURES]
    y = df["HighGlucose"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Decision Tree Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    metrics = {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    return model, metrics, cm


def decision_tree_pruned(df):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report

    X = df[TREE_FEATURES]
    y = df["HighGlucose"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Pruned Decision Tree Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    metrics = {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    return model, metrics, cm


def random_forest_model(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    X = df[TREE_FEATURES]
    y = df["HighGlucose"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    metrics = {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    return model, metrics, cm


def adaboost_model(df):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report

    X = df[TREE_FEATURES].copy()
    y = df["HighGlucose"].copy()

    # Handle missing values
    X = X.fillna(X.median())

    # Remove rows where target is missing
    valid_rows = y.notna()
    X = X[valid_rows]
    y = y[valid_rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("AdaBoost Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    metrics = {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    return model, metrics, cm


def gradient_boosting_model(df):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report

    X = df[TREE_FEATURES]
    X = X.fillna(X.median())
    y = df["HighGlucose"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        loss="log_loss",
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Gradient Boosting Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    metrics = {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    return model, metrics, cm


def xgboost_model(df):
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, classification_report

    X = df[TREE_FEATURES]
    X = X.fillna(X.median())
    y = df["HighGlucose"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("XGBoost Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    metrics = {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    return model, metrics, cm


def compare(sample_df, m1, m2, m3):
    """Compare Decision Tree, Pruned Decision Tree and Random Forest on samples."""
    X_sample = sample_df[TREE_FEATURES]
    y_actual = sample_df["HighGlucose"]
    pred_dt = m1.predict(X_sample)
    pred_pruned = m2.predict(X_sample)
    pred_rf = m3.predict(X_sample)

    results = pd.DataFrame({
        "Row Index": sample_df.index,
        "Actual": y_actual.values,
        "Decision Tree": pred_dt,
        "Pruned Decision Tree": pred_pruned,
        "Random Forest": pred_rf,
    })
    print("\nPrediction Comparison for 10 Sample Rows")
    print("=" * 80)
    print(results.to_string(index=False))
    return results


def _generate_tree_plots(names, metrics_list, cms):
    colors = ["#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#e67e22", "#16a085"]

    # Accuracy comparison bar chart
    plt.figure(figsize=(10, 5))
    accs = [m["Accuracy"] for m in metrics_list]
    bars = plt.bar(names, accs, color=colors)
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy of Tree & Ensemble Classifiers")
    plt.ylim(0, 1.0)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.005,
                 f"{acc:.4f}", ha="center", fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    acc_path = os.path.join(PLOT_DIR, "tree_accuracy_comparison.png")
    plt.savefig(acc_path, dpi=140)
    plt.close()

    # Confusion matrices grid (6 models -> 2x3)
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9))
    axes = np.atleast_1d(axes).ravel()
    for idx, (name, cm) in enumerate(zip(names, cms)):
        ax = axes[idx]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                        fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "High"], fontsize=9)
        ax.set_yticklabels(["Normal", "High"], fontsize=9)
        ax.set_title(name)
    for idx in range(len(names), len(axes)):
        axes[idx].axis("off")
    fig.suptitle("Confusion Matrices (High >= 140)", fontsize=14)
    plt.tight_layout()
    cm_path = os.path.join(PLOT_DIR, "tree_confusion_matrix.png")
    plt.savefig(cm_path, dpi=140)
    plt.close(fig)

    return "tree_accuracy_comparison.png", "tree_confusion_matrix.png"


def ensemble_summary():
    """Train all tree/ensemble classifiers and return a summary for the page."""
    data = _tree_data()

    names = [
        "Decision Tree", "Pruned Decision Tree", "Random Forest",
        "AdaBoost", "Gradient Boosting", "XGBoost",
    ]
    funcs = [
        decision_tree, decision_tree_pruned, random_forest_model,
        adaboost_model, gradient_boosting_model, xgboost_model,
    ]

    metrics_list, cms, models = [], [], []
    for fn in funcs:
        model, metrics, cm = fn(data)
        models.append(model)
        metrics_list.append(metrics)
        cms.append(cm)

    sample_df = data.sample(n=10, random_state=42)
    comparison = compare(sample_df, models[0], models[1], models[2])
    plots = _generate_tree_plots(names, metrics_list, cms)

    return {
        "names": names,
        "metrics": metrics_list,
        "confusions": cms,
        "comparison": comparison,
        "feature_names": TREE_FEATURES,
        "threshold": CLASS_THRESHOLD,
        "high_cases": int(data["HighGlucose"].sum()),
        "normal_cases": int((data["HighGlucose"] == 0).sum()),
        "train_rows": len(data),
        "accuracy_plot": plots[0],
        "cm_plot": plots[1],
    }


def regression_summary():
    """Compare all candidate regressors on the glucose target.

    Delegates training to models.train_model.train_all() (cached per
    dataset) and exposes the comparison, best model and plots for the
    /regression page.  The target stays a continuous glucose value -
    it is never converted into classes.
    """
    from models.train_model import train_all

    result = train_all(use_cache=True)
    if result.get("error"):
        return result

    comparison_rows = [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in result["comparison"]
    ]
    return {
        "error": None,
        "task_type": result["task_type"],
        "target": result["target"],
        "comparison": comparison_rows,
        "best_model": result["best_model"],
        "best_metrics": result["best_metrics"],
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "feature_count": len(result["feature_names"]),
        "plots": result["plots"],
        "trained_at": result["trained_at"],
    }


# MAIN PROGRAM
if __name__ == "__main__":
    data = _tree_data()
    print("Dataset Shape:", data.shape)
    print("\nFirst 5 Rows:")
    print(data.head())
    print(data["HighGlucose"])

    m1, _, _ = decision_tree(data)
    m2, _, _ = decision_tree_pruned(data)
    m3, _, _ = random_forest_model(data)

    sample_df = data.sample(n=10, random_state=42)
    compare(sample_df, m1, m2, m3)

    print("=" * 100)
    m4, _, _ = adaboost_model(data)
    print("=" * 100)
    m5, _, _ = gradient_boosting_model(data)
    print("=" * 100)
    m6, _, _ = xgboost_model(data)
    print("=" * 100)