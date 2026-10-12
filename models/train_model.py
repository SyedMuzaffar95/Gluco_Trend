"""Explicit training workflow for the glucose prediction model.

Used by:
  - Docker image build:  RUN python models/train_model.py
  - Flask /train route:   train_and_save()
  - Flask /regression and /model-comparison routes: train_all() (cached)

Importing this module never starts a training job.
"""
from models.load import load_dataset
import hashlib
import os
import tempfile
import time
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from models.load import load_dataset
from models.preprocessing import prepare_features

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:  # xgboost is optional; the rest still trains
    XGB_AVAILABLE = False


from pathlib import Path
import os
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "glucose_model.joblib"
PLOT_DIR = BASE_DIR / "static" / "plots"

# Set this to the actual CSV filename in your dataset folder.
DATA_PATH = BASE_DIR / "dataset" / "YOUR_DATASET.csv"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def _dataset_key():
    """Cheap fingerprint of the dataset file for cache invalidation."""
    try:
        stat = os.stat(DATA_PATH)
        raw = "{}:{}:{}".format(DATA_PATH, stat.st_size, int(stat.st_mtime))
    except OSError:
        raw = DATA_PATH
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def detect_task_type(y):
    """Classify the target as regression or classification from its schema.

    Continuous numeric targets (e.g. glucose concentration) stay regression;
    they are never arbitrarily converted into classes.
    """
    if pd.api.types.is_float_dtype(y) and y.nunique() > 10:
        return "regression"
    if y.dtype == object or str(y.dtype) == "category":
        return "classification"
    if pd.api.types.is_numeric_dtype(y) and y.nunique() <= 10:
        return "classification"
    if pd.api.types.is_numeric_dtype(y):
        return "regression"
    raise ValueError("The target must be categorical or numeric.")


def chronological_split(X, y, timestamps, test_size=TEST_SIZE):
    """80/20 time-ordered split so the model never trains on the future."""
    order = np.argsort(pd.to_datetime(timestamps, dayfirst=True,
                                      errors="coerce").values)
    cut = int(len(order) * (1 - test_size))
    train_idx, test_idx = order[:cut], order[cut:]
    return (X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx])


def _regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(mse ** 0.5), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def _candidate_estimators():
    """(name, estimator, needs_subsample) triples for the regression suite."""
    candidates = [
        ("Linear Regression", LinearRegression(), False),
        ("Ridge Regression", Ridge(alpha=1.0), False),
        ("Decision Tree Regressor",
         DecisionTreeRegressor(random_state=RANDOM_STATE), False),
        ("Random Forest Regressor",
         RandomForestRegressor(n_estimators=100, max_depth=8,
                               random_state=RANDOM_STATE, n_jobs=-1), False),
        ("K-Nearest Neighbors Regressor",
         KNeighborsRegressor(n_neighbors=10), False),
        ("Support Vector Regressor", SVR(kernel="rbf"), True),
        ("Gradient Boosting Regressor",
         GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                   max_depth=3, random_state=RANDOM_STATE),
         False),
    ]
    if XGB_AVAILABLE:
        candidates.append((
            "XGBoost Regressor",
            XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=RANDOM_STATE, n_jobs=-1),
            False,
        ))
    return candidates


def _generate_comparison_plots(rows, best_name, y_test, best_preds):
    names = [r["model"] for r in rows]
    r2s = [r["R2"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2d6a4f" if n == best_name else "#52b788" for n in names]
    bars = ax.bar(names, r2s, color=colors)
    ax.set_ylabel("R² (test set)")
    ax.set_title("Regression Model Comparison - Test R²")
    ax.set_ylim(0, max(1.0, max(r2s) * 1.05))
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                "{:.4f}".format(val), ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    comparison_plot = "regression_model_comparison.png"
    plt.savefig(os.path.join(PLOT_DIR, comparison_plot), dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, best_preds, alpha=0.3, s=10, color="#3498db")
    lo = float(min(y_test.min(), best_preds.min()))
    hi = float(max(y_test.max(), best_preds.max()))
    ax.plot([lo, hi], [lo, hi], color="#e63946", linewidth=1.5)
    ax.set_xlabel("Actual Glucose (mg/dL)")
    ax.set_ylabel("Predicted Glucose (mg/dL)")
    ax.set_title("Actual vs Predicted - {}".format(best_name))
    plt.tight_layout()
    scatter_plot = "regression_actual_vs_predicted.png"
    plt.savefig(os.path.join(PLOT_DIR, scatter_plot), dpi=140)
    plt.close()

    residuals = y_test - best_preds
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuals, bins=60, color="#3498db", alpha=0.85)
    ax.axvline(0.0, color="#e63946", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (actual - predicted, mg/dL)")
    ax.set_ylabel("Sample Count")
    ax.set_title("Residual Distribution - {}".format(best_name))
    plt.tight_layout()
    residuals_plot = "regression_residuals.png"
    plt.savefig(os.path.join(PLOT_DIR, residuals_plot), dpi=140)
    plt.close()

    return {
        "comparison": comparison_plot,
        "scatter": scatter_plot,
        "residuals": residuals_plot,
    }


def train_all(use_cache=True):
    """Train every candidate regressor and compare them on the test set.

    Returns a dict with the comparison table, best model name, fitted
    estimators (private keys) and plot filenames.  Cached per dataset
    fingerprint so Flask pages do not retrain on every request.
    """
    key = _dataset_key()
    if use_cache and _CACHE["result"] is not None and _CACHE["key"] == key:
        return _CACHE["result"]

    if not os.path.exists(DATA_PATH):
        return {"error": "Dataset not found: {}".format(DATA_PATH)}

    df = load_dataset()
    if TARGET not in df.columns:
        return {
            "error": "Target '{}' not found. Available columns: {}".format(
                TARGET, df.columns.tolist()
            ),
        }

    X, y, preprocessor, numeric_cols, categorical_cols = prepare_features(df)
    task_type = detect_task_type(y)
    if task_type != "regression":
        return {
            "error": (
                "Target '{}' was detected as {} but this training module "
                "implements regression only.".format(TARGET, task_type)
            ),
        }

    timestamps = df["timestamp"]
    X_train, X_test, y_train, y_test = chronological_split(
        X, y, timestamps, test_size=TEST_SIZE
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    rows = []
    fitted = {}
    for name, estimator, needs_subsample in _candidate_estimators():
        started = time.time()
        if needs_subsample and len(X_train_t) > SVR_MAX_TRAIN_ROWS:
            rng = np.random.RandomState(RANDOM_STATE)
            take = rng.choice(len(X_train_t), size=SVR_MAX_TRAIN_ROWS,
                              replace=False)
            take.sort()
            fit_X, fit_y = X_train_t[take], y_train.iloc[take]
        else:
            fit_X, fit_y = X_train_t, y_train
        try:
            estimator.fit(fit_X, fit_y)
            preds = estimator.predict(X_test_t)
        except Exception as exc:  # noqa: BLE001 - one broken model must not stop the rest
            rows.append({
                "model": name,
                "MAE": None, "MSE": None, "RMSE": None, "R2": None,
                "fit_seconds": round(time.time() - started, 2),
                "note": "training failed: {}".format(exc),
            })
            continue
        metrics = _regression_metrics(y_test, preds)
        rows.append({
            "model": name,
            **metrics,
            "fit_seconds": round(time.time() - started, 2),
            "note": (
                "trained on a {}-row subsample".format(SVR_MAX_TRAIN_ROWS)
                if needs_subsample and len(X_train_t) > SVR_MAX_TRAIN_ROWS
                else None
            ),
        })
        fitted[name] = estimator

    scored = [r for r in rows if r["R2"] is not None]
    if not scored:
        return {"error": "Every candidate model failed to train."}
    best_row = max(scored, key=lambda r: r["R2"])
    best_name = best_row["model"]
    best_preds = fitted[best_name].predict(X_test_t)

    plots = _generate_comparison_plots(
        scored, best_name, y_test.to_numpy(), best_preds
    )

    result = {
        "error": None,
        "task_type": "regression",
        "target": TARGET,
        "comparison": sorted(rows, key=lambda r: (r["R2"] is None, -(r["R2"] or 0))),
        "best_model": best_name,
        "best_metrics": best_row,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_names": list(X.columns),
        "numeric_cols": list(numeric_cols),
        "categorical_cols": list(categorical_cols),
        "plots": plots,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "_fitted": fitted,
        "_best_estimator": fitted[best_name],
        "_preprocessor": preprocessor,
    }
    _CACHE["key"] = key
    _CACHE["result"] = result
    return result


def _field_metadata(X, numeric_cols, categorical_cols):
    """Per-feature stats used by the prediction form and its validation."""
    stats = {}
    for col in numeric_cols:
        series = pd.to_numeric(X[col], errors="coerce")
        stats[col] = {
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
        }
    categories = {}
    for col in categorical_cols:
        values = sorted(X[col].dropna().astype(str).unique().tolist())
        categories[col] = values
    return stats, categories


def train_and_save():
    """Train, compare and persist the best model plus full metadata.

    The artifact is written to a temporary file first, so a failed run
    never overwrites a previously valid model.
    """
    result = train_all(use_cache=False)
    if result.get("error"):
        return result

    best_pipeline = Pipeline(steps=[
        ("preprocessor", result["_preprocessor"]),
        ("estimator", result["_best_estimator"]),
    ])

    X, y, _, numeric_cols, categorical_cols = prepare_features(load_dataset())
    field_stats, categories = _field_metadata(X, numeric_cols, categorical_cols)

    artifact = {
        "model": best_pipeline,
        "best_model_name": result["best_model"],
        "task_type": "regression",
        "target": TARGET,
        "feature_names": result["feature_names"],
        "numeric_cols": result["numeric_cols"],
        "categorical_cols": result["categorical_cols"],
        "categories": categories,
        "field_stats": field_stats,
        "comparison": result["comparison"],
        "metrics": result["best_metrics"],
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "trained_at": result["trained_at"],
        "dataset_path": DATA_PATH,
        "model_name": type(result["_best_estimator"]).__name__,
    }

    directory = os.path.dirname(MODEL_PATH)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".joblib", dir=directory)
    os.close(fd)
    try:
        joblib.dump(artifact, tmp_path)
        os.replace(tmp_path, MODEL_PATH)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "error": None,
        "best_model": artifact["best_model_name"],
        "metrics": artifact["metrics"],
        "comparison": artifact["comparison"],
        "model_path": MODEL_PATH,
        "model_size_kb": round(os.path.getsize(MODEL_PATH) / 1024, 1),
        "trained_at": artifact["trained_at"],
        "train_rows": artifact["train_rows"],
        "test_rows": artifact["test_rows"],
        "plots": result["plots"],
    }


def load_artifact():
    """Load the saved model package, or return None when unavailable."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:  # noqa: BLE001 - corrupt artifact is treated as missing
        return None


def artifact_status():
    """Cheap status dictionary for the dashboard, train and predict pages."""
    if not os.path.exists(MODEL_PATH):
        return {
            "exists": False,
            "path": MODEL_PATH,
            "message": "No trained model saved yet. Use the Train Model page.",
        }
    artifact = load_artifact()
    if artifact is None:
        return {
            "exists": False,
            "path": MODEL_PATH,
            "message": "The saved model file could not be read. Retrain the model.",
        }
    return {
        "exists": True,
        "path": MODEL_PATH,
        "best_model": artifact.get("best_model_name"),
        "task_type": artifact.get("task_type"),
        "target": artifact.get("target"),
        "trained_at": artifact.get("trained_at"),
        "model_size_kb": round(os.path.getsize(MODEL_PATH) / 1024, 1),
        "metrics": artifact.get("metrics"),
        "feature_count": len(artifact.get("feature_names", [])),
        "message": "Model artifact available.",
    }


def main():
    """CLI entrypoint used by the Docker image build."""
    print("Training glucose regression models...")
    print("Dataset:", DATA_PATH)
    print("Target :", TARGET)
    result = train_and_save()
    if result.get("error"):
        raise SystemExit("Training failed: {}".format(result["error"]))
    print("Best model:", result["best_model"])
    print("Test metrics:", result["metrics"])
    print("Model saved at:", result["model_path"])
    return result


if __name__ == "__main__":
    main()
