import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "glucose"

DROP_COLUMNS = [
    TARGET,
    "timestamp",
    "user_id",
    "device_id",
    "notes",
    # This feature contains the current glucose value in its rolling window,
    # so it would leak the target during prediction.
    "glucose_roll_mean_1h",
]


def prepare_features(df: pd.DataFrame):
    """Create leakage-safe model features from the raw GlucoBench data."""
    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], dayfirst=True, errors="coerce"
    )

    # Time features
    data["hour"] = data["timestamp"].dt.hour
    data["minute"] = data["timestamp"].dt.minute
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    # Cyclic time representation
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

    X = data.drop(columns=DROP_COLUMNS, errors="ignore")
    y = data[TARGET].copy()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numeric_features = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            # Scaling makes KNN distance meaningful across different units.
            ("scaler", StandardScaler()),
            ("knn_imputer", KNNImputer(
                n_neighbors=5,
                weights="distance"
            )),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return X, y, preprocessor, numeric_features, categorical_features


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact missing-value report."""
    result = df.isna().sum().rename("missing").to_frame()
    result["percent"] = (result["missing"] / len(df) * 100).round(2)
    return result[result["missing"] > 0].sort_values("missing", ascending=False)
