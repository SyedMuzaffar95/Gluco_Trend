"""Prediction workflow driven by the saved model artifact.

The input form is generated from the metadata stored with the trained
pipeline (feature order, numeric ranges, categorical choices), so the
prediction always uses exactly the features the model was trained on.
"""

import math
import numbers

import numpy as np
import pandas as pd

from models.train_model import TARGET, artifact_status, load_artifact

DISCLAIMER = (
    "This is an experimental machine-learning output computed from the "
    "GlucoBench dataset. It is not a medical diagnosis and is not a "
    "substitute for professional medical advice, testing or treatment."
)

# Features computed from simpler form inputs instead of being asked directly.
_DERIVED = {"is_weekend", "hour_sin", "hour_cos"}


def _group_for(name):
    """Rough form grouping for readable layout (display only)."""
    if name.startswith("glucose_lag"):
        return "Glucose History"
    if any(k in name for k in ("insulin", "carb", "meal", "medication", "alcohol")):
        return "Insulin & Meals"
    if any(k in name for k in ("exercise", "heart", "skin", "gsr",
                               "stress", "sleep", "cgm")):
        return "Activity & Vitals"
    if any(k in name for k in ("age", "sex", "weight", "hba1c",
                               "region", "timezone", "ratio", "sensitivity")):
        return "Demographics"
    if name in ("hour", "minute", "day_of_week", "month"):
        return "Time"
    return "Other"


def _label_for(name):
    return name.replace("_", " ").title()


def form_spec():
    """Build the dynamic prediction form specification from the artifact."""
    status = artifact_status()
    if not status.get("exists"):
        return {
            "error": status.get(
                "message",
                "No trained model is available. Train a model first.",
            ),
            "status": status,
        }

    artifact = load_artifact()
    if artifact is None:
        return {
            "error": "The saved model could not be loaded. Please retrain.",
            "status": status,
        }

    field_stats = artifact.get("field_stats", {})
    categories = artifact.get("categories", {})
    fields = []
    for col in artifact.get("feature_names", []):
        if col in _DERIVED:
            continue
        if col in artifact.get("numeric_cols", []):
            stats = field_stats.get(col, {})
            fields.append({
                "name": col,
                "label": _label_for(col),
                "type": "number",
                "group": _group_for(col),
                "min": stats.get("min"),
                "max": stats.get("max"),
                "default": stats.get("median"),
                "required": True,
            })
        else:
            choices = categories.get(col, [])
            fields.append({
                "name": col,
                "label": _label_for(col),
                "type": "select",
                "group": _group_for(col),
                "choices": choices,
                "default": choices[0] if choices else None,
                "required": True,
            })

    groups = []
    for field in fields:
        if field["group"] not in groups:
            groups.append(field["group"])

    return {
        "error": None,
        "status": status,
        "best_model": artifact.get("best_model_name"),
        "model_name": artifact.get("model_name"),
        "target": artifact.get("target", TARGET),
        "fields": fields,
        "groups": groups,
        "disclaimer": DISCLAIMER,
    }


def _validate_number(name, raw, stats):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, "'{}' must be a number.".format(_label_for(name))
    if math.isnan(value) or math.isinf(value):
        return None, "'{}' must be a finite number.".format(_label_for(name))
    if stats:
        lo, hi = stats.get("min"), stats.get("max")
        span = (hi - lo) if (lo is not None and hi is not None) else 0.0
        slack = max(1.0, 0.2 * span) if span else 1.0
        if lo is not None and value < lo - slack:
            return None, "'{}' is below the dataset minimum ({}).".format(
                _label_for(name), lo
            )
        if hi is not None and value > hi + slack:
            return None, "'{}' is above the dataset maximum ({}).".format(
                _label_for(name), hi
            )
    return value, None


def predict_from_form(form):
    """Validate the submitted form and return a glucose prediction."""
    artifact = load_artifact()
    if artifact is None:
        return {
            "error": (
                "No trained model is available. Train a model on the "
                "Train Model page first."
            ),
        }

    feature_names = artifact["feature_names"]
    numeric_cols = set(artifact.get("numeric_cols", []))
    categories = artifact.get("categories", {})
    field_stats = artifact.get("field_stats", {})

    values = {}
    errors = []
    for col in feature_names:
        if col in _DERIVED:
            continue
        raw = (form.get(col) or "").strip()
        if raw == "":
            errors.append("'{}' is required.".format(_label_for(col)))
            continue
        if col in numeric_cols:
            value, error = _validate_number(col, raw, field_stats.get(col, {}))
        else:
            allowed = categories.get(col, [])
            value = raw
            error = None
            if allowed and raw not in allowed:
                error = "'{}' must be one of: {}.".format(
                    _label_for(col), ", ".join(allowed)
                )
        if error:
            errors.append(error)
        else:
            values[col] = value

    if errors:
        return {"error": None, "validation_errors": errors}

    # Derive engineered time features from the simpler form inputs.
    hour = values.get("hour")
    day_of_week = values.get("day_of_week")
    if hour is not None:
        values["hour_sin"] = round(math.sin(2 * math.pi * hour / 24), 6)
        values["hour_cos"] = round(math.cos(2 * math.pi * hour / 24), 6)
    if day_of_week is not None:
        values["is_weekend"] = 1 if day_of_week >= 5 else 0

    row = pd.DataFrame(
        [[values.get(col, np.nan) for col in feature_names]],
        columns=feature_names,
    )

    try:
        prediction = artifact["model"].predict(row)
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        return {"error": "Prediction failed: {}".format(exc)}

    predicted = float(prediction[0])
    result = {
        "error": None,
        "validation_errors": [],
        "prediction": round(predicted, 2),
        "target": artifact.get("target", TARGET),
        "model_name": artifact.get("best_model_name"),
        "task_type": artifact.get("task_type"),
        "disclaimer": DISCLAIMER,
    }

    # A single regression point has no classification confidence; show how
    # the value sits relative to common glucose reference zones instead.
    if artifact.get("task_type") == "regression":
        if predicted < 70:
            zone = "below the usual reference range"
        elif predicted <= 140:
            zone = "within the usual reference range"
        else:
            zone = "above the usual reference range"
        result["zone"] = zone
        result["note"] = (
            "No classification confidence is reported because this is a "
            "regression model producing a continuous glucose estimate."
        )
    return result
