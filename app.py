import os

from flask import Flask, render_template, request

from models.load import load_dataset
from models.eda import (
    dataset_info,
    dataset_description,
    missing_values,
    duplicate_values,
    glucose_statistics,
    glucose_distribution,
    glucose_boxplot,
    correlation_heatmap,
)
from models.preprocessing import missing_report, prepare_features
from models.regression import (
    simple_linear_regression,
    simple_logistic_regression,
    ensemble_summary,
    regression_summary,
)
from models.clustering import (
    unsupervised_summary,
    features_frame,
    hierarchical_model,
    tsne_model,
    umap_model,
    isolation_forest_model,
    one_class_svm_model,
    autoencoder_model,
    augment_data,
    elbow_method,
    silhouette_analysis,
    kmeans_plus_plus_model,
    minibatch_kmeans_model,
)
from models.preprocessing import TARGET
from models.train_model import (
    train_all,
    train_and_save,
    artifact_status,
)
from models.predict import form_spec, predict_from_form


app = Flask(__name__)


@app.route("/")
def index():
    """Homepage: dataset overview fed from the real backend data."""
    from models.load import DATASET_PATH
    from models.train_model import artifact_status

    overview = {
        "loaded": False,
        "dataset_name": os.path.basename(DATASET_PATH),
        "error": None,
    }
    try:
        df = load_dataset()
        overview.update({
            "loaded": True,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "feature_count": int(df.shape[1]) - 1,
            "target": TARGET,
            "task_type": "regression",
        })
    except Exception as exc:  # noqa: BLE001 - show empty state, never crash
        overview["error"] = "Dataset could not be loaded: {}".format(exc)

    return render_template(
        "index.html",
        overview=overview,
        artifact=artifact_status(),
    )


@app.route("/load")
def load():
    df = load_dataset()
    return render_template(
        "load.html",
        data=df.head().to_html(classes="data-table", index=False),
        rows=df.shape[0],
        columns=df.shape[1],
    )


@app.route("/eda")
def eda():
    df = load_dataset()
    description = dataset_description(df)
    missing = missing_values(df)
    correlation, heatmap_image = correlation_heatmap(df)

    return render_template(
        "eda.html",
        info=dataset_info(df),
        description=description.to_html(classes="data-table"),
        missing=missing[missing > 0].to_frame("Missing").to_html(
            classes="data-table"
        ),
        duplicates=duplicate_values(df),
        glucose_stats=glucose_statistics(df),
        distribution_image=glucose_distribution(df),
        boxplot_image=glucose_boxplot(df),
        correlation=correlation.to_html(classes="data-table", float_format=lambda x: f"{x:.2f}"),
        heatmap_image=heatmap_image,
    )


@app.route("/preprocessing")
def preprocessing():
    df = load_dataset()
    report = missing_report(df)
    X, y, _, numeric, categorical = prepare_features(df)

    before_missing = int(df.isna().sum().sum())
    after_knn = 0  # KNN is fitted inside the training pipeline; no leakage.

    return render_template(
        "preprocessing.html",
        rows=len(df),
        columns=len(df.columns),
        before_missing=before_missing,
        report=report.to_html(classes="data-table"),
        numeric_count=len(numeric),
        categorical_count=len(categorical),
        model_features=X.shape[1],
        after_knn=after_knn,
    )


@app.route("/simple-regression")
def simple_regression():
    slr_metrics, slr_results, slope, intercept, slr_plot = simple_linear_regression()
    log_metrics, confusion, log_results, log_plot = simple_logistic_regression()

    return render_template(
        "simple_regression.html",
        slr_metrics=slr_metrics,
        slr_results=slr_results.head(15).to_html(classes="data-table", index=False),
        slope=round(slope, 4),
        intercept=round(intercept, 4),
        slr_plot_image=slr_plot,
        log_metrics=log_metrics,
        confusion=confusion,
        log_results=log_results.head(15).to_html(classes="data-table", index=False),
        log_plot_image=log_plot,
    )


@app.route("/ensemble")
def ensemble():
    summary = ensemble_summary()
    comparison_df = summary.pop("comparison")

    return render_template(
        "ensemble.html",
        comparison=comparison_df.to_html(classes="data-table", index=False),
        **summary,
    )


def _run_method(fn, *args, **kwargs):
    """Run one unsupervised method; convert failures into an error dict."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - never expose stack traces
        return {"error": "Could not run this method: {}".format(exc)}


@app.route("/unsupervised")
def unsupervised():
    summary = unsupervised_summary()
    data = features_frame()

    summary["hier_result"] = _run_method(hierarchical_model, data)
    summary["tsne_result"] = _run_method(tsne_model, data)
    summary["umap_result"] = _run_method(umap_model, data)
    summary["if_result"] = _run_method(isolation_forest_model, load_dataset())
    summary["ocsvm_result"] = _run_method(one_class_svm_model, load_dataset())
    summary["ae_result"] = _run_method(autoencoder_model, data)
    summary["aug_result"] = _run_method(augment_data, load_dataset())
    summary["elbow_result"] = _run_method(elbow_method, data)
    summary["kpp_result"] = _run_method(kmeans_plus_plus_model, data)
    summary["mbkm_result"] = _run_method(minibatch_kmeans_model, data)
    summary["sil_result"] = _run_method(silhouette_analysis, data)

    return render_template("unsupervised.html", **summary)


@app.route("/isolation-forest")
def isolation_forest_page():
    return render_template(
        "isolation_forest.html",
        **_run_method(isolation_forest_model, load_dataset()),
    )


@app.route("/one-class-svm")
def one_class_svm_page():
    return render_template(
        "one_class_svm.html",
        **_run_method(one_class_svm_model, load_dataset()),
    )


@app.route("/dashboard")
def dashboard():
    df = load_dataset()
    status = artifact_status()
    missing_total = int(df.isna().sum().sum())
    return render_template(
        "dashboard.html",
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        column_names=list(df.columns),
        target=TARGET,
        task_type="regression",
        missing_total=missing_total,
        duplicates=int(df.duplicated().sum()),
        numeric_count=int(df.select_dtypes(include=["number"]).shape[1]),
        categorical_count=int(df.select_dtypes(exclude=["number"]).shape[1]),
        artifact=status,
    )


@app.route("/regression")
def regression_page():
    return render_template(
        "regression.html",
        **_run_method(regression_summary),
    )


@app.route("/model-comparison")
def model_comparison_page():
    result = _run_method(train_all, use_cache=True)
    if not result.get("error"):
        result = {
            k: v for k, v in result.items() if not k.startswith("_")
        }
    return render_template("model_comparison.html", **result)


@app.route("/train", methods=["GET", "POST"])
def train_page():
    result = None
    if request.method == "POST":
        result = _run_method(train_and_save)
    return render_template(
        "train_model.html",
        result=result,
        status=artifact_status(),
    )


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    spec = form_spec()
    prediction = None
    if request.method == "POST":
        if spec.get("error"):
            prediction = {"error": spec["error"], "validation_errors": []}
        else:
            prediction = predict_from_form(request.form)
    return render_template(
        "predict.html",
        spec=spec,
        prediction=prediction,
    )


if __name__ == "__main__":
    # Run Flask only; EDA plots are generated when /eda is opened.
    # Port 5001 keeps this project separate from other apps using 5000.
    # Inside Docker, PORT=5001 and the app binds 0.0.0.0 so the
    # published port reaches the container.
    port = int(os.environ.get("PORT", "5001"))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    app.run(debug=True, use_reloader=False, port=port, host=host)
