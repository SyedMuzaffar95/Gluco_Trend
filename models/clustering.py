import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from models.load import load_dataset


PLOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "static", "plots"
)
os.makedirs(PLOT_DIR, exist_ok=True)

CLUSTER_FEATURES = ["glucose", "insulin_bolus", "carbs", "stress_level"]

# Row caps that keep slow O(n^2) algorithms usable inside a Flask request.
HIERARCHICAL_MAX_ROWS = 4000
TSNE_MAX_ROWS = 2000
UMAP_MAX_ROWS = 5000
SILHOUETTE_MAX_ROWS = 5000


def _clean_int(value, default, lo, hi):
    """Return a clamped integer, falling back to default on invalid input."""
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, parsed))


def _clean_float(value, default, lo, hi):
    """Return a clamped float, falling back to default on invalid input."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, parsed))


def features_frame():
    """Return the 4 modelling features (missing values dropped)."""
    df = load_dataset()
    return df[CLUSTER_FEATURES].dropna().copy()


def _filled(df):
    """4-feature frame with missing values filled with column medians."""
    X = df[CLUSTER_FEATURES]
    return X.fillna(X.median())


def _scaled(df):
    """StandardScaler output of the 4 filled features."""
    return StandardScaler().fit_transform(_filled(df))


def _sample_n(df, cap):
    """Random sample capped at `cap` rows; returns (frame, original_rows)."""
    if len(df) > cap:
        return df.sample(n=cap, random_state=42), len(df)
    return df, len(df)


def _plot_labels(coords, labels, title, filename):
    """Scatter plot of 2-D coordinates coloured by cluster labels."""
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                         cmap="viridis", alpha=0.5, s=10)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=140)
    plt.close()
    return filename


def kmeans_model(df):
    from sklearn.cluster import KMeans
    X = df[CLUSTER_FEATURES]
    X = X.fillna(X.median())
    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    model.fit(X)
    labels = model.labels_
    centers = model.cluster_centers_
    print("\n========== K-MEANS CLUSTERING ==========")
    print("\nCluster Labels:")
    print(labels)
    print("\nCluster Centers:")
    print(centers)

    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))
    print("\nCluster Label Counts:")
    print(label_counts)

    return model, labels, centers, label_counts


def dbscan_model(df, eps=0.5, min_samples=5):
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    eps = _clean_float(eps, 0.5, 0.05, 50.0)
    min_samples = _clean_int(min_samples, 5, 2, 200)
    X = df[CLUSTER_FEATURES]
    X = X.fillna(X.median())
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    labels = model.labels_
    unique_clusters = set(labels.tolist())
    n_clusters = len(unique_clusters - {-1})
    n_noise = list(labels).count(-1)
    mask = labels != -1
    if n_clusters >= 2 and sum(mask) > n_clusters:
        score = silhouette_score(X[mask], labels[mask])
    else:
        score = None

    print("\n========== DBSCAN CLUSTERING ==========")
    print("\nCluster Labels:")
    print(labels)
    print("\nNumber of clusters:", n_clusters)
    print("Noise points:", n_noise)
    print("Silhouette Score:", score)

    return model, labels, n_clusters, n_noise, score


def elbow_method(df, max_clusters=10):
    """K-Means inertia (WCSS) for k = 1..max_clusters plus a knee-based k.

    Uses the same median-filled feature space as `kmeans_model` so the elbow
    curve matches the clustering the app already performs.  The recommended
    k is the point with the largest second difference of the inertia curve.
    """
    max_clusters = _clean_int(max_clusters, 10, 2, 10)

    X = _filled(df)
    n_rows = len(X)
    if n_rows < 2:
        return {"error": "Elbow method needs at least 2 samples."}

    k_max = int(min(max_clusters, n_rows))
    cluster_range = list(range(1, k_max + 1))
    inertias = []
    for k in cluster_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias.append(round(float(model.inertia_), 4))

    inertia_drops = [
        round(inertias[i - 1] - inertias[i], 4) for i in range(1, len(inertias))
    ]

    recommended_k = 2
    if len(inertias) >= 3:
        second_diffs = [
            inertia_drops[i - 1] - inertia_drops[i]
            for i in range(1, len(inertia_drops))
        ]
        recommended_k = int(np.argmax(second_diffs)) + 2
    recommended_k = int(max(2, min(recommended_k, k_max)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cluster_range, inertias, marker="o", color="#3498db")
    ax.scatter([recommended_k], [inertias[recommended_k - 1]], color="#e63946",
               s=120, zorder=5,
               label="Recommended k={}".format(recommended_k))
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Elbow Method for K-Means")
    ax.set_xticks(cluster_range)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "elbow_curve.png"), dpi=140)
    plt.close()

    return {
        "error": None,
        "max_clusters": int(k_max),
        "n_used": int(n_rows),
        "n_total": int(n_rows),
        "sampled": False,
        "cluster_range": cluster_range,
        "inertias": inertias,
        "inertia_drops": inertia_drops,
        "recommended_k": recommended_k,
        "plot": "elbow_curve.png",
    }


def silhouette_analysis(df, algorithm="kmeans", n_clusters=3,
                        eps=0.5, min_samples=5):
    """Silhouette score for kmeans / hierarchical / dbscan on scaled features.

    Data is StandardScaler-transformed and capped at SILHOUETTE_MAX_ROWS rows
    because silhouette is O(n^2).  For DBSCAN, noise points labelled -1 are
    excluded and the remaining labels are validated before scoring.  When a
    score cannot be computed the result carries `silhouette: None` and a
    human-readable `reason` instead of raising.
    """
    from sklearn.cluster import AgglomerativeClustering

    if not isinstance(algorithm, str):
        return {
            "error": "algorithm must be a string: kmeans, dbscan or hierarchical.",
        }
    algorithm = algorithm.strip().lower()
    if algorithm in ("agglomerative", "hierarchical-clustering"):
        algorithm = "hierarchical"
    if algorithm not in ("kmeans", "dbscan", "hierarchical"):
        return {
            "error": "Unsupported algorithm '{}'. Use kmeans, dbscan or hierarchical.".format(algorithm),
        }

    sampled_df, total = _sample_n(df, SILHOUETTE_MAX_ROWS)
    X = _scaled(sampled_df)
    n_rows = len(X)
    if n_rows < 2:
        return {"error": "Silhouette analysis needs at least 2 samples."}

    score = None
    reason = None
    n_noise = None
    clusters_found = None

    if algorithm == "dbscan":
        eps = _clean_float(eps, 0.5, 0.05, 50.0)
        min_samples = _clean_int(min_samples, 5, 2, 200)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        mask = labels != -1
        n_noise = int((~mask).sum())
        valid_labels = labels[mask]
        clusters_found = len(set(valid_labels.tolist()))
        if clusters_found < 2:
            reason = (
                "DBSCAN produced {} valid cluster(s) after excluding noise; "
                "silhouette needs at least 2.".format(clusters_found)
            )
        elif len(valid_labels) <= clusters_found:
            reason = "Not enough non-noise points to compute a silhouette score."
        else:
            score = silhouette_score(X[mask], valid_labels)
    else:
        n_clusters = _clean_int(n_clusters, 3, 2, max(2, n_rows - 1))
        n_clusters = int(min(n_clusters, n_rows - 1))
        if algorithm == "kmeans":
            labels = KMeans(n_clusters=n_clusters, random_state=42,
                            n_init=10).fit_predict(X)
        else:
            labels = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward"
            ).fit_predict(X)
        unique = set(labels.tolist())
        clusters_found = len(unique)
        if clusters_found < 2:
            reason = "Only {} cluster(s) found; silhouette needs at least 2.".format(
                clusters_found
            )
        elif n_rows <= clusters_found:
            reason = "Every sample forms its own cluster; silhouette is undefined."
        else:
            score = silhouette_score(X, labels)

    return {
        "error": None,
        "algorithm": algorithm,
        "n_clusters": int(clusters_found if clusters_found is not None else 0),
        "silhouette": round(float(score), 4) if score is not None else None,
        "reason": reason,
        "n_noise": n_noise,
        "n_used": int(n_rows),
        "n_total": int(total),
        "sampled": bool(total > n_rows),
    }


def kmeans_plus_plus_model(df, n_clusters=3):
    """K-Means with explicit init="k-means++" on median-filled features.

    Returns the fitted model together with labels, centres and a PCA
    scatter plot, following the dict convention of `hierarchical_model`.
    """
    X = _filled(df)
    n_rows = len(X)
    if n_rows < 2:
        return {"error": "K-Means++ needs at least 2 samples."}

    n_clusters = _clean_int(n_clusters, 3, 2, max(2, n_rows - 1))
    n_clusters = int(min(n_clusters, n_rows - 1))

    model = KMeans(n_clusters=n_clusters, init="k-means++",
                   random_state=42, n_init=10)
    model.fit(X)
    labels = model.labels_
    centers = model.cluster_centers_

    unique, counts = np.unique(labels, return_counts=True)
    label_counts = {int(k): int(v) for k, v in zip(unique, counts)}

    score = None
    if 2 <= n_clusters < n_rows:
        score = round(float(silhouette_score(X, labels)), 4)

    X_scaled = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2).fit_transform(X_scaled)
    plot = _plot_labels(coords, labels, "K-Means++ Clustering",
                        "kmeans_plus_plus_clusters.png")

    sample_df = X.head(10).copy()
    sample_df.insert(0, "row", range(1, len(sample_df) + 1))
    sample_df["cluster"] = labels[:len(sample_df)]

    return {
        "error": None,
        "n_clusters": int(n_clusters),
        "init": "k-means++",
        "label_counts": label_counts,
        "centers": [[round(float(c), 4) for c in row] for row in centers],
        "silhouette": score,
        "model": model,
        "n_used": int(n_rows),
        "n_total": int(n_rows),
        "sampled": False,
        "sample_html": sample_df.to_html(classes="data-table", index=False),
        "plot": plot,
    }


def minibatch_kmeans_model(df, n_clusters=3, batch_size=1024):
    """MiniBatch K-Means on median-filled features (fast on large datasets).

    Same output structure as `kmeans_plus_plus_model`; batch_size is
    clamped to [16, 8192] and never exceeds the number of rows.
    """
    from sklearn.cluster import MiniBatchKMeans

    X = _filled(df)
    n_rows = len(X)
    if n_rows < 2:
        return {"error": "MiniBatch K-Means needs at least 2 samples."}

    n_clusters = _clean_int(n_clusters, 3, 2, max(2, n_rows - 1))
    n_clusters = int(min(n_clusters, n_rows - 1))
    batch_size = _clean_int(batch_size, 1024, 16, 8192)
    batch_size = int(min(batch_size, n_rows))

    model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size,
                            random_state=42, n_init=10)
    model.fit(X)
    labels = model.labels_
    centers = model.cluster_centers_

    unique, counts = np.unique(labels, return_counts=True)
    label_counts = {int(k): int(v) for k, v in zip(unique, counts)}

    score = None
    if 2 <= n_clusters < n_rows:
        score = round(float(silhouette_score(X, labels)), 4)

    X_scaled = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2).fit_transform(X_scaled)
    plot = _plot_labels(coords, labels, "MiniBatch K-Means Clustering",
                        "minibatch_kmeans_clusters.png")

    sample_df = X.head(10).copy()
    sample_df.insert(0, "row", range(1, len(sample_df) + 1))
    sample_df["cluster"] = labels[:len(sample_df)]

    return {
        "error": None,
        "n_clusters": int(n_clusters),
        "batch_size": int(batch_size),
        "label_counts": label_counts,
        "centers": [[round(float(c), 4) for c in row] for row in centers],
        "silhouette": score,
        "model": model,
        "n_used": int(n_rows),
        "n_total": int(n_rows),
        "sampled": False,
        "sample_html": sample_df.to_html(classes="data-table", index=False),
        "plot": plot,
    }


def pca_model(df, n_components=None):
    from sklearn.preprocessing import StandardScaler as SKLearnStandardScaler
    from sklearn.decomposition import PCA as SKLearnPCA
    if n_components is None:
        n_components = min(3, len(CLUSTER_FEATURES))
    n_components = _clean_int(n_components, min(3, len(CLUSTER_FEATURES)),
                              1, len(CLUSTER_FEATURES))
    X = df[CLUSTER_FEATURES].dropna()
    scaler = SKLearnStandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = SKLearnPCA(n_components=n_components)
    pca.fit(X_scaled)

    pca_explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    pca_components = [
        {"feature": feat, "values": [round(float(c), 4) for c in comp]}
        for feat, comp in zip(CLUSTER_FEATURES, pca.components_)
    ]

    print("\n========== PCA ==========")
    print("\nExplained Variance Ratio:")
    print(pca_explained)
    print("\nComponent Loadings:")
    for item in pca_components:
        print("  {}: {}".format(item["feature"], item["values"]))

    return pca, pca_explained, pca_components


def _generate_clustering_plots(kmeans_labels, kmeans_centers, dbscan_labels, pca_explained):
    df = load_dataset()
    X = df[CLUSTER_FEATURES].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels[:len(X_2d)], cmap="viridis", alpha=0.5, s=10)
    centers_2d = pca_2d.transform(kmeans_centers)
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c="red", marker="X", s=200, label="Centers")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-Means Clustering (2 clusters)")
    ax.legend()
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()
    kmeans_path = os.path.join(PLOT_DIR, "kmeans_clusters.png")
    plt.savefig(kmeans_path, dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    dl = dbscan_labels[:len(X_2d)]
    mask = dl != -1
    ax.scatter(X_2d[~mask, 0], X_2d[~mask, 1], c="gray", alpha=0.3, s=10, label="Noise")
    if mask.any():
        scatter2 = ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=dl[mask], cmap="viridis", alpha=0.5, s=10, label="Clusters")
        plt.colorbar(scatter2, ax=ax)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("DBSCAN Clustering")
    ax.legend()
    plt.tight_layout()
    dbscan_path = os.path.join(PLOT_DIR, "dbscan_clusters.png")
    plt.savefig(dbscan_path, dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    components = ["PC{}".format(i + 1) for i in range(len(pca_explained))]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#9b59b6"][:len(pca_explained)]
    bars = ax.bar(components, pca_explained, color=colors)
    for bar, val in zip(bars, pca_explained):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, "{:.4f}".format(val), ha="center", fontsize=10)
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA - Explained Variance per Component")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    pca_path = os.path.join(PLOT_DIR, "pca_variance.png")
    plt.savefig(pca_path, dpi=140)
    plt.close()

    return "kmeans_clusters.png", "dbscan_clusters.png", "pca_variance.png"


def unsupervised_summary(dbscan_eps=0.5, dbscan_min_samples=5,
                         pca_n_components=None):
    df = load_dataset()
    data = df[CLUSTER_FEATURES].dropna().copy()

    model_k, labels_k, centers_k, label_counts = kmeans_model(data)
    model_d, labels_d, n_clusters, n_noise, silhouette = dbscan_model(
        data, eps=dbscan_eps, min_samples=dbscan_min_samples
    )
    pca_obj, pca_explained, pca_components = pca_model(
        data, n_components=pca_n_components
    )
    pca_cumulative = [round(float(v), 4) for v in np.cumsum(pca_explained)]

    plots = _generate_clustering_plots(labels_k, centers_k, labels_d, pca_explained)

    return {
        "features": CLUSTER_FEATURES,
        "kmeans_label_counts": label_counts,
        "kmeans_centers": [[round(float(c), 4) for c in row] for row in centers_k],
        "dbscan_n_clusters": n_clusters,
        "dbscan_n_noise": n_noise,
        "dbscan_silhouette": silhouette,
        "dbscan_eps": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
        "pca_explained": pca_explained,
        "pca_cumulative": pca_cumulative,
        "pca_n_components": len(pca_explained),
        "pca_components": pca_components,
        "n_rows": len(data),
        "kmeans_plot": plots[0],
        "dbscan_plot": plots[1],
        "pca_plot": plots[2],
    }


def hierarchical_model(df, n_clusters=3, linkage="ward"):
    """Agglomerative (hierarchical) clustering on scaled features."""
    from sklearn.cluster import AgglomerativeClustering

    n_clusters = _clean_int(n_clusters, 3, 2, 10)
    if linkage not in ("ward", "complete", "average", "single"):
        linkage = "ward"

    sampled_df, total = _sample_n(df, HIERARCHICAL_MAX_ROWS)
    X_scaled = _scaled(sampled_df)

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X_scaled)

    unique, counts = np.unique(labels, return_counts=True)
    label_counts = {int(k): int(v) for k, v in zip(unique, counts)}

    score = None
    if 2 <= n_clusters < len(X_scaled):
        score = silhouette_score(X_scaled, labels)

    coords = PCA(n_components=2).fit_transform(X_scaled)
    plot = _plot_labels(coords, labels,
                        "Hierarchical Clustering",
                        "hierarchical_clusters.png")

    sample_df = _filled(sampled_df).head(10).copy()
    sample_df.insert(0, "row", range(1, len(sample_df) + 1))
    sample_df["cluster"] = labels[:len(sample_df)]

    return {
        "error": None,
        "n_clusters": n_clusters,
        "linkage": linkage,
        "label_counts": label_counts,
        "silhouette": round(float(score), 4) if score is not None else None,
        "n_used": int(len(labels)),
        "n_total": int(total),
        "sampled": bool(total > len(labels)),
        "sample_html": sample_df.to_html(classes="data-table", index=False),
        "plot": plot,
    }


def tsne_model(df, n_components=2, perplexity=30):
    """t-SNE projection on scaled features (sampled for speed)."""
    from sklearn.manifold import TSNE

    n_components = _clean_int(n_components, 2, 2, 3)
    perplexity = _clean_float(perplexity, 30, 5, 100)

    sampled_df, total = _sample_n(df, TSNE_MAX_ROWS)
    X_scaled = _scaled(sampled_df)

    # perplexity must stay well below the number of samples
    max_perplexity = max(5.0, (len(X_scaled) - 1) / 3.0)
    perplexity = min(perplexity, max_perplexity)

    embedding = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        init="pca",
    ).fit_transform(X_scaled)

    color_labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(
        X_scaled
    )
    plot = _plot_labels(embedding[:, :2], color_labels,
                        "t-SNE Projection (coloured by K-Means cluster)",
                        "tsne_projection.png")

    sample_df = _filled(sampled_df).head(10).copy()
    sample_df.insert(0, "row", range(1, len(sample_df) + 1))
    for dim in range(n_components):
        sample_df["t-SNE {}".format(dim + 1)] = np.round(embedding[:10, dim], 4)

    return {
        "error": None,
        "n_components": n_components,
        "perplexity": round(float(perplexity), 2),
        "n_used": int(len(embedding)),
        "n_total": int(total),
        "sampled": bool(total > len(embedding)),
        "original_dims": [int(total), len(CLUSTER_FEATURES)],
        "sample_html": sample_df.to_html(classes="data-table", index=False),
        "plot": plot,
    }


def umap_model(df, n_components=2, n_neighbors=15, min_dist=0.1):
    """UMAP projection on scaled features (umap-learn package)."""
    try:
        import umap
    except ImportError:
        return {
            "error": "UMAP is not installed. Run: pip install umap-learn",
        }

    n_components = _clean_int(n_components, 2, 2, 3)
    n_neighbors = _clean_int(n_neighbors, 15, 2, 100)
    min_dist = _clean_float(min_dist, 0.1, 0.0, 0.99)

    sampled_df, total = _sample_n(df, UMAP_MAX_ROWS)
    X_scaled = _scaled(sampled_df)
    n_neighbors = min(n_neighbors, max(2, len(X_scaled) - 1))

    embedding = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    ).fit_transform(X_scaled)

    color_labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(
        X_scaled
    )
    plot = _plot_labels(embedding[:, :2], color_labels,
                        "UMAP Projection (coloured by K-Means cluster)",
                        "umap_projection.png")

    sample_df = _filled(sampled_df).head(10).copy()
    sample_df.insert(0, "row", range(1, len(sample_df) + 1))
    for dim in range(n_components):
        sample_df["UMAP {}".format(dim + 1)] = np.round(embedding[:10, dim], 4)

    return {
        "error": None,
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "n_used": int(len(embedding)),
        "n_total": int(total),
        "sampled": bool(total > len(embedding)),
        "sample_html": sample_df.to_html(classes="data-table", index=False),
        "plot": plot,
    }


def _anomaly_summary(df, status_column, limit=50):
    """Status counts + first `limit` anomaly records for the results tables."""
    n_normal = int((df[status_column] == "Normal").sum())
    n_anomaly = int((df[status_column] == "Anomaly").sum())
    total = n_normal + n_anomaly
    anomaly_pct = round(100.0 * n_anomaly / total, 2) if total else 0.0

    cols = [c for c in ["user_id"] + CLUSTER_FEATURES if c in df.columns]
    records = df.loc[df[status_column] == "Anomaly", cols + [status_column]]
    records = records.head(limit)
    records_html = records.rename(columns={status_column: "Status"}).to_html(
        classes="data-table", index=False
    )

    return {
        "total": total,
        "n_normal": n_normal,
        "n_anomaly": n_anomaly,
        "anomaly_pct": anomaly_pct,
        "status_counts": [("Normal", n_normal), ("Anomaly", n_anomaly)],
        "records_shown": int(len(records)),
        "records_html": records_html,
    }


def isolation_forest_model(df):
    """Isolation Forest with fixed parameters (1 -> Normal, -1 -> Anomaly).

    Runs on the full dataset; no user parameters.
    """
    from sklearn.ensemble import IsolationForest

    features = df[CLUSTER_FEATURES].fillna(df[CLUSTER_FEATURES].median())

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
    )
    preds = model.fit_predict(features)

    # Store the result as a new column on the in-memory frame.
    df["IsolationForest_Status"] = np.where(preds == 1, "Normal", "Anomaly")

    summary = _anomaly_summary(df, "IsolationForest_Status", limit=50)
    summary["error"] = None
    summary["algorithm"] = "Isolation Forest"
    summary["params"] = "n_estimators=100, contamination=0.05, random_state=42"
    return summary


def one_class_svm_model(df):
    """One-Class SVM with fixed parameters (1 -> Normal, -1 -> Anomaly).

    StandardScaler is applied before training; runs on the full dataset.
    """
    from sklearn.svm import OneClassSVM

    features = df[CLUSTER_FEATURES].fillna(df[CLUSTER_FEATURES].median())
    X_scaled = StandardScaler().fit_transform(features)

    model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    model.fit(X_scaled)
    preds = model.predict(X_scaled)

    # Store the result as a new column on the in-memory frame.
    df["OneClassSVM_Status"] = np.where(preds == 1, "Normal", "Anomaly")

    summary = _anomaly_summary(df, "OneClassSVM_Status", limit=50)
    summary["error"] = None
    summary["algorithm"] = "One-Class SVM"
    summary["params"] = "kernel='rbf', nu=0.05, gamma='scale' (StandardScaler applied)"
    return summary


def _plot_autoencoder_error(errors, threshold, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(errors, bins=60, color="#3498db", alpha=0.85)
    ax.axvline(threshold, color="#e63946", linestyle="--", linewidth=1.5,
               label="Threshold (95th percentile)")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Sample Count")
    ax.set_title("Autoencoder Reconstruction Error Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=140)
    plt.close()
    return filename


def autoencoder_model(df, encoding_dim=2):
    """Shallow neural autoencoder built with sklearn's MLPRegressor.

    sklearn has no dedicated neural Autoencoder class and this project does
    not use TensorFlow/Keras/PyTorch, so the autoencoder is an MLPRegressor
    trained to reconstruct its own input (X -> X).  The middle hidden layer
    is the encoding; the reconstruction error is the anomaly score.
    """
    from sklearn.neural_network import MLPRegressor

    encoding_dim = _clean_int(encoding_dim, 2, 1, 3)

    X_scaled = _scaled(df)
    hidden = (8, encoding_dim, 8)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=400,
        random_state=42,
    )
    mlp.fit(X_scaled, X_scaled)

    # Manual forward pass so the bottleneck activation can be extracted.
    activations = [X_scaled]
    activation = X_scaled
    for i, (coefs, intercepts) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        activation = activation @ coefs + intercepts
        if i < len(mlp.coefs_) - 1:
            activation = np.maximum(activation, 0.0)  # ReLU hidden layers
        activations.append(activation)

    encoding = activations[2]        # middle hidden layer = bottleneck
    reconstruction = activations[-1]  # identity output layer

    errors = np.mean((X_scaled - reconstruction) ** 2, axis=1)
    threshold = float(np.percentile(errors, 95))
    anomalies = errors > threshold

    n_anomaly = int(anomalies.sum())
    anomaly_pct = round(100.0 * n_anomaly / len(errors), 2)

    plot = _plot_autoencoder_error(errors, threshold, "autoencoder_error.png")

    sample_df = _filled(df).head(10).copy()
    sample_df.insert(0, "row", range(1, len(sample_df) + 1))
    for dim in range(encoding.shape[1]):
        sample_df["enc {}".format(dim + 1)] = np.round(encoding[:10, dim], 4)
    sample_df["error"] = np.round(errors[:10], 6)
    sample_df["status"] = np.where(anomalies[:10], "Anomaly", "Normal")

    return {
        "error": None,
        "encoding_dim": int(encoding_dim),
        "hidden_layers": "8 - {} - 8".format(encoding_dim),
        "total": int(len(errors)),
        "mean_error": round(float(errors.mean()), 6),
        "median_error": round(float(np.median(errors)), 6),
        "max_error": round(float(errors.max()), 6),
        "threshold": round(threshold, 6),
        "n_anomaly": n_anomaly,
        "anomaly_pct": anomaly_pct,
        "sample_html": sample_df.to_html(classes="data-table", index=False),
        "plot": plot,
    }


def augment_data(df, n_samples=500, random_state=42):
    """Create synthetic rows in memory; the original dataset is untouched.

    Strategy: sample existing rows, add Gaussian noise (5% of each column's
    standard deviation) to numeric columns, then clip to the original
    column range so no unrealistic values appear.  Identifier columns are
    left unchanged and categorical values are copied as-is.
    """
    n_samples = _clean_int(n_samples, 500, 10, 10000)
    random_state = _clean_int(random_state, 42, 0, 10 ** 6)

    original_size = len(df)
    rng = np.random.RandomState(random_state)
    idx = rng.randint(0, original_size, size=n_samples)
    synthetic = df.iloc[idx].reset_index(drop=True).copy()

    id_columns = {"user_id", "device_id"}
    numeric_cols = [
        col for col in synthetic.select_dtypes(include=[np.number]).columns
        if col not in id_columns
    ]

    for col in numeric_cols:
        series = df[col]
        std = float(series.std())
        if np.isnan(std) or std == 0:
            continue
        noise = rng.normal(0.0, 0.05 * std, size=n_samples)
        noisy = (synthetic[col].astype(float) + noise).clip(
            series.min(), series.max()
        )
        if pd.api.types.is_integer_dtype(series.dtype):
            synthetic[col] = noisy.round().astype(series.dtype)
        else:
            synthetic[col] = noisy.round(2).astype(series.dtype)

    return {
        "error": None,
        "original_size": int(original_size),
        "generated": int(n_samples),
        "final_size": int(original_size + n_samples),
        "random_state": int(random_state),
        "noised_columns": len(numeric_cols),
        "sample_html": synthetic.head(10).to_html(classes="data-table", index=False),
        "note": (
            "Synthetic rows are generated in memory only and never written "
            "to disk. The original dataset stays unchanged."
        ),
    }


if __name__ == "__main__":
    df = load_dataset()
    data = df[CLUSTER_FEATURES].dropna().copy()
    print("Dataset Shape:", data.shape)
    print("\nFirst 5 Rows:")
    print(data.head())

    m1, l1, c1, lc1 = kmeans_model(data)
    print("=" * 100)
    m2, l2, nc, nn, ss = dbscan_model(data)
    print("=" * 100)
    m3, pe, pc = pca_model(data)
    print("=" * 100)