"""
segmentation.py
Production-ready Customer Segmentation module for SuperMart Intelligence Suite (SIS)

Implements:
- Dataset loading and validation
- Dataset preview & summary statistics
- Feature scaling (StandardScaler)
- Elbow Method
- K-Means Clustering & Profiling
- Hierarchical Clustering & Dendrogram
- Hardcoded test customer prediction (No interactive inputs)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram


FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]


def load_data(path: str) -> pd.DataFrame:
    """Loads and validates dataset from a CSV path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at path: {path}")

    df = pd.read_csv(path)
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Dataset is missing required features: {missing_cols}")

    print(f"Successfully loaded {len(df)} customer records.")
    return df


def show_dataset_preview(df: pd.DataFrame) -> None:
    """Prints basic summary and health check of the dataset."""
    print("\n===== Dataset Preview =====")
    print(df.head())
    print("\n===== Dataset Info =====")
    print(df.info())
    print("\n===== Missing Values =====")
    print(df[FEATURES].isnull().sum())


def summary_statistics(df: pd.DataFrame) -> None:
    """Displays statistical highlights for defined features."""
    print("\n===== Summary Statistics =====")
    print(f"Total Customers : {len(df)}")
    print(f"Average Income  : ${df[FEATURES[0]].mean():.2f}k")
    print(f"Average Spending: {df[FEATURES[1]].mean():.2f}")


def elbow_method(X_scaled: pd.DataFrame):
    """Calculates WCSS across k=1..10 using scaled features. Returns (wcss, fig)."""
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 11), wcss, marker="o")
    ax.set_title("Elbow Method (Scaled Features)")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.grid(True)
    fig.tight_layout()

    return wcss, fig


def run_kmeans(df: pd.DataFrame, X_scaled: pd.DataFrame, k: int = 5):
    """Executes K-Means clustering and generates profile summaries."""
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    result = df.copy()
    result["Cluster"] = labels

    profile = (
        result.groupby("Cluster")
        .agg(
            Mean_Income=(FEATURES[0], "mean"),
            Mean_Spending=(FEATURES[1], "mean"),
            Count=(FEATURES[0], "count"),
        )
        .round(1)
        .reset_index()
    )

    print("\nK-Means Cluster Profile:")
    print(profile)

    return result, model, profile


def plot_kmeans(df: pd.DataFrame, model: KMeans, scaler: StandardScaler):
    """Plots K-Means clusters + inverted centroids back in original scale. Returns fig."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        df[FEATURES[0]], df[FEATURES[1]], c=df["Cluster"], cmap="viridis", alpha=0.7
    )

    centroids_original = scaler.inverse_transform(model.cluster_centers_)
    ax.scatter(
        centroids_original[:, 0],
        centroids_original[:, 1],
        marker="X",
        s=220,
        c="red",
        label="Centroids",
    )

    ax.set_title("K-Means Customer Segments")
    ax.set_xlabel(FEATURES[0])
    ax.set_ylabel(FEATURES[1])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def run_hierarchical(df: pd.DataFrame, X_scaled: pd.DataFrame, n_clusters: int = 5):
    """Runs Agglomerative Hierarchical Clustering on scaled data."""
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)

    result = df.copy()
    result["HCluster"] = labels

    profile = (
        result.groupby("HCluster")
        .agg(
            Mean_Income=(FEATURES[0], "mean"),
            Mean_Spending=(FEATURES[1], "mean"),
            Count=(FEATURES[0], "count"),
        )
        .round(1)
        .reset_index()
    )

    print("\nHierarchical Cluster Profile:")
    print(profile)

    return result, profile


def plot_dendrogram(X_scaled: pd.DataFrame):
    """Generates Ward linkage dendrogram from scaled features. Returns fig."""
    Z = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, ax=ax)
    ax.set_title("Customer Hierarchical Dendrogram")
    ax.set_xlabel("Customer Indices")
    ax.set_ylabel("Euclidean Distance")
    fig.tight_layout()
    return fig


def predict_customer(
    income: float, spending: float, model: KMeans, scaler: StandardScaler
) -> int:
    """Scales input features and predicts cluster assignment for a single customer."""
    input_df = pd.DataFrame([[income, spending]], columns=FEATURES)
    input_scaled = scaler.transform(input_df)
    cluster = model.predict(input_scaled)[0]
    return int(cluster)


def main():
    DATA_PATH = r"C:\Users\sarva\Downloads\Mall_Customers.csv"
    TEST_INCOME = 70.0
    TEST_SPENDING = 78.0

    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    show_dataset_preview(df)
    summary_statistics(df)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df[FEATURES]), columns=FEATURES
    )

    _, elbow_fig = elbow_method(X_scaled)
    # or plt.show() if running interactively
    elbow_fig.savefig("elbow_output.png")

    km_df, km_model, _ = run_kmeans(df, X_scaled, k=5)
    kmeans_fig = plot_kmeans(km_df, km_model, scaler)
    kmeans_fig.savefig("kmeans_output.png")

    run_hierarchical(df, X_scaled, n_clusters=5)
    dendro_fig = plot_dendrogram(X_scaled)
    dendro_fig.savefig("dendrogram_output.png")

    print("\n===== Predict New Customer =====")
    print(
        f"Testing for -> Income: ${TEST_INCOME}k, Spending Score: {TEST_SPENDING}")
    predicted_cluster = predict_customer(
        TEST_INCOME, TEST_SPENDING, km_model, scaler
    )
    print(f"Predicted Cluster Segment: {predicted_cluster}")

    print("\nPipeline executed successfully.")


if __name__ == "__main__":
    main()
