"""
training/train_segmentation.py
Owner: Customer Segmentation

RUN THIS FILE ONCE (locally, or in a notebook/Colab) to train the segmentation
model and save it to models/segmentation/. The frontend/app NEVER runs this
file -- it only loads the .pkl files this script produces, via
modules/segmentation_predictor.py.

Usage:
    python training/train_segmentation.py
    (run from the repo root, so both the data path and output path resolve)

Produces (in models/segmentation/):
    kmeans_model.pkl     - fitted KMeans model
    scaler.pkl           - fitted StandardScaler (needed to transform new inputs)
    cluster_profile.csv  - human-readable summary of each cluster
"""

import os
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
DATA_PATH = "data/Mall_Customers.csv"
K = 5  # chosen via elbow method -- see notebooks/ for the analysis

# This file lives in training/, but output goes to models/segmentation/
# -- computed relative to repo root, not relative to this file, so it's
# unambiguous regardless of where train_segmentation.py itself sits.
SAVE_DIR = os.path.join("models", "segmentation")


def train_and_save():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run this script from the repo root:\n"
            f"    python training/train_segmentation.py"
        )

    os.makedirs(SAVE_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset missing required columns: {missing}")

    print(f"Loaded {len(df)} customer records.")

    # Fit scaler + model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    model = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    df["Cluster"] = labels
    profile = (
        df.groupby("Cluster")
        .agg(
            Mean_Income=(FEATURES[0], "mean"),
            Mean_Spending=(FEATURES[1], "mean"),
            Count=(FEATURES[0], "count"),
        )
        .round(1)
        .reset_index()
    )
    print("\nCluster profile:")
    print(profile)

    # Save trained artifacts -- these are what the app will load, never retrain
    with open(os.path.join(SAVE_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(SAVE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    profile.to_csv(os.path.join(SAVE_DIR, "cluster_profile.csv"), index=False)

    print(f"\nSaved model artifacts to: {SAVE_DIR}")
    print("Training complete. The app will now load these files instead of retraining.")


if __name__ == "__main__":
    train_and_save()
