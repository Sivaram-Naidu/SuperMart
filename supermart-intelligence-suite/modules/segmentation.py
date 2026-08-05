"""
segmentation_predictor.py
Owner: Customer Segmentation

This is the file to hand off to the Integration/Frontend Owner.

It loads the ALREADY-TRAINED model from models/segmentation/*.pkl and
exposes simple prediction functions. It does NOT train anything -- if the
.pkl files are missing, it raises a clear error telling you to run
train_model.py first.

The model is loaded lazily and cached at module level, so importing this
file or calling its functions repeatedly (e.g. on every Streamlit rerun)
does NOT reload from disk each time -- only the first call touches disk.
"""

import os
import pickle
from typing import Any, Optional
import pandas as pd

FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]

_MODEL_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "models", "segmentation")
_MODEL_DIR = os.path.normpath(_MODEL_DIR)

_MODEL_PATH = os.path.join(_MODEL_DIR, "kmeans_model.pkl")
_SCALER_PATH = os.path.join(_MODEL_DIR, "scaler.pkl")
_PROFILE_PATH = os.path.join(_MODEL_DIR, "cluster_profile.csv")

# Module-level cache -- populated once on first use, reused after that
_model: Optional[Any] = None
_scaler: Optional[Any] = None
_profile: Optional[pd.DataFrame] = None


def _load_artifacts():
    """Loads the trained model + scaler from disk exactly once, then caches them."""
    global _model, _scaler, _profile

    if _model is not None and _scaler is not None:
        return  # already loaded, skip disk access entirely

    if not os.path.exists(_MODEL_PATH) or not os.path.exists(_SCALER_PATH):
        raise FileNotFoundError(
            "Trained model files not found in models/segmentation/. "
            "Run 'python models/segmentation/train_model.py' once from the "
            "repo root before using this module."
        )

    with open(_MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    with open(_SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)

    if os.path.exists(_PROFILE_PATH):
        _profile = pd.read_csv(_PROFILE_PATH)


def predict_customer_segment(income: float, spending: float) -> int:
    """
    Predict which cluster a new customer belongs to.

    Parameters
    ----------
    income : Annual income in k$ (e.g. 70.0 for $70k)
    spending : Spending score, 0-100

    Returns
    -------
    int - the predicted cluster ID (0 to k-1)

    This is the function to hand off to the frontend, e.g.:
        cluster = predict_customer_segment(70.0, 78.0)
    """
    _load_artifacts()
    assert _scaler is not None, "Failed to load scaler"
    assert _model is not None, "Failed to load model"
    input_df = pd.DataFrame([[income, spending]], columns=FEATURES)
    input_scaled = _scaler.transform(input_df)
    return int(_model.predict(input_scaled)[0])


def get_cluster_profile() -> pd.DataFrame:
    """
    Returns the saved cluster summary table (mean income/spending + count per
    cluster), for the frontend to display alongside a prediction.
    """
    _load_artifacts()
    assert _profile is not None, "Failed to load cluster profile"
    return _profile


def get_num_clusters() -> int:
    """Returns how many clusters the loaded model has (k)."""
    _load_artifacts()
    assert _model is not None, "Failed to load model"
    return _model.n_clusters


# --------------------------------------------------------------------------- #
# Quick manual test -- run this file directly to confirm the saved model loads
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    cluster = predict_customer_segment(70.0, 78.0)
    print(f"Predicted cluster for income=70k, spending=78: {cluster}")
    print("\nCluster profile:")
    print(get_cluster_profile())
