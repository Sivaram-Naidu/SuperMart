"""Deep learning backend helpers for the SuperMart SIS dashboard.

The module exposes reusable loading, preprocessing, inference, metadata,
and evaluation helpers for the ANN churn model and the CNN fashion model.
It intentionally avoids any Streamlit page layout so the UI can live in
``pages/deep_learning.py`` and the router can call into this backend.
"""

from __future__ import annotations

import json
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import h5py
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models" / "deep_learning"
ANN_MODEL_PATH = MODEL_DIR / "ann_churn_model.keras"
CNN_MODEL_PATH = MODEL_DIR / "sis_fashion_cnn.keras"
CNN_LEGACY_MODEL_PATH = MODEL_DIR / "sis_fashion_cnn.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

ANN_RAW_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

ANN_FEATURE_COLUMNS = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
]

ANN_LABEL_CANDIDATES = ["Exited", "exited", "Exited ", "target", "label"]

CNN_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

CNN_LABEL_CANDIDATES = ["label", "Label", "target", "Target", "class", "Class"]
CNN_IMAGE_SIZE = (28, 28)


def _normalize_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _read_keras_config(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as archive:
        return json.loads(archive.read("config.json").decode("utf-8"))


def _read_h5_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        metadata: dict[str, Any] = {
            "attrs": {key: handle.attrs[key] for key in handle.attrs.keys()}
        }
    return metadata


def _layer_summary_from_keras_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    layers = []
    for layer in config.get("config", {}).get("layers", []):
        layer_config = layer.get("config", {})
        layers.append(
            {
                "type": layer.get("class_name"),
                "name": layer_config.get("name"),
                "units": layer_config.get("units"),
                "filters": layer_config.get("filters"),
                "activation": layer_config.get("activation"),
                "shape": layer_config.get("batch_shape") or layer_config.get("input_shape"),
            }
        )
    return layers


def _optimizer_name_from_compile_config(compile_config: Any) -> str | None:
    if isinstance(compile_config, dict):
        optimizer = compile_config.get("optimizer")
        if isinstance(optimizer, dict):
            return optimizer.get("class_name") or optimizer.get("config", {}).get("name")
        if isinstance(optimizer, str):
            return optimizer

        optimizer_config = compile_config.get("optimizer_config")
        if isinstance(optimizer_config, dict):
            return optimizer_config.get("class_name") or optimizer_config.get("config", {}).get("name")
    if isinstance(compile_config, str):
        return compile_config
    return None


def _loss_name_from_compile_config(compile_config: Any) -> str | None:
    if isinstance(compile_config, dict):
        loss = compile_config.get("loss")
        if isinstance(loss, str):
            return loss
        if isinstance(loss, dict):
            return loss.get("class_name") or loss.get("config", {}).get("name")
    return None


@lru_cache(maxsize=1)
def get_ann_model_info() -> dict[str, Any]:
    config = _read_keras_config(ANN_MODEL_PATH)
    layers = _layer_summary_from_keras_config(config)
    compile_config = config.get("compile_config", {})
    return {
        "name": config.get("config", {}).get("name", "ANN churn model"),
        "type": config.get("class_name", "Sequential"),
        "input_shape": tuple(config.get("build_config", {}).get("input_shape", [])),
        "output_units": layers[-1].get("units") if layers else None,
        "layers": layers,
        "optimizer": _optimizer_name_from_compile_config(compile_config),
        "loss": _loss_name_from_compile_config(compile_config),
        "metrics": compile_config.get("metrics", []),
        "feature_columns": ANN_FEATURE_COLUMNS,
        "raw_columns": ANN_RAW_COLUMNS,
        "label_candidates": ANN_LABEL_CANDIDATES,
    }


@lru_cache(maxsize=1)
def get_cnn_model_info() -> dict[str, Any]:
    keras_config = _read_keras_config(CNN_MODEL_PATH)
    h5_metadata = _read_h5_metadata(CNN_LEGACY_MODEL_PATH)
    layers = _layer_summary_from_keras_config(keras_config)

    training_config: dict[str, Any] = {}
    raw_training_config = h5_metadata.get("attrs", {}).get("training_config")
    if isinstance(raw_training_config, bytes):
        raw_training_config = raw_training_config.decode("utf-8")
    if isinstance(raw_training_config, str):
        try:
            training_config = json.loads(raw_training_config)
        except json.JSONDecodeError:
            training_config = {}

    return {
        "name": keras_config.get("config", {}).get("name", "CNN fashion model"),
        "type": keras_config.get("class_name", "Sequential"),
        "input_shape": tuple(keras_config.get("build_config", {}).get("input_shape", [])),
        "output_units": layers[-1].get("units") if layers else None,
        "layers": layers,
        "optimizer": _optimizer_name_from_compile_config(training_config),
        "loss": _loss_name_from_compile_config(training_config),
        "metrics": training_config.get("metrics", []),
        "class_names": CNN_CLASS_NAMES,
        "label_candidates": CNN_LABEL_CANDIDATES,
        "image_size": CNN_IMAGE_SIZE,
    }


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict[str, Any]:
    """Load the ANN/CNN models and scaler with graceful failure handling."""

    bundle: dict[str, Any] = {
        "ok": False,
        "error": None,
        "tensorflow_version": None,
        "ann_model": None,
        "cnn_model": None,
        "scaler": None,
    }

    try:
        import tensorflow as tf  # noqa: WPS433 - runtime import keeps module importable without TF
        from tensorflow.keras.models import load_model
    except Exception as exc:  # pragma: no cover - exercised in runtime validation
        bundle["error"] = f"TensorFlow import failed: {exc}"
        return bundle

    try:
        bundle["ann_model"] = load_model(str(ANN_MODEL_PATH))
        bundle["cnn_model"] = load_model(str(CNN_MODEL_PATH))
        bundle["scaler"] = joblib.load(SCALER_PATH)
        bundle["tensorflow_version"] = tf.__version__
        bundle["ok"] = True
    except Exception as exc:  # pragma: no cover - exercised in runtime validation
        bundle["error"] = f"Model loading failed: {exc}"
    return bundle


def get_backend_status() -> dict[str, Any]:
    bundle = load_model_bundle()
    return {
        "ok": bool(bundle.get("ok")),
        "error": bundle.get("error"),
        "tensorflow_version": bundle.get("tensorflow_version"),
        "ann_loaded": bundle.get("ann_model") is not None,
        "cnn_loaded": bundle.get("cnn_model") is not None,
        "scaler_loaded": bundle.get("scaler") is not None,
    }


def _detect_label_column(dataframe: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized_to_column = {_normalize_name(column): column for column in dataframe.columns}
    for candidate in candidates:
        column = normalized_to_column.get(_normalize_name(candidate))
        if column is not None:
            return column
    return None


def summarize_dataset(dataframe: pd.DataFrame, model_key: str) -> dict[str, Any]:
    summary = {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "missing_values": int(dataframe.isna().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "numeric_columns": int(len(dataframe.select_dtypes(include=[np.number]).columns)),
    }
    if model_key == "ann":
        summary["label_column"] = _detect_label_column(dataframe, ANN_LABEL_CANDIDATES)
    else:
        summary["label_column"] = _detect_label_column(dataframe, CNN_LABEL_CANDIDATES)
    return summary


def _prepare_ann_feature_frame(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None, str | None]:
    normalized_columns = {_normalize_name(column): column for column in dataframe.columns}
    label_column = _detect_label_column(dataframe, ANN_LABEL_CANDIDATES)
    label_series = dataframe[label_column].copy() if label_column else None

    if all(_normalize_name(column) in normalized_columns for column in ANN_RAW_COLUMNS):
        source = dataframe.copy()
        geography = source[normalized_columns[_normalize_name("Geography")]].astype(str).str.strip().str.lower()
        gender = source[normalized_columns[_normalize_name("Gender")]].astype(str).str.strip().str.lower()

        feature_frame = pd.DataFrame(
            {
                "CreditScore": pd.to_numeric(source[normalized_columns[_normalize_name("CreditScore")]], errors="coerce"),
                "Gender": gender.map({"male": 1, "female": 0}),
                "Age": pd.to_numeric(source[normalized_columns[_normalize_name("Age")]], errors="coerce"),
                "Tenure": pd.to_numeric(source[normalized_columns[_normalize_name("Tenure")]], errors="coerce"),
                "Balance": pd.to_numeric(source[normalized_columns[_normalize_name("Balance")]], errors="coerce"),
                "NumOfProducts": pd.to_numeric(source[normalized_columns[_normalize_name("NumOfProducts")]], errors="coerce"),
                "HasCrCard": pd.to_numeric(source[normalized_columns[_normalize_name("HasCrCard")]], errors="coerce"),
                "IsActiveMember": pd.to_numeric(source[normalized_columns[_normalize_name("IsActiveMember")]], errors="coerce"),
                "EstimatedSalary": pd.to_numeric(source[normalized_columns[_normalize_name("EstimatedSalary")]], errors="coerce"),
                "Geography_Germany": geography.eq("germany").astype(int),
                "Geography_Spain": geography.eq("spain").astype(int),
            }
        )
    else:
        available = {_normalize_name(column): column for column in dataframe.columns}
        required = [available.get(_normalize_name(column)) for column in ANN_FEATURE_COLUMNS]
        missing = [column for column, source in zip(ANN_FEATURE_COLUMNS, required) if source is None]
        if missing:
            raise ValueError(
                "ANN predictions need either the raw churn columns or the encoded feature columns. Missing: "
                + ", ".join(missing)
            )
        feature_frame = dataframe[[available[_normalize_name(column)] for column in ANN_FEATURE_COLUMNS]].copy()
        feature_frame.columns = ANN_FEATURE_COLUMNS
        for column in ANN_FEATURE_COLUMNS:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")

    feature_frame = feature_frame.astype(float)
    if feature_frame.isna().any().any():
        raise ValueError("ANN feature columns contain non-numeric or missing values after preprocessing.")
    return feature_frame, label_series, label_column


def _prepare_cnn_feature_frame(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None, str | None]:
    label_column = _detect_label_column(dataframe, CNN_LABEL_CANDIDATES)
    label_series = dataframe[label_column].copy() if label_column else None

    feature_candidates = [column for column in dataframe.columns if column != label_column]
    feature_frame = dataframe[feature_candidates].apply(pd.to_numeric, errors="coerce")
    numeric_columns = list(feature_frame.select_dtypes(include=[np.number]).columns)

    if len(numeric_columns) < 784:
        raise ValueError(
            "CNN predictions need at least 784 numeric pixel columns per row or a 28x28 image upload."
        )

    feature_frame = feature_frame[numeric_columns[:784]].copy()
    if feature_frame.isna().any().any():
        raise ValueError("CNN pixel columns contain non-numeric or missing values after preprocessing.")
    return feature_frame, label_series, label_column


def predict_ann(bundle: dict[str, Any], dataframe: pd.DataFrame, threshold: float = 0.3) -> dict[str, Any]:
    if not bundle.get("ok"):
        raise RuntimeError(bundle.get("error") or "ANN model bundle is unavailable.")

    features, label_series, label_column = _prepare_ann_feature_frame(dataframe)
    scaled = bundle["scaler"].transform(features)
    probabilities = np.asarray(bundle["ann_model"].predict(scaled, verbose=0)).reshape(-1)
    predicted_labels = np.where(probabilities >= threshold, "Churn", "Stay")
    confidence = np.where(probabilities >= threshold, probabilities, 1.0 - probabilities)

    result = dataframe.copy().reset_index(drop=True)
    result["prediction_probability"] = probabilities
    result["prediction_label"] = predicted_labels
    result["prediction_confidence"] = confidence
    result["prediction_status"] = np.where(probabilities >= threshold, "Alert", "Healthy")

    return {
        "dataframe": result,
        "features": features,
        "labels": label_series.reset_index(drop=True) if label_series is not None else None,
        "label_column": label_column,
        "probabilities": probabilities,
        "predictions": predicted_labels,
        "confidence": confidence,
        "threshold": threshold,
    }


def predict_cnn(bundle: dict[str, Any], dataframe: pd.DataFrame) -> dict[str, Any]:
    if not bundle.get("ok"):
        raise RuntimeError(bundle.get("error") or "CNN model bundle is unavailable.")

    features, label_series, label_column = _prepare_cnn_feature_frame(dataframe)
    pixel_values = features.to_numpy(dtype=np.float32)
    if pixel_values.max(initial=0.0) > 1.0:
        pixel_values = pixel_values / 255.0

    images = pixel_values.reshape((-1, CNN_IMAGE_SIZE[0], CNN_IMAGE_SIZE[1], 1))
    probabilities = np.asarray(bundle["cnn_model"].predict(images, verbose=0))
    predicted_indices = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    predicted_labels = [CNN_CLASS_NAMES[index] for index in predicted_indices]

    result = dataframe.copy().reset_index(drop=True)
    result["prediction_index"] = predicted_indices
    result["prediction_label"] = predicted_labels
    result["prediction_confidence"] = confidence
    result["prediction_status"] = np.where(confidence >= 0.5, "Confident", "Review")

    return {
        "dataframe": result,
        "features": features,
        "labels": label_series.reset_index(drop=True) if label_series is not None else None,
        "label_column": label_column,
        "probabilities": probabilities,
        "predictions": predicted_labels,
        "confidence": confidence,
        "predicted_indices": predicted_indices,
    }


def evaluate_binary_predictions(labels: pd.Series, probabilities: np.ndarray, threshold: float = 0.3) -> dict[str, Any]:
    y_true = pd.to_numeric(labels, errors="coerce").astype(int).to_numpy()
    y_pred = (probabilities >= threshold).astype(int)

    confusion = confusion_matrix(y_true, y_pred)
    fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": roc_thresholds,
        "support": int(len(y_true)),
    }


def evaluate_multiclass_predictions(labels: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    y_true = pd.to_numeric(labels, errors="coerce").astype(int).to_numpy()
    predicted_indices = np.argmax(probabilities, axis=1)
    classes = np.arange(probabilities.shape[1])
    y_true_bin = label_binarize(y_true, classes=classes)

    confusion = confusion_matrix(y_true, predicted_indices, labels=classes)

    curves = []
    for class_index in classes:
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], probabilities[:, class_index])
            curves.append({"class_index": int(class_index), "fpr": fpr, "tpr": tpr})
        except ValueError:
            continue

    try:
        roc_auc = float(roc_auc_score(y_true_bin, probabilities, average="macro", multi_class="ovr"))
    except ValueError:
        roc_auc = None

    return {
        "accuracy": float(accuracy_score(y_true, predicted_indices)),
        "precision": float(precision_score(y_true, predicted_indices, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, predicted_indices, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, predicted_indices, average="macro", zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion,
        "roc_curves": curves,
        "support": int(len(y_true)),
    }


def build_model_report(model_key: str) -> dict[str, Any]:
    if model_key == "ann":
        info = get_ann_model_info()
    elif model_key == "cnn":
        info = get_cnn_model_info()
    else:
        raise ValueError(f"Unsupported model key: {model_key}")

    activations = [layer["activation"] for layer in info["layers"] if layer.get("activation")]
    return {
        "name": info.get("name"),
        "type": info.get("type"),
        "input_shape": info.get("input_shape"),
        "output_units": info.get("output_units"),
        "optimizer": info.get("optimizer"),
        "loss": info.get("loss"),
        "metrics": info.get("metrics", []),
        "layer_count": len(info.get("layers", [])),
        "activations": activations,
        "layers": info.get("layers", []),
        "feature_columns": info.get("feature_columns"),
        "class_names": info.get("class_names"),
        "image_size": info.get("image_size"),
        "training_epochs": None,
        "batch_size": None,
        "training_dataset": None,
    }


