"""Deep Learning dashboard for the SuperMart Intelligence Suite.

This page is strictly a frontend consumer of the existing backend helpers
in `modules/deep_learning.py`. It collects the exact model inputs required
by the ANN churn predictor and the CNN image classifier, sends those inputs
to the backend inference functions, and renders the returned outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from components.navbar import render_navbar
from modules.deep_learning import (
    build_model_report,
    get_backend_status,
    load_model_bundle,
    predict_ann,
    predict_cnn,
)


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


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _init_state() -> None:
    defaults = {
        "deep_learning_history": [],
        "deep_learning_last_prediction": None,
        "deep_learning_last_model": "ann",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _card(title: str, value: str, note: str, accent: str = "#1976D2") -> None:
    st.markdown(
        f"""
        <div class="dl-stat-card" style="border-top:4px solid {accent};">
            <p class="dl-stat-label">{title}</p>
            <p class="dl-stat-value">{value}</p>
            <p class="dl-stat-note">{note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _cards(items: list[tuple[str, str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        with column:
            _card(*item)


def _hero() -> None:
    st.markdown(
        """
        <div class="dl-hero">
            <div class="dl-chip-row">
                <span class="dl-chip">ANN churn inference</span>
                <span class="dl-chip">CNN fashion inference</span>
                <span class="dl-chip">Backend-driven outputs only</span>
            </div>
            <h1>Deep Learning Dashboard</h1>
            <p>
                Enter the exact model inputs required by the existing ANN and CNN backends,
                run inference, and review the model outputs without any retraining or placeholder metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _backend_banner(status: dict[str, Any]) -> None:
    if status["ok"]:
        st.markdown(
            f"<div class='dl-note'>Backend ready. TensorFlow {status['tensorflow_version']} loaded the ANN scaler and both model artifacts successfully.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='dl-warning'><strong>Backend unavailable.</strong> {status['error']}</div>",
            unsafe_allow_html=True,
        )


def _render_model_info(model_key: str) -> dict[str, Any]:
    report = build_model_report(model_key)
    summary_left, summary_right = st.columns([1, 1.15])

    with summary_left:
        _cards(
            [
                ("Model type", str(report.get("type")), "Sequential architecture", "#1976D2"),
                ("Input shape", str(report.get("input_shape")), "Expected backend input", "#22C55E"),
                ("Output units", str(report.get("output_units")), "Final layer width", "#F59E0B"),
                ("Layers", f"{report.get('layer_count', 0):,}", "Learned layers in the model", "#8B5CF6"),
            ]
        )
        st.markdown("<div class='dl-section-heading'>Compilation summary</div>", unsafe_allow_html=True)
        # Render compilation fields as formatted JSON for clarity and stability
        compilation = {
            "optimizer": report.get("optimizer") or "Not stored",
            "loss": report.get("loss") or "Not stored",
            "metrics": report.get("metrics") or [],
            "activations": report.get("activations") or [],
            "training_epochs": report.get("training_epochs") or "Not stored",
            "batch_size": report.get("batch_size") or "Not stored",
            "training_dataset": report.get("training_dataset") or "Not stored",
        }
        st.json(compilation)

    with summary_right:
        st.markdown("<div class='dl-section-heading'>Layer-by-layer summary</div>", unsafe_allow_html=True)
        layers = pd.DataFrame(report.get("layers", []))
        if not layers.empty:
            st.dataframe(layers, use_container_width=True, hide_index=True)
        else:
            st.info("No layer metadata was parsed from the model artifact.")

    return report


def _history_frame() -> pd.DataFrame:
    history = st.session_state.get("deep_learning_history", [])
    if not history:
        return pd.DataFrame(columns=["timestamp", "model", "prediction", "confidence", "probability", "status"])
    return pd.DataFrame(history)


def _append_history(entry: dict[str, Any]) -> None:
    history = list(st.session_state.get("deep_learning_history", []))
    history.append(entry)
    st.session_state["deep_learning_history"] = history[-250:]


def _ann_form() -> dict[str, Any]:
    st.markdown(
        """
        <div class="dl-panel">
            <p class="dl-panel-title">ANN input form</p>
            <p class="dl-panel-subtitle">Provide the raw churn features expected by the existing scaler and ANN model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("ann_input_form"):
        cols = st.columns(2)
        with cols[0]:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5, step=1)
        with cols[1]:
            balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=100.0)
            num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1, step=1)
            has_card = st.selectbox("Has Credit Card", ["No", "Yes"])
            active_member = st.selectbox("Is Active Member", ["No", "Yes"])
            salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=100.0)

        submitted = st.form_submit_button("Predict")

    return {
        "submitted": submitted,
        "values": {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": 1 if has_card == "Yes" else 0,
            "IsActiveMember": 1 if active_member == "Yes" else 0,
            "EstimatedSalary": salary,
        },
    }


def _cnn_form() -> dict[str, Any]:
    st.markdown(
        """
        <div class="dl-panel">
            <p class="dl-panel-title">CNN image input</p>
            <p class="dl-panel-subtitle">Upload a single grayscale or color fashion image. The page will resize it to 28x28 and send the pixel frame to the existing CNN backend.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    submit = st.button("Predict", use_container_width=True, disabled=uploaded_file is None)
    return {"submitted": submit, "file": uploaded_file}


def _image_to_frame(uploaded_file) -> tuple[pd.DataFrame, Image.Image]:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    pixels = np.asarray(image, dtype=np.float32).reshape(1, -1)
    columns = [f"pixel_{index:03d}" for index in range(pixels.shape[1])]
    return pd.DataFrame(pixels, columns=columns), image


def _binary_chart(probability: float, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Stay", "Churn"],
            y=[1.0 - probability, probability],
            marker_color=["#93C5FD", "#1976D2"],
        )
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444")
    fig.update_layout(
        title="Prediction probability",
        yaxis=dict(range=[0, 1]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=310,
    )
    return fig


def _multiclass_chart(probabilities: np.ndarray) -> go.Figure:
    top_indices = np.argsort(probabilities)[-5:][::-1]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[CNN_CLASS_NAMES[index] for index in top_indices],
            y=[float(probabilities[index]) for index in top_indices],
            marker_color="#1976D2",
        )
    )
    fig.update_layout(
        title="Top class probabilities",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=310,
        xaxis_tickangle=-25,
    )
    return fig


def _render_prediction_result(result: dict[str, Any], model_key: str, elapsed_seconds: float) -> None:
    if model_key == "ann":
        probability = float(result["probabilities"][0])
        confidence = float(result["confidence"][0])
        predicted = str(result["predictions"][0])
        status = "Churn Alert" if probability >= result["threshold"] else "Stable"
        chart = _binary_chart(probability, result["threshold"])
        predicted_probability = probability
    else:
        probability_array = np.asarray(result["probabilities"])[0]
        predicted = str(result["predictions"][0])
        confidence = float(result["confidence"][0])
        predicted_probability = confidence
        status = "Confident" if confidence >= 0.5 else "Review"
        chart = _multiclass_chart(probability_array)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _card("Prediction", predicted, "Backend output", "#1976D2")
    with c2:
        _card("Confidence", f"{confidence:.2%}", "Highest class confidence", "#22C55E")
    with c3:
        _card("Probability", f"{predicted_probability:.2%}", "Backend probability", "#F59E0B")
    with c4:
        _card("Inference time", f"{elapsed_seconds:.3f}s", "Frontend timing only", "#8B5CF6")

    st.markdown(
        f"<div class='dl-note'><strong>Status:</strong> {status}</div>",
        unsafe_allow_html=True,
    )

    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

    history_entry = {
        "timestamp": _now(),
        "model": "ANN" if model_key == "ann" else "CNN",
        "prediction": predicted,
        "confidence": confidence,
        "probability": predicted_probability,
        "status": status,
    }
    _append_history(history_entry)

    st.download_button(
        "Export prediction report",
        pd.DataFrame([history_entry | {"inference_time_seconds": elapsed_seconds}]).to_csv(index=False).encode("utf-8"),
        file_name=f"deep_learning_{model_key}_prediction.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_deep_learning_page() -> None:
    render_navbar()
    _init_state()

    status = get_backend_status()
    _hero()
    _backend_banner(status)

    st.markdown(
        """
        <div class="dl-panel">
            <p class="dl-panel-title">Control center</p>
            <p class="dl-panel-subtitle">Choose a model, enter only the required inputs, and send them to the existing backend for inference.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_choice = st.selectbox(
        "Model selection",
        ["ANN Customer Churn", "CNN Fashion Classification"],
        index=0 if st.session_state.get("deep_learning_last_model") == "ann" else 1,
    )
    model_key = "ann" if model_choice.startswith("ANN") else "cnn"
    st.session_state["deep_learning_last_model"] = model_key

    _render_model_info(model_key)

    input_col, output_col = st.columns([1.05, 0.95])
    with input_col:
        if model_key == "ann":
            form = _ann_form()
            threshold = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.30, step=0.01)
            if form["submitted"]:
                frame = pd.DataFrame([form["values"]])
                started = time.perf_counter()
                try:
                    bundle = load_model_bundle()
                    result = predict_ann(bundle, frame, threshold=threshold)
                    elapsed = time.perf_counter() - started
                    st.session_state["deep_learning_last_prediction"] = {
                        "model_key": model_key,
                        "result": result,
                        "elapsed": elapsed,
                        "threshold": threshold,
                    }
                    st.success("ANN prediction completed successfully.")
                except Exception as exc:
                    st.error(str(exc))
        else:
            form = _cnn_form()
            if form["submitted"] and form.get("file") is not None:
                frame, image = _image_to_frame(form["file"])
                st.image(image, caption="Uploaded image resized to 28x28 grayscale", width=220)
                started = time.perf_counter()
                try:
                    bundle = load_model_bundle()
                    result = predict_cnn(bundle, frame)
                    elapsed = time.perf_counter() - started
                    st.session_state["deep_learning_last_prediction"] = {
                        "model_key": model_key,
                        "result": result,
                        "elapsed": elapsed,
                    }
                    st.success("CNN prediction completed successfully.")
                except Exception as exc:
                    st.error(str(exc))

    with output_col:
        st.markdown(
            """
            <div class="dl-panel">
                <p class="dl-panel-title">Prediction output</p>
                <p class="dl-panel-subtitle">Displays the backend response, confidence, probability, timing, and status for the latest run.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        last = st.session_state.get("deep_learning_last_prediction")
        if last and last.get("model_key") == model_key:
            _render_prediction_result(last["result"], model_key, last["elapsed"])
        else:
            st.info("Run a prediction to populate the output cards and chart.")

    st.markdown(
        """
        <div class="dl-panel">
            <p class="dl-panel-title">Prediction history</p>
            <p class="dl-panel-subtitle">Session-level history of the latest predictions, ready for export or review.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    history = _history_frame()
    if not history.empty:
        st.dataframe(history.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No predictions have been run in this session yet.")

    st.markdown(
        '<div class="footer">SuperMart SIS Deep Learning Dashboard</div>',
        unsafe_allow_html=True,
    )
