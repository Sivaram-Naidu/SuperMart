import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model

# -----------------------------------------------------
# Load Models
# -----------------------------------------------------

@st.cache_resource
def load_models():
    ann_model = load_model("models/deep_learning/ann_churn_model.keras")
    scaler = joblib.load("models/deep_learning/scaler.pkl")

    cnn_model = load_model("models/deep_learning/sis_fashion_cnn.keras")

    return ann_model, scaler, cnn_model


ann_model, scaler, cnn_model = load_models()

# -----------------------------------------------------
# CNN Class Names
# -----------------------------------------------------

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

st.sidebar.title("Deep Learning")

page = st.sidebar.radio(
    "Select Module",
    [
        "ANN Customer Churn Prediction",
        "CNN Fashion Classification"
    ]
)

# =====================================================
# ANN PAGE
# =====================================================

if page == "ANN Customer Churn Prediction":

    st.title("Customer Churn Prediction")

    st.write("Enter Customer Details")

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0
    )

    products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=2
    )

    credit_card = st.selectbox(
        "Has Credit Card",
        ["Yes", "No"]
    )

    active_member = st.selectbox(
        "Is Active Member",
        ["Yes", "No"]
    )

    salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0
    )

    if st.button("Predict Churn"):

        gender = 1 if gender == "Male" else 0

        has_card = 1 if credit_card == "Yes" else 0

        active = 1 if active_member == "Yes" else 0

        germany = 1 if geography == "Germany" else 0
        spain = 1 if geography == "Spain" else 0

        customer = np.array([[
            credit_score,
            gender,
            age,
            tenure,
            balance,
            products,
            has_card,
            active,
            salary,
            germany,
            spain
        ]])

        customer = scaler.transform(customer)

        probability = ann_model.predict(customer, verbose=0)[0][0]

        threshold = 0.3

        st.subheader("Prediction Result")

        if probability >= threshold:
            st.error("Customer is likely to Churn")
        else:
            st.success("Customer is likely to Stay")

        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )
        # =====================================================
# CNN PAGE
# =====================================================

elif page == "CNN Fashion Classification":

    st.title("Fashion Image Classification")

    st.write("Upload a fashion image (28x28 grayscale recommended).")

    uploaded_file = st.file_uploader(
        "Choose an Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")

        st.image(image, caption="Uploaded Image", width=250)

        # Resize to 28x28
        image = image.resize((28, 28))

        # Convert to array
        img = np.array(image).astype("float32")

        # Normalize
        img = img / 255.0

        # Reshape for CNN
        img = img.reshape(1, 28, 28, 1)

        prediction = cnn_model.predict(img, verbose=0)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction)

        st.subheader("Prediction")

        st.success(CLASS_NAMES[predicted_class])

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )


