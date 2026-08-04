"""
Sentiment Analysis Module

This module trains, saves, loads, and predicts restaurant review sentiment
using Bag of Words and Multinomial Naive Bayes.

Author: Sarupya Tiguti
"""

import os
import re

import joblib
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# -------------------------------------------------------------------
# Download NLTK resources if not available
# -------------------------------------------------------------------
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
MODEL_DIR = os.path.join("models", "sentiment")
MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")

MAX_FEATURES = 1500

ps = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))


# -------------------------------------------------------------------
# Text Preprocessing
# -------------------------------------------------------------------
def preprocess_text(text):
    """
    Clean and preprocess a restaurant review.

    Steps:
    - Remove special characters
    - Convert to lowercase
    - Remove stopwords
    - Apply Porter Stemming

    Args:
        text (str): Input review

    Returns:
        str: Cleaned review
    """

    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(words)


# -------------------------------------------------------------------
# Model Training
# -------------------------------------------------------------------
def train_model():
    """
    Train the sentiment analysis model and save it.

    Returns:
        float: Model accuracy
    """

    os.makedirs(MODEL_DIR, exist_ok=True)

    dataset = pd.read_csv(
        "data/Restaurant_Reviews.tsv",
        delimiter="\t",
        quoting=3
    )

    corpus = [preprocess_text(review) for review in dataset["Review"]]

    vectorizer = CountVectorizer(max_features=MAX_FEATURES)

    X = vectorizer.fit_transform(corpus).toarray()
    y = dataset["Liked"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print("✅ Model trained successfully!")
    print(f"✅ Accuracy: {accuracy * 100:.2f}%")
    print(f"✅ Model saved to: {MODEL_FILE}")
    print(f"✅ Vectorizer saved to: {VECTORIZER_FILE}")

    return accuracy


# -------------------------------------------------------------------
# Load Saved Model
# -------------------------------------------------------------------
def load_model():
    """
    Load the trained model and vectorizer.

    If the model files are not found,
    train a new model automatically.

    Returns:
        tuple: (model, vectorizer)
    """

    if (
        not os.path.exists(MODEL_FILE)
        or not os.path.exists(VECTORIZER_FILE)
    ):
        print("⚠️ Trained model not found. Training a new model...")
        train_model()

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    return model, vectorizer


# -------------------------------------------------------------------
# Sentiment Prediction
# -------------------------------------------------------------------
def predict_sentiment(review):
    """
    Predict the sentiment of a restaurant review.

    Args:
        review (str): Restaurant review

    Returns:
        str: Positive 😊 or Negative 😞
    """

    if not review or not review.strip():
        return "Please enter a valid review."

    model, vectorizer = load_model()

    cleaned_review = preprocess_text(review)

    review_vector = vectorizer.transform(
        [cleaned_review]
    ).toarray()

    prediction = model.predict(review_vector)

    if prediction[0] == 1:
        return "Positive 😊"

    return "Negative 😞"


# -------------------------------------------------------------------
# Test Module
# -------------------------------------------------------------------
if __name__ == "__main__":

    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        accuracy = train_model()
        print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    else:
        print("✅ Trained model already exists.")

    print("\n----- Sample Predictions -----")

    review1 = "I loved this restaurant"
    review2 = "The food was terrible"

    print(f"\nReview: {review1}")
    print(f"Prediction: {predict_sentiment(review1)}")

    print(f"\nReview: {review2}")
    print(f"Prediction: {predict_sentiment(review2)}")