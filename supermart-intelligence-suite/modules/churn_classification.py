import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

MODEL_KEYS = {
    "naive_bayes": "Naive Bayes",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}


# ---------------------------------------------------------------------------
# FR-1.1 — Load and prepare churn dataset
# ---------------------------------------------------------------------------
def load_data(path: str = "data/Churn_Modelling.csv") -> pd.DataFrame:
    """Load the raw churn dataset."""
    return pd.read_csv(path)


def preprocess_features(df: pd.DataFrame):
    """
    Drop identifier columns, encode categoricals, and split into X/y.

    Returns:
        (X, y)
    """
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    return X, y


def prepare_features(df: pd.DataFrame):
    """
    Public entry point matching the module contract (FR-1.1): encode
    features and split train/test in one call.

    Returns:
        (X_train, X_test, y_train, y_test, scaler, feature_columns)
    """
    X, y = preprocess_features(df)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_columns


# ---------------------------------------------------------------------------
# FR-1.2 — Train and compare classifiers
# ---------------------------------------------------------------------------
def train_models(X_train, y_train) -> dict:
    """
    Fit Naive Bayes, Decision Tree, and Random Forest classifiers.

    Returns:
        dict mapping model_key -> fitted model object,
        e.g. {"naive_bayes": model, "decision_tree": model, "random_forest": model}
    """
    models = {
        "naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Compute accuracy/precision/recall for each fitted model.

    Returns:
        A DataFrame with one row per model — used to render the
        comparison table/chart (FR-1.2).
    """
    rows = []
    for key, model in models.items():
        y_pred = model.predict(X_test)
        rows.append({
            "model_key": key,
            "Model": MODEL_KEYS[key],
            "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Recall": round(recall_score(y_test, y_pred), 3),
        })
    return pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)


def get_best_model(models: dict, comparison: pd.DataFrame):
    """Return (model_key, fitted_model) for the highest-accuracy model."""
    best_key = comparison.iloc[0]["model_key"]
    return best_key, models[best_key]


# ---------------------------------------------------------------------------
# FR-1.3 — Public prediction function (called by app.py)
# ---------------------------------------------------------------------------
def predict_churn(model_name: str, customer_features: dict, models: dict, scaler, feature_columns: list) -> dict:
    """
    Public inference function called by app.py (FR-1.3).

    Inputs:
        model_name: one of "naive_bayes", "decision_tree", "random_forest"
        customer_features: dict of a single customer's feature values,
            matching feature_columns (after Geography/Gender encoding)
        models: dict returned by train_models()
        scaler: fitted StandardScaler from prepare_features()
        feature_columns: column order returned by prepare_features()

    Returns:
        {"prediction": "Churn" | "No Churn", "confidence": float}
    """
    if model_name not in models:
        raise ValueError(f"Unknown model_name '{model_name}'. Options: {list(models.keys())}")

    row = pd.DataFrame([customer_features])[feature_columns]
    row_scaled = scaler.transform(row)

    model = models[model_name]
    prediction = model.predict(row_scaled)[0]

    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(row_scaled)[0].max())
    else:
        confidence = 1.0

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "confidence": round(confidence, 4),
    }


# ---------------------------------------------------------------------------
# Full pipeline — convenient for standalone testing / a single cached call
# ---------------------------------------------------------------------------
def run_classification(path: str = "data/Churn_Modelling.csv") -> dict:
    """
    Runs the full pipeline once. Intended to be wrapped in
    @st.cache_resource in app.py so training happens once per session,
    not on every click (same pattern as the market basket module).
    """
    df = load_data(path)
    X_train, X_test, y_train, y_test, scaler, feature_columns = prepare_features(df)
    models = train_models(X_train, y_train)
    comparison = evaluate_models(models, X_test, y_test)
    best_key, best_model = get_best_model(models, comparison)

    return {
        "models": models,
        "comparison": comparison,
        "best_model_key": best_key,
        "best_model": best_model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "X_test": X_test,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Standalone execution / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    output = run_classification()

    print("\n========== MODEL COMPARISON ==========\n")
    print(output["comparison"][["Model", "Accuracy (%)", "Precision", "Recall"]])
    print("\nBest Model:", output["comparison"].iloc[0]["Model"])

    sample_customer = {
        "CreditScore": 650,
        "Gender": 1,
        "Age": 55,
        "Tenure": 3,
        "Balance": 120000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 90000.0,
        "Geography_Germany": True,
        "Geography_Spain": False,
    }
    result = predict_churn(
        "random_forest", sample_customer,
        output["models"], output["scaler"], output["feature_columns"],
    )
    print("\nSample prediction (random_forest):", result)