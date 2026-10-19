import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_PATH = "data/channel_dataset.csv"
MODEL_PATH = "models/classifier.pkl"
RANDOM_SEED = 42
TEST_SIZE = 0.20


def train_model():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "Dataset not found. Run python dataset.py first."
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "snr_db",
        "channel_type",
        "rician_k_factor",
        "impairment",
        "ber_nrz_l",
        "ber_manchester",
        "ber_differential_manchester",
        "ber_ami",
        "best_scheme"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    print("Dataset loaded:", len(df), "records")

    print("\nBest-scheme distribution:")
    print(df["best_scheme"].value_counts())

    print("\nBest scheme by channel condition:")
    print(
        df.groupby(["channel_type", "best_scheme"])
        .size()
        .unstack(fill_value=0)
    )

    print("\nAverage BER by channel condition:")
    print(
        df.groupby("channel_type")[
            [
                "ber_nrz_l",
                "ber_manchester",
                "ber_differential_manchester",
                "ber_ami"
            ]
        ].mean().round(6)
    )

    X = df[
        [
            "snr_db",
            "channel_type",
            "rician_k_factor",
            "impairment"
        ]
    ]

    y = df["best_scheme"]

    class_counts = y.value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "Only one scheme appears in best_scheme. "
            "Fix the dataset before training."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Every class needs at least two records for stratified splitting."
        )

    categorical_features = ["channel_type", "impairment"]
    numerical_features = ["snr_db", "rician_k_factor"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1
                )
            )
        ]
    )

    print("\nTraining classifier...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    labels = model.named_steps["classifier"].classes_

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=labels))
    print("Class order:", list(labels))

    os.makedirs("models", exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print("\nModel saved to", MODEL_PATH)


if __name__ == "__main__":
    train_model()