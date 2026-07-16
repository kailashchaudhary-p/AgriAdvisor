import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATASET_PATH = "Crop_recommendation.csv"
MODEL_PATH = "crop_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

CROP_PROFILES = {
    "Rice": {"N": (80, 120), "P": (20, 40), "K": (20, 80), "temperature": (20, 35), "humidity": (70, 90), "ph": (5.5, 7.0), "rainfall": (180, 260)},
    "Maize": {"N": (70, 130), "P": (10, 40), "K": (30, 80), "temperature": (18, 30), "humidity": (55, 75), "ph": (5.5, 7.0), "rainfall": (100, 200)},
    "Chickpea": {"N": (20, 60), "P": (10, 30), "K": (20, 60), "temperature": (18, 28), "humidity": (40, 70), "ph": (6.0, 7.5), "rainfall": (80, 140)},
    "Kidney Beans": {"N": (20, 50), "P": (10, 25), "K": (20, 50), "temperature": (15, 28), "humidity": (45, 75), "ph": (6.0, 7.0), "rainfall": (100, 180)},
    "Pigeon Peas": {"N": (20, 60), "P": (15, 35), "K": (30, 70), "temperature": (20, 35), "humidity": (45, 70), "ph": (6.0, 7.2), "rainfall": (90, 160)},
    "Moth Beans": {"N": (20, 60), "P": (15, 35), "K": (20, 60), "temperature": (22, 35), "humidity": (35, 65), "ph": (6.4, 7.5), "rainfall": (50, 100)},
    "Mung Bean": {"N": (20, 60), "P": (10, 30), "K": (15, 45), "temperature": (22, 35), "humidity": (40, 75), "ph": (6.0, 7.0), "rainfall": (70, 120)},
    "Black Gram": {"N": (20, 60), "P": (10, 30), "K": (20, 60), "temperature": (22, 35), "humidity": (40, 70), "ph": (6.0, 7.5), "rainfall": (60, 120)},
    "Lentil": {"N": (20, 50), "P": (10, 30), "K": (20, 60), "temperature": (15, 25), "humidity": (45, 75), "ph": (6.0, 7.5), "rainfall": (70, 120)},
    "Pomegranate": {"N": (20, 60), "P": (10, 30), "K": (20, 60), "temperature": (20, 35), "humidity": (35, 65), "ph": (6.0, 7.0), "rainfall": (80, 150)},
    "Banana": {"N": (80, 140), "P": (20, 50), "K": (80, 140), "temperature": (25, 35), "humidity": (80, 95), "ph": (6.0, 7.0), "rainfall": (200, 300)},
    "Mango": {"N": (20, 80), "P": (20, 50), "K": (60, 120), "temperature": (25, 35), "humidity": (70, 90), "ph": (6.0, 7.0), "rainfall": (200, 300)},
    "Grapes": {"N": (20, 80), "P": (20, 40), "K": (20, 100), "temperature": (15, 30), "humidity": (55, 75), "ph": (6.0, 7.0), "rainfall": (100, 180)},
    "Watermelon": {"N": (30, 80), "P": (10, 30), "K": (20, 60), "temperature": (20, 35), "humidity": (55, 80), "ph": (6.0, 7.0), "rainfall": (80, 160)},
    "Muskmelon": {"N": (20, 60), "P": (10, 30), "K": (20, 50), "temperature": (20, 35), "humidity": (55, 80), "ph": (6.0, 7.0), "rainfall": (80, 160)},
    "Apple": {"N": (20, 70), "P": (10, 40), "K": (20, 80), "temperature": (10, 25), "humidity": (60, 80), "ph": (6.0, 7.0), "rainfall": (100, 200)},
    "Orange": {"N": (20, 60), "P": (10, 30), "K": (20, 60), "temperature": (20, 35), "humidity": (55, 75), "ph": (6.0, 7.0), "rainfall": (80, 150)},
    "Papaya": {"N": (40, 80), "P": (20, 50), "K": (40, 80), "temperature": (25, 35), "humidity": (70, 90), "ph": (6.0, 7.0), "rainfall": (150, 300)},
    "Coconut": {"N": (50, 120), "P": (20, 50), "K": (100, 200), "temperature": (25, 35), "humidity": (80, 95), "ph": (6.0, 7.0), "rainfall": (180, 300)},
    "Cotton": {"N": (50, 100), "P": (10, 30), "K": (40, 80), "temperature": (20, 35), "humidity": (50, 70), "ph": (6.0, 7.0), "rainfall": (80, 160)},
    "Jute": {"N": (60, 100), "P": (20, 40), "K": (50, 100), "temperature": (20, 35), "humidity": (70, 90), "ph": (6.0, 7.0), "rainfall": (120, 220)},
    "Coffee": {"N": (50, 90), "P": (10, 30), "K": (40, 80), "temperature": (15, 25), "humidity": (70, 90), "ph": (5.5, 6.5), "rainfall": (140, 220)},
}


def build_synthetic_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: List[Dict[str, object]] = []

    for crop, profile in CROP_PROFILES.items():
        for _ in range(30):
            row = {
                "N": int(round(rng.uniform(*profile["N"]))),
                "P": int(round(rng.uniform(*profile["P"]))),
                "K": int(round(rng.uniform(*profile["K"]))),
                "temperature": round(rng.uniform(*profile["temperature"]), 1),
                "humidity": round(rng.uniform(*profile["humidity"]), 1),
                "ph": round(rng.uniform(*profile["ph"]), 2),
                "rainfall": round(rng.uniform(*profile["rainfall"]), 1),
                "label": crop,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(DATASET_PATH, index=False)
    return df


def load_dataset() -> pd.DataFrame:
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        expected = {"N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"}
        if expected.issubset(set(df.columns)):
            return df
    return build_synthetic_dataset()


def train_and_save_model() -> Dict[str, object]:
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=220, random_state=42)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    return {
        "accuracy": round(float(accuracy), 4),
        "dataset_size": int(len(df)),
        "features": FEATURE_COLUMNS,
        "crops": sorted(list(label_encoder.classes_)),
    }


def load_model_artifacts() -> Tuple[object, object, object]:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH)):
        train_and_save_model()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, scaler, label_encoder


def predict_crop(features: List[float]) -> str:
    model, scaler, label_encoder = load_model_artifacts()
    features_array = np.array([features], dtype=float)
    scaled_features = scaler.transform(features_array)
    prediction = model.predict(scaled_features)[0]
    return str(label_encoder.inverse_transform([prediction])[0])


def get_farming_tips(crop_name: str) -> List[str]:
    tips = {
        "Rice": [
            "Use well-drained fields and maintain consistent irrigation.",
            "Apply nitrogen in split doses for better growth.",
        ],
        "Maize": [
            "Plant in rows with enough spacing for sunlight.",
            "Use balanced fertilizer and monitor pests regularly.",
        ],
        "Banana": [
            "Keep soil moist but avoid waterlogging.",
            "Mulch around plants to preserve moisture.",
        ],
    }
    return tips.get(crop_name, [
        "Use certified seeds and follow soil-test-based fertilization.",
        "Monitor irrigation, weeds, and pests regularly.",
    ])


if __name__ == "__main__":
    summary = train_and_save_model()
    print("Training completed")
    print(summary)
