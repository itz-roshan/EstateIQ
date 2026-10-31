"""Quick asset/inference validation for the local EstateIQ project.

Run from the project root with the same Python environment used by Streamlit:
    python validate_model_assets.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent

ASSETS = {
    "US model": ROOT / "Property prize prediction" / "house_price_ann.keras",
    "US preprocessor": ROOT / "Property prize prediction" / "models" / "house_price_preprocessor.joblib",
    "India model": ROOT / "Location Based Price Prediction using ANN" / "location_based_price_ann.keras",
    "India preprocessor": ROOT / "Location Based Price Prediction using ANN" / "models" / "location_price_preprocessor.joblib",
    "Room CNN": ROOT / "Property Image Classification Using CNN" / "property_cnn_model.keras",
    "Demand LSTM": ROOT / "Property demand forecast" / "property_demand_lstm.keras",
    "Review RNN": ROOT / "Customer_Review_Sentiment_Analysis_RNN" / "customer_review_rnn.keras",
    "Review tokenizer": ROOT / "Customer_Review_Sentiment_Analysis_RNN" / "customer_review_tokenizer.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    print(f"PASS  {message}")


def main() -> None:
    print("EstateIQ model asset validation")
    print("Python:", __import__("sys").version.split()[0])
    print("TensorFlow:", tf.__version__)
    for label, path in ASSETS.items():
        require(path.is_file(), f"{label} file exists")

    # US ANN: check raw feature schema, transformed size, and a typical valid input.
    us_pre = joblib.load(ASSETS["US preprocessor"])
    us_model = tf.keras.models.load_model(ASSETS["US model"], compile=False)
    expected_us = list(us_pre.feature_names_in_)
    require(len(expected_us) == us_pre.n_features_in_, "US preprocessor schema is consistent")

    us_row = {
        "Bedrooms": 3, "Bathrooms": 2.25, "Sqft_living": 1910,
        "Sqft_lot": 7600, "Floors": 2.0, "Waterfront": 0,
        "View": 0, "Condition": 3, "Grade": 7, "Sqft_above": 1560,
        "Sqft_basement": 350, "Yr_built": 1975, "Yr_renovated": 0,
        "zipcode": "98002", "Lat": 47.3063, "Long": -122.2150,
        "Sqft_living15": 1840, "Sqft_lot15": 7620, "sale_year": 2015,
        "sale_month": 3, "house_age_at_sale": 40,
        "has_renovation": 0, "years_since_renovation": 0,
    }
    us_frame = pd.DataFrame([us_row]).reindex(columns=expected_us)
    us_ready = us_pre.transform(us_frame).astype(np.float32)
    require(us_ready.shape[1] == int(us_model.input_shape[-1]), "US preprocessor/model input dimensions match")
    us_log = float(np.asarray(us_model(us_ready, training=False)).reshape(-1)[0])
    us_price = float(np.expm1(us_log))
    require(np.isfinite(us_price) and 100_000 <= us_price <= 1_000_000,
            f"US typical input gives a finite historical price (${us_price:,.0f})")

    # India model: verify schema and dimensions, but do not imply its weak estimate is reliable.
    india_pre = joblib.load(ASSETS["India preprocessor"])
    india_model = tf.keras.models.load_model(ASSETS["India model"], compile=False)
    require(len(india_pre.feature_names_in_) == india_pre.n_features_in_, "India preprocessor schema is consistent")
    india_df = pd.read_csv(ROOT / "Location Based Price Prediction using ANN" / "data" / "house_data.csv")
    india_sample = india_df.loc[:, list(india_pre.feature_names_in_)].iloc[[0]].copy()
    amenity_names = [c for c in india_pre.feature_names_in_ if c not in {"Area", "No. of Bedrooms", "Resale", "Amenity_Yes_Count", "Amenity_Unknown_Count", "City", "City_Location"}]
    for col in amenity_names:
        india_sample[col] = pd.to_numeric(india_sample[col], errors="coerce").replace(9, np.nan)
    for col in ["Area", "No. of Bedrooms", "Resale", "Amenity_Yes_Count", "Amenity_Unknown_Count"]:
        india_sample[col] = pd.to_numeric(india_sample[col], errors="coerce")
    india_sample["City"] = india_sample["City"].astype(str)
    india_sample["City_Location"] = india_sample["City_Location"].astype(str)
    india_ready = india_pre.transform(india_sample).astype(np.float32)
    require(india_ready.shape[1] == int(india_model.input_shape[-1]), "India preprocessor/model input dimensions match")
    india_raw = np.asarray(india_model(india_ready, training=False)).reshape(-1)
    require(india_raw.size == 1 and np.isfinite(india_raw[0]), "India ANN returns one finite log-price output")
    india_metrics = pd.read_csv(ROOT / "Location Based Price Prediction using ANN" / "outputs" / "ann_test_metrics.csv")
    r2 = float(india_metrics.loc[0, "R2"])
    print(f"INFO  India ANN held-out R² = {r2:.3f}; keep academic-only warning visible")

    # Room CNN: confirm its contract and run one real image through the new model.
    room_model = tf.keras.models.load_model(ASSETS["Room CNN"], compile=False)
    room_dir = ROOT / "Property Image Classification Using CNN" / "House_Room_Dataset"
    sample_path = room_dir / "Livingroom" / "living_31.jpg"
    require(sample_path.is_file(), "Room test sample is available")
    with Image.open(sample_path) as im:
        image = im.convert("RGB").resize((96, 96), Image.Resampling.LANCZOS)
        batch = np.asarray(image, dtype=np.float32)[None, ...]
    require(tuple(room_model.input_shape[1:]) == (96, 96, 3), "Room CNN accepts 96x96 RGB images")
    room_scores = np.asarray(room_model(batch, training=False)).reshape(-1)
    require(room_scores.shape == (5,), "Room CNN returns five room scores")
    require(np.isfinite(room_scores).all() and abs(float(room_scores.sum()) - 1.0) < 1e-3,
            "Room CNN scores are finite and sum to 1")
    labels = ["Bathroom", "Bedroom", "Dining", "Kitchen", "Living room"]
    best = int(np.argmax(room_scores))
    print(f"INFO  living_31.jpg folder label=Living room; model top match={labels[best]} ({room_scores[best]:.1%})")
    print("INFO  App will report uncertain unless top score >=62% and margin >=10 percentage points")

    demand_model = tf.keras.models.load_model(ASSETS["Demand LSTM"], compile=False)
    require(len(demand_model.input_shape) == 3 and demand_model.input_shape[1:] == (12, 1),
            "Demand LSTM expects a 12-step univariate sequence")
    review_model = tf.keras.models.load_model(ASSETS["Review RNN"], compile=False)
    require(review_model.output_shape[-1] == 3, "Review RNN returns three sentiment classes")

    print("\nAll structural/inference checks passed.")
    print("Note: passing this check verifies installation and input/output contracts, not market-grade prediction accuracy.")


if __name__ == "__main__":
    main()
