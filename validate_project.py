"""Lightweight integrity checks for the EstateIQ project (TensorFlow not required)."""
from __future__ import annotations

import json
import sys
import zipfile
import tomllib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent

ASSETS = {
    "US house dataset": ROOT / "Property prize prediction/data/House Prices.csv",
    "US ANN model": ROOT / "Property prize prediction/house_price_ann.keras",
    "US preprocessor": ROOT / "Property prize prediction/models/house_price_preprocessor.joblib",
    "India listings dataset": ROOT / "Location Based Price Prediction using ANN/data/house_data.csv",
    "India ANN model": ROOT / "Location Based Price Prediction using ANN/location_based_price_ann.keras",
    "India preprocessor": ROOT / "Location Based Price Prediction using ANN/models/location_price_preprocessor.joblib",
    "Room CNN model": ROOT / "Property Image Classification Using CNN/property_cnn_model.keras",
    "Demand dataset": ROOT / "Property demand forecast/data/property_demand_forecasting_synthetic.csv",
    "Demand LSTM model": ROOT / "Property demand forecast/property_demand_lstm.keras",
    "Review RNN model": ROOT / "Customer_Review_Sentiment_Analysis_RNN/customer_review_rnn.keras",
    "Review tokenizer": ROOT / "Customer_Review_Sentiment_Analysis_RNN/customer_review_tokenizer.json",
    "Review dataset": ROOT / "Customer_Review_Sentiment_Analysis_RNN/data/Reviews.csv",
}

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"[PASS] {label}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"[FAIL] {label}" + (f" — {detail}" if detail else ""))


def keras_shapes(path: Path) -> tuple[list[int], int | None]:
    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member:
            raise ValueError(f"Corrupt archive member: {corrupt_member}")
        config = json.loads(archive.read("config.json"))
    layers = config.get("config", {}).get("layers", [])
    input_shape = []
    if layers:
        input_shape = layers[0].get("config", {}).get("batch_shape", []) or []
    dense_units = [
        layer.get("config", {}).get("units")
        for layer in layers
        if layer.get("class_name") == "Dense"
    ]
    return input_shape, (dense_units[-1] if dense_units else None)


print("EstateIQ · Project integrity check")
print("=" * 42)
for label, path in ASSETS.items():
    check(f"{label} exists", path.is_file(), str(path.relative_to(ROOT)))

# Dataset checks.
us_path = ASSETS["US house dataset"]
india_path = ASSETS["India listings dataset"]
demand_path = ASSETS["Demand dataset"]
if us_path.is_file() and india_path.is_file() and demand_path.is_file():
    us = pd.read_csv(us_path)
    india = pd.read_csv(india_path)
    demand = pd.read_csv(demand_path)
    check("US dataset is readable", len(us) > 0 and {"Date", "Price", "zipcode", "Yr_built", "Yr_renovated", "Lat", "Long"}.issubset(us.columns), f"{len(us):,} rows")
    check("India dataset is readable", len(india) > 0 and {"Price", "Area", "Location", "No. of Bedrooms", "Resale", "City", "MaintenanceStaff"}.issubset(india.columns), f"{len(india):,} rows · {india['City'].nunique()} cities")
    check("Demand dataset is readable", len(demand) >= 17 and {"Date", "Sales_Count"}.issubset(demand.columns), f"{len(demand):,} monthly rows")
else:
    us = india = demand = None
    check("Core datasets readable", False, "One or more CSV files are missing")

# The app forms must transform to exactly the feature width expected by each ANN.
try:
    pre_path = ASSETS["US preprocessor"]
    model_path = ASSETS["US ANN model"]
    if pre_path.is_file() and model_path.is_file() and us is not None:
        pre = joblib.load(pre_path)
        row = us.head(1).copy()
        row["zipcode"] = row["zipcode"].astype(str)
        dates = pd.to_datetime(row["Date"].astype(str), format="%Y%m%dT%H%M%S", errors="coerce")
        row["sale_year"] = dates.dt.year
        row["sale_month"] = dates.dt.month
        row["house_age_at_sale"] = (row["sale_year"] - row["Yr_built"]).clip(lower=0)
        row["has_renovation"] = (row["Yr_renovated"] > 0).astype(int)
        row["years_since_renovation"] = np.where(row["Yr_renovated"] > 0, (row["sale_year"] - row["Yr_renovated"]).clip(lower=0), 0)
        row = row.drop(columns=["ID", "Date", "Price"], errors="ignore")
        transformed = pre.transform(row)
        input_shape, output_units = keras_shapes(model_path)
        check("US ANN preprocessor ↔ model input", transformed.shape[1] == input_shape[-1] == 92, f"{transformed.shape[1]} features → model expects {input_shape[-1]}")
        check("US ANN output shape", output_units == 1, f"{output_units} output unit(s)")
except Exception as exc:
    check("US ANN pipeline test", False, f"{type(exc).__name__}: {exc}")

try:
    pre_path = ASSETS["India preprocessor"]
    model_path = ASSETS["India ANN model"]
    if pre_path.is_file() and model_path.is_file() and india is not None:
        pre = joblib.load(pre_path)
        excluded = {"Price", "Area", "Location", "No. of Bedrooms", "Resale", "City", "City_Location", "Price_per_Area", "Amenity_Yes_Count", "Amenity_Unknown_Count"}
        amenity_cols = [col for col in india.columns if col not in excluded]
        city = str(india["City"].dropna().iloc[0])
        location = str(india.loc[india["City"].astype(str) == city, "Location"].dropna().iloc[0])
        selected = [x for x in ["CarParking", "Gymnasium", "LiftAvailable", "24X7Security"] if x in amenity_cols]
        unavailable = [col for col in ["SwimmingPool"] if col in amenity_cols]
        unknown = [col for col in amenity_cols if col not in selected and col not in unavailable]
        values = {"Area": 1200.0, "No. of Bedrooms": 2, "Resale": 0, "Amenity_Yes_Count": len(selected), "Amenity_Unknown_Count": len(unknown), "City": city, "City_Location": f"{city} | {location}"}
        values.update({col: (1 if col in selected else 0 if col in unavailable else np.nan) for col in amenity_cols})
        transformed = pre.transform(pd.DataFrame([values]))
        input_shape, output_units = keras_shapes(model_path)
        check("India ANN preprocessor ↔ model input", transformed.shape[1] == input_shape[-1] == 496, f"{transformed.shape[1]} features → model expects {input_shape[-1]}")
        check("India ANN output shape", output_units == 1, f"{output_units} output unit(s)")
except Exception as exc:
    check("India ANN pipeline test", False, f"{type(exc).__name__}: {exc}")

# CNN/LSTM/RNN model input/output contracts.
contracts = [
    ("Room CNN", ASSETS["Room CNN model"], [None, 128, 128, 3], 5),
    ("Demand LSTM", ASSETS["Demand LSTM model"], [None, 12, 1], 1),
    ("Review RNN", ASSETS["Review RNN model"], [None, 120], 3),
]
for label, path, expected_input, expected_output in contracts:
    if path.is_file():
        try:
            actual_input, actual_output = keras_shapes(path)
            check(f"{label} input contract", actual_input == expected_input, f"saved input shape: {actual_input}")
            check(f"{label} output contract", actual_output == expected_output, f"saved output units: {actual_output}")
        except Exception as exc:
            check(f"{label} archive integrity", False, f"{type(exc).__name__}: {exc}")

# Tokenizer and image-category folder checks.
reviews_path = ASSETS["Review dataset"]
if reviews_path.is_file():
    try:
        review_sample = pd.read_csv(reviews_path, nrows=5)
        required_review_cols = {"Summary", "Text", "Score"}
        check("Review dataset sample is readable", required_review_cols.issubset(review_sample.columns), f"sample columns: {len(review_sample.columns)}")
    except Exception as exc:
        check("Review dataset sample is readable", False, f"{type(exc).__name__}: {exc}")

tokenizer_path = ASSETS["Review tokenizer"]
if tokenizer_path.is_file():
    try:
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        check("Review tokenizer JSON is readable", isinstance(tokenizer, dict) and bool(tokenizer))
    except Exception as exc:
        check("Review tokenizer JSON is readable", False, str(exc))

image_root = ROOT / "Property Image Classification Using CNN/House_Room_Dataset"
expected_classes = ["Bathroom", "Bedroom", "Dinning", "Kitchen", "Livingroom"]
if image_root.is_dir():
    class_counts = {name: sum(1 for f in (image_root / name).iterdir() if f.is_file()) if (image_root / name).is_dir() else 0 for name in expected_classes}
    check("Five room-image categories present", all(class_counts.values()), ", ".join(f"{k}: {v:,}" for k, v in class_counts.items()))
    for name in expected_classes:
        folder = image_root / name
        first_image = next((f for f in folder.iterdir() if f.is_file()), None) if folder.is_dir() else None
        try:
            if first_image:
                with Image.open(first_image) as image:
                    image.verify()
                image_ok = True
            else:
                image_ok = False
            check(f"{name} sample image opens", image_ok)
        except Exception as exc:
            check(f"{name} sample image opens", False, str(exc))
else:
    check("Room image dataset present", False)

# Streamlit theme config: this keeps the app readable even with a dark system theme.
try:
    theme_config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text(encoding="utf-8"))
    theme_ok = (
        theme_config.get("theme", {}).get("base") == "light"
        and theme_config.get("theme", {}).get("textColor") == "#15213B"
        and theme_config.get("theme", {}).get("sidebar", {}).get("backgroundColor") == "#101A33"
    )
    check("High-contrast Streamlit theme config", theme_ok)
except Exception as exc:
    check("High-contrast Streamlit theme config", False, str(exc))

# Python syntax check for the Streamlit entrypoint.
try:
    compile((ROOT / "app.py").read_text(encoding="utf-8"), str(ROOT / "app.py"), "exec")
    check("Streamlit app syntax", True)
except Exception as exc:
    check("Streamlit app syntax", False, str(exc))

print("=" * 42)
print(f"Result: {passed} passed · {failed} failed")
if failed:
    sys.exit(1)
print("All available offline integrity checks passed.")
