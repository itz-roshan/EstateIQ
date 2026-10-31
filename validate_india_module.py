from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

from india_comparable import estimate_india_comparable

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Location Based Price Prediction using ANN" / "data" / "house_data.csv"
OUT = ROOT / "Location Based Price Prediction using ANN" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _cohort_prediction(row: pd.Series, city_groups, locality_groups, global_train: pd.DataFrame):
    city = str(row["City"])
    locality = str(row["Location"])
    bedrooms = int(row["No. of Bedrooms"])
    area = float(row["Area"])
    city_rows = city_groups.get(city, global_train.iloc[0:0])
    locality_rows = locality_groups.get((city, locality), global_train.iloc[0:0])
    low, high = area * 0.75, area * 1.25
    candidates = [
        ("Same locality, bedroom and area", locality_rows[
            (locality_rows["No. of Bedrooms"] == bedrooms)
            & locality_rows["Area"].between(low, high)
        ], 3),
        ("Same city, bedroom and area", city_rows[
            (city_rows["No. of Bedrooms"] == bedrooms)
            & city_rows["Area"].between(low, high)
        ], 10),
        ("Same locality and area (bedroom mix)", locality_rows[
            locality_rows["Area"].between(low, high)
        ], 3),
        ("Same city and area (bedroom mix)", city_rows[
            city_rows["Area"].between(low, high)
        ], 10),
        ("Same city and bedroom (area mix)", city_rows[
            city_rows["No. of Bedrooms"] == bedrooms
        ], 10),
        ("Same locality (all available listings)", locality_rows, 3),
        ("Same city (all available listings)", city_rows, 1),
        ("All India listings (fallback)", global_train, 1),
    ]
    for _, candidates_df, minimum in candidates:
        if len(candidates_df) >= minimum:
            return float(candidates_df["Price_per_sqft"].median()) * area
    return float(global_train["Price_per_sqft"].median()) * area


def main() -> None:
    data = pd.read_csv(DATA)
    amenity_cols = [
        c for c in data.columns
        if c not in {"Price", "Area", "Location", "No. of Bedrooms", "Resale", "City"}
    ]
    amenity_unknown_pct = float((data[amenity_cols] == 9).to_numpy().mean() * 100)

    screenshot = estimate_india_comparable(
        data,
        city="Bangalore",
        location="5th Stage Raja Rajeshwari Nagar",
        area_sqft=1300,
        bedrooms=5,
        resale=0,
    )
    assert screenshot["sample_count"] == 5, screenshot
    assert screenshot["bedroom_match_count"] == 0, screenshot
    assert screenshot["area_out_of_support"] is True, screenshot
    assert abs(screenshot["estimate"] - 4_906_200) < 1, screenshot

    # Strict untouched 15% hold-out; comparable pools use training rows only.
    clean = data.dropna(subset=["City", "Location", "Price", "Area", "No. of Bedrooms"]).copy()
    clean["City"] = clean["City"].astype(str).str.strip()
    clean["Location"] = clean["Location"].astype(str).str.strip()
    for col in ["Price", "Area", "No. of Bedrooms"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna(subset=["Price", "Area", "No. of Bedrooms"])
    clean = clean[(clean["Price"] > 0) & (clean["Area"] > 0)].copy()
    clean["No. of Bedrooms"] = clean["No. of Bedrooms"].round().astype(int)
    clean["Price_per_sqft"] = clean["Price"] / clean["Area"]

    train_idx, test_idx = train_test_split(
        np.arange(len(clean)), test_size=0.15, random_state=42
    )
    train = clean.iloc[train_idx].copy().reset_index(drop=True)
    test = clean.iloc[test_idx].copy()
    city_groups = {k: v for k, v in train.groupby("City", sort=False)}
    locality_groups = {
        k: v for k, v in train.groupby(["City", "Location"], sort=False)
    }

    preds = np.array([
        _cohort_prediction(row, city_groups, locality_groups, train)
        for _, row in test.iterrows()
    ], dtype=float)
    actual = test["Price"].to_numpy(dtype=float)

    metrics = {
        "method": "Comparable median price per sq ft × area",
        "rows_total": int(len(clean)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "MAE_INR": float(mean_absolute_error(actual, preds)),
        "RMSE_INR": float(mean_squared_error(actual, preds) ** 0.5),
        "R2": float(r2_score(actual, preds)),
        "MedianAE_INR": float(median_absolute_error(actual, preds)),
        "MAPE_percent": float(np.mean(np.abs(actual - preds) / actual) * 100),
        "amenity_unknown_percent": amenity_unknown_pct,
    }
    pd.DataFrame([metrics]).to_csv(OUT / "comparable_test_metrics.csv", index=False)
    print("Screenshot regression test: PASS")
    print(f"Screenshot reference: ₹{screenshot['estimate']:,.0f}")
    print(f"Screenshot comparable group: {screenshot['method']} ({screenshot['sample_count']} rows)")
    print(f"Selected-bedroom matches: {screenshot['bedroom_match_count']}")
    print(f"5-bedroom Bangalore 5th–95th area range: {screenshot['bedroom_area_support_low']:,.0f}–{screenshot['bedroom_area_support_high']:,.0f} sq ft")
    print("Held-out comparable evaluation:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
