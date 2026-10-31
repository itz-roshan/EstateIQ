"""Transparent locality-comparable reference estimator for EstateIQ India module.

The source ANN is retained as an academic artifact but is not used for a headline
price because its saved held-out R² is negative. This estimator uses observed
listing prices per square foot from comparable records and reports sample support.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def estimate_india_comparable(
    listings: pd.DataFrame,
    *,
    city: str,
    location: str,
    area_sqft: float,
    bedrooms: int,
    resale: int | None = None,
    area_band: float = 0.25,
    min_locality: int = 3,
    min_city: int = 10,
) -> dict[str, Any]:
    """Return a median price-per-sq-ft reference from a transparent comp cohort.

    Cohort priority:
      1. Same locality + same bedrooms + similar area (>=3 rows)
      2. Same city + same bedrooms + similar area (>=10 rows)
      3. Same locality + similar area (>=3 rows)
      4. Same city + similar area (>=10 rows)
      5. Same city + same bedrooms (>=10 rows)
      6. Same locality (>=3 rows)
      7. Same city; then global data as a final fallback.

    Amenities and resale are intentionally not used as price adjustments here:
    amenity attributes are mostly unreported, and several cohorts are too small.
    The UI must disclose that limitation instead of implying unsupported premiums.
    """
    required = {"City", "Location", "Price", "Area", "No. of Bedrooms"}
    missing = required.difference(listings.columns)
    if missing:
        raise ValueError(f"India listings dataset is missing columns: {sorted(missing)}")

    area_sqft = float(area_sqft)
    bedrooms = int(bedrooms)
    if not np.isfinite(area_sqft) or area_sqft <= 0:
        raise ValueError("Built-up area must be a positive number.")

    frame = listings[["City", "Location", "Price", "Area", "No. of Bedrooms"]].copy()
    frame["City"] = frame["City"].astype("string").str.strip()
    frame["Location"] = frame["Location"].astype("string").str.strip()
    for col in ["Price", "Area", "No. of Bedrooms"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["City", "Location", "Price", "Area", "No. of Bedrooms"])
    frame = frame[(frame["Price"] > 0) & (frame["Area"] > 0)].copy()
    frame["No. of Bedrooms"] = frame["No. of Bedrooms"].round().astype(int)
    frame["Price_per_sqft"] = frame["Price"] / frame["Area"]

    city_rows = frame[frame["City"].astype(str) == str(city)].copy()
    local_rows = city_rows[city_rows["Location"].astype(str) == str(location)].copy()
    low_area = area_sqft * (1.0 - area_band)
    high_area = area_sqft * (1.0 + area_band)

    city_bed_rows = city_rows[city_rows["No. of Bedrooms"] == bedrooms]
    city_bed_area = city_bed_rows[city_bed_rows["Area"].between(low_area, high_area)]
    local_bed_area = local_rows[
        (local_rows["No. of Bedrooms"] == bedrooms)
        & local_rows["Area"].between(low_area, high_area)
    ]
    local_area = local_rows[local_rows["Area"].between(low_area, high_area)]
    city_area = city_rows[city_rows["Area"].between(low_area, high_area)]

    # A bedroom/area combination is out of support if the selected bedroom class
    # has enough city records to define a range and the requested area falls outside it.
    if len(city_bed_rows) >= 5:
        support_low = float(city_bed_rows["Area"].quantile(0.05))
        support_high = float(city_bed_rows["Area"].quantile(0.95))
        area_out_of_support = not (support_low <= area_sqft <= support_high)
    else:
        support_low = float(city_bed_rows["Area"].min()) if len(city_bed_rows) else None
        support_high = float(city_bed_rows["Area"].max()) if len(city_bed_rows) else None
        area_out_of_support = True

    cohorts = [
        ("Same locality, bedroom and area", local_bed_area, min_locality),
        ("Same city, bedroom and area", city_bed_area, min_city),
        ("Same locality and area (bedroom mix)", local_area, min_locality),
        ("Same city and area (bedroom mix)", city_area, min_city),
        ("Same city and bedroom (area mix)", city_bed_rows, min_city),
        ("Same locality (all available listings)", local_rows, min_locality),
        ("Same city (all available listings)", city_rows, 1),
        ("All India listings (fallback)", frame, 1),
    ]

    selected = None
    method = ""
    for name, candidate, minimum in cohorts:
        if len(candidate) >= minimum:
            selected = candidate.copy()
            method = name
            break
    if selected is None or selected.empty:
        raise ValueError("No usable comparable listing rows were found.")

    ppsf = selected["Price_per_sqft"].replace([np.inf, -np.inf], np.nan).dropna()
    if ppsf.empty:
        raise ValueError("Comparable listings do not contain valid price-per-square-foot values.")

    median_ppsf = float(ppsf.median())
    p10_ppsf = float(ppsf.quantile(0.10))
    p90_ppsf = float(ppsf.quantile(0.90))
    bedroom_matches = int((selected["No. of Bedrooms"] == bedrooms).sum())
    bedroom_match_rate = bedroom_matches / len(selected)

    return {
        "estimate": median_ppsf * area_sqft,
        "lower_estimate": p10_ppsf * area_sqft,
        "upper_estimate": p90_ppsf * area_sqft,
        "median_ppsf": median_ppsf,
        "p10_ppsf": p10_ppsf,
        "p90_ppsf": p90_ppsf,
        "sample_count": int(len(selected)),
        "method": method,
        "bedroom_match_count": bedroom_matches,
        "bedroom_match_rate": float(bedroom_match_rate),
        "locality_count": int(len(local_rows)),
        "locality_bedroom_area_count": int(len(local_bed_area)),
        "city_bedroom_count": int(len(city_bed_rows)),
        "city_bedroom_area_count": int(len(city_bed_area)),
        "area_out_of_support": bool(area_out_of_support),
        "bedroom_area_support_low": support_low,
        "bedroom_area_support_high": support_high,
        "selected_comp_areas_low": float(selected["Area"].quantile(0.10)),
        "selected_comp_areas_high": float(selected["Area"].quantile(0.90)),
        "resale_filter_applied": False,
        "amenities_used": False,
        "confidence": "Low" if len(selected) < 10 or area_out_of_support else "Moderate",
    }
