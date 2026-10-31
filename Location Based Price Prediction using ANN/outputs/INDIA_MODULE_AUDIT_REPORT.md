# India Property module — audit and correction report

## Executive finding

The previous ANN prediction path is not suitable as a headline price estimator. Its saved held-out evaluation reports **R² = -0.0645**, which is worse than a simple training-median baseline (**R² = -0.0414**). The model can return outputs far outside the source price range when it receives uncommon combinations of city, locality, bedroom count and area.

The application now uses a transparent comparable-listing reference instead of presenting that ANN number as a reliable valuation. The original ANN model, preprocessor and notebook are retained for academic inspection.

## Source-data audit

| Check | Result |
|---|---:|
| Listing rows | 32,963 |
| Cities | 6 |
| Distinct city/locality pairs | 1,790 |
| Amenity columns | 35 |
| Amenity values coded as unknown (`9`) | 69.38% |
| Five-bedroom Bangalore listings | 22 |

The amenity source is very sparse: about 69.4% of values for each amenity are explicitly unknown. Because of this, the app does not infer that an unselected amenity is unavailable, and it does not add or subtract a price premium from amenity selections. This avoids showing misleading price jumps from weak correlations.

## Why the screenshot result was not credible

The screenshot inputs were:

- City: Bangalore
- Locality: 5th Stage Raja Rajeshwari Nagar
- Area: 1,300 sq ft
- Bedrooms: 5

The dataset contains only **5 listings** for that exact locality, all with **2 or 3 bedrooms** and areas between approximately **1,105 and 1,500 sq ft**. It contains 22 five-bedroom Bangalore listings, but their 5th–95th percentile area range is approximately **3,632–9,660 sq ft**. Therefore, 5 bedrooms at 1,300 sq ft is outside observed training support.

The previous ANN returned approximately ₹15,673 for that combination. That is an extrapolation failure, not a credible property estimate.

## Comparable estimator evaluation

The revised method uses the median price per square foot of a matched comparable cohort, multiplied by the requested built-up area. Candidate cohorts are selected in this order:

1. Same locality, same bedroom count and similar area (at least 3 listings).
2. Same city, same bedroom count and similar area (at least 10 listings).
3. Same locality and similar area (at least 3 listings).
4. Same city and similar area (at least 10 listings).
5. Same city and bedroom count, then broader fallbacks.

On an untouched 15% random holdout (4,945 test rows; `random_state=42`), using only the remaining training rows to form comparable groups:

| Metric | Comparable reference |
|---|---:|
| MAE | ₹57.50 lakh |
| Median absolute error | ₹16.87 lakh |
| RMSE | ₹1.92 crore |
| R² | 0.2521 |
| MAPE | 47.33% |

This is a measurable improvement over the old ANN (MAE about ₹82.25 lakh, R² -0.0645), but it is still **not production-grade valuation accuracy**. Large outliers in the data keep RMSE and MAPE high. Outputs must remain labelled as references, not market valuations.

## Corrected screenshot scenario

For Bangalore / 5th Stage Raja Rajeshwari Nagar / 1,300 sq ft / 5 bedrooms:

| Output | Result |
|---|---:|
| Comparable cohort | Same locality and similar area (bedroom mix) |
| Comparable listings | 5 |
| Same-bedroom matches | 0 |
| Median rate | ₹3,774 / sq ft |
| Reference value | **₹49.06 lakh** |
| Indicative 10th–90th percentile range | ₹32.20–₹79.02 lakh |
| Confidence | Low — bedroom/area combination unsupported |

The UI will explicitly warn that the displayed figure is a locality/area reference based on 2–3-bedroom listings, **not an estimate for a verified 5-bedroom property**.

## Expected input behaviour

- **Area:** scales the reference value using comparable price per sq ft.
- **Bedrooms:** used when enough similar-bedroom listings exist. When the selected bedroom/area combination is outside the data range, the app warns and falls back to broader comparables.
- **Amenities:** recorded in the form but deliberately not used as a price adjustment because of the 69.38% unknown rate. Changing Yes/No selections will not create unsupported price increases/decreases.
- **Resale:** recorded in the form but not used as a separate adjustment in this benchmark version.

## Files and verification

- `india_comparable.py`: dependency-light comparable selection and reference calculation.
- `validate_india_module.py`: screenshot regression check and held-out evaluation.
- `comparable_test_metrics.csv`: machine-readable held-out metrics.
- The existing ANN `.keras` file, joblib preprocessor, and original notebook are preserved.

The Python syntax check passed. The screenshot scenario regression check passed and returned ₹49,06,200 with 5 comparable rows, 0 five-bedroom matches, and an out-of-support warning.
