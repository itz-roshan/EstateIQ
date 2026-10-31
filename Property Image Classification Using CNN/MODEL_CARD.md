# EstateIQ Room Photo Checker — Model Card

## Purpose

Classify a property/interior photograph into one of five educational room categories: Bathroom, Bedroom, Dining room (`Dinning` folder name), Kitchen, or Living room (`Livingroom` folder name).

## Model and input contract

- Architecture: compact residual CNN, `EstateIQ_Room_CNN_v2`.
- Input: RGB image resized to 96 × 96.
- Pixel range at the application boundary: float values in 0–255.
- Pixel scaling: included inside the saved Keras model (divide by 255 exactly once).
- Output: five softmax values in the fixed class order stored in `outputs/class_names.json`.
- App inference: uses the same RGB conversion, 96 × 96 resize and raw-pixel input contract as training.
- Uncertainty handling: a room label is confirmed only when the top score is at least 0.62 and exceeds the second score by at least 0.10; otherwise the app labels it `Uncertain · possible [class]`.

## Dataset audit

- Raw images: 5,250.
- Usable unique images after exact-duplicate audit: 5,174.
- Split: 3,621 train / 776 validation / 777 independent test.
- Exact duplicate groups audited: 58.
  - 40 same-class duplicate groups: one copy retained.
  - 18 exact duplicate groups with conflicting folder labels: all copies excluded.
- Unreadable image files: 0.

## Independent test results

| Metric | Score |
|---|---:|
| Accuracy | 53.7% |
| Macro F1 | 0.531 |
| Weighted F1 | 0.526 |

| Class | Precision | Recall | F1 | Test images |
|---|---:|---:|---:|---:|
| Bathroom | 0.469 | 0.744 | 0.575 | 90 |
| Bedroom | 0.573 | 0.317 | 0.408 | 186 |
| Dining room | 0.640 | 0.696 | 0.667 | 171 |
| Kitchen | 0.551 | 0.415 | 0.474 | 142 |
| Living room | 0.475 | 0.601 | 0.531 | 188 |

The earlier baseline scored 41.7% accuracy and 28.7% Living room recall on this same fixed, deduplicated test split. The updated model improved overall accuracy by 12.0 percentage points and Living room recall by 31.4 percentage points.

## Limitations

- Test accuracy is still moderate; Bedroom recall is especially weak. The model should not be described as highly accurate or production-ready.
- The source data includes open-plan and adjacent-room scenes. In such images, the distinction between living and dining areas can be subjective.
- Folder labels may contain residual mistakes beyond the exact-duplicate audit; a visual review of every image was not possible.
- Softmax output scores are not calibrated probabilities. A 60% score does not mean a 60% real-world chance of correctness.
- Performance on user-uploaded photos from different cameras, countries, lighting, or property styles may differ from this test split.

## Reproducibility artifacts

- `property_image_code.ipynb`: corrected TensorFlow/Keras training and evaluation notebook.
- `outputs/data_split.csv`: fixed split manifest.
- `outputs/duplicate_audit.csv`: exact duplicate audit.
- `outputs/test_classification_report.csv`: per-class test scores.
- `outputs/test_confusion_matrix.csv`: confusion matrix.
- `outputs/test_predictions.csv`: per-image held-out predictions.
- `outputs/training_history.csv`: training/validation history.
- `outputs/TRAINING_REPORT.md`: concise result report.
