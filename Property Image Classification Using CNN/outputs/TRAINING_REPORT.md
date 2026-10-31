# EstateIQ Room Photo Checker — Training & Evaluation Report

## Dataset and split

- Source images: **5,250**
- Usable images after exact duplicate audit: **5,174**
- Train / validation / independent test: **3,621 / 776 / 777**
- Class order: `Bathroom, Bedroom, Dinning, Kitchen, Livingroom`
- Exact duplicate groups audited: **58**
- Unreadable images: **0**

## Baseline comparison on the same independent test split

The original CNN achieved 41.7% accuracy and 0.375 macro F1 on this exact deduplicated test split. Its Living room recall was 28.7%. The retrained CNN achieved 53.7% accuracy, 0.531 macro F1, and 60.1% Living room recall.

## Independent test results

- Accuracy: **53.7%**
- Macro F1: **0.531**
- Weighted F1: **0.526**
- Best validation epoch: **7**
- Keras export parity max difference: **0.000000**

| Class | Precision | Recall | F1 | Test count |
|---|---:|---:|---:|---:|
| Bathroom | 0.469 | 0.744 | 0.575 | 90 |
| Bedroom | 0.573 | 0.317 | 0.408 | 186 |
| Dinning | 0.640 | 0.696 | 0.667 | 171 |
| Kitchen | 0.551 | 0.415 | 0.474 | 142 |
| Livingroom | 0.475 | 0.601 | 0.531 | 188 |

## Evaluation notes

A deterministic stratified split was created before training: the validation split controls early stopping, and the independent test split is held out from gradient updates. Validation and test loaders are not shuffled so predictions remain aligned with labels.
Exact duplicate images with contradictory folder labels are excluded; duplicate copies within a single class are represented once.
The source dataset has some open-plan/adjacent-room images. Room boundaries can be subjective. Softmax confidence is not calibrated and should not be read as a guaranteed probability.
The app must supply RGB pixels at 96×96 in the 0–255 range; the model performs its own 1/255 rescaling.