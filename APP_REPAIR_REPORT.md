ESTATEIQ APPLICATION REPAIR REPORT
Date: 2026-10-31

SUMMARY
- The original UI theme, page layout, navigation and visual cards are preserved.
- Fixed a US ANN input bug that could create astronomical displayed prices.
- Prevented invalid renovation years from entering the model.
- Added training-range guards and model/preprocessor compatibility checks.
- Removed the temporary raw-output debug banner.
- Improved repeat-inference responsiveness and model cache invalidation.
- Room Photo Checker now avoids presenting weak, close predictions as confirmed labels.
- India ANN remains explicitly experimental because its test score is below the median baseline.

US PRICE MODEL ROOT CAUSE
The form originally accepted any integer from 0 to 2015 for "Year renovated". The field is a calendar year, not a Boolean flag. A value of 1 was interpreted as year 1 AD. The derived "years_since_renovation" became about 2,014 years and the standardized input became an extreme outlier. The trained ANN therefore returned a raw log-price near 73.7; expm1() converted that into an astronomical dollar amount.

A reproducible forward pass using the saved preprocessor and saved ANN weights gave approximately:
- Year renovated = 0 (never): raw log output 12.48, back-transformed price about $263K for the screenshot-like inputs.
- Year renovated = 1: raw log output 73.75, an invalid out-of-distribution result.

FIX
- The renovation field is now a dropdown containing "Never renovated" (stored as 0) and actual calendar years present in the training data.
- The submitted year must be between the build year and sale year.
- The app blocks predictions outside a broad range derived from the training target distribution.
- The temporary Debug: Raw ANN Output banner was removed.

INFERENCE AND PERFORMANCE
- Cached Keras models and preprocessing pipelines now use file path, modification timestamp and size in the cache key. Replacing a model invalidates the cache.
- Small prediction batches use direct model(inputs, training=False) calls, avoiding the tf.data setup overhead of model.predict().
- Model/preprocessor input feature dimensions are checked before inference.

ROOM PHOTO CHECKER
Independent test set results for the supplied retrained CNN:
- Test images: 777
- Accuracy: 53.7%
- Macro F1: 0.531
- Living room recall: 60.1%
- Bedroom recall: 31.7%

The room classifier is not yet production-grade. The exact held-out sample `living_31.jpg` (folder label: Livingroom) is predicted as Dinning at 53.1% by the current CNN. The UI now confirms a label only when its top model score is >= 62% and its lead over the second choice is >= 10 percentage points. This sample is therefore shown as uncertain rather than a confirmed Dining room. Scores are not calibrated probabilities.

INDIA PRICE MODEL
- Saved held-out R²: -0.0645 (approximately -0.065).
- This is worse than the simple median baseline. The app retains the academic ANN output for project demonstration, but labels it as experimental and shows the reliability warning. The locality median remains available as a separate historical dataset reference.

OTHER MODULES
- Demand forecast data is synthetic and is labelled as a demonstration forecast.
- Review sentiment model output is a model classification, not a guarantee of sentiment.
- The app's original styling and page names have not been redesigned.

CHECKS RUN IN THIS WORKSPACE
PASS: app.py Python syntax compilation.
PASS: US saved preprocessor exposes the expected 23 raw input columns.
PASS: US transformed input dimension matches the saved ANN's 92 inputs.
PASS: Screenshot-like US input with renovation year 0 produces a plausible raw log-price.
PASS: Screenshot-like US input with renovation year 1 reproduces the out-of-distribution raw output and is now rejected by the form guard.
PASS: India saved preprocessor exposes the expected raw input columns and saved model has its separate fitted feature dimension.
PASS: Room CNN metadata confirms 96x96 RGB input and five classes in the same order used by app.py.
PASS: Room CNN independent test report and confusion matrix files are included.

LIMITATIONS
- TensorFlow was not installed in the build workspace, so a live TensorFlow-backed Streamlit run could not be performed here. Saved model weights were inspected and US numerical inference was reproduced using the saved weights and fitted scikit-learn preprocessor.
- The room classifier's moderate test accuracy means some real photographs will still be uncertain or misclassified. More consistently labelled, diverse images and transfer learning are needed for a genuinely strong model.
- The India ANN's negative R² means its numerical output should not be used as a real valuation.
