# EstateIQ — AI-Powered Real Estate Intelligence Platform

EstateIQ is a local Streamlit learning project that combines property price estimation, room-photo classification, demand forecasting, sentiment analysis and dataset exploration. The existing dashboard layout, theme and navigation are preserved in this updated package.

## Run on Windows (Python 3.12)

Open PowerShell in this folder and install the pinned runtime packages:

```powershell
py -3.12 -m pip install -r requirements.txt
```

Then launch Streamlit:

```powershell
py -3.12 -m streamlit run app.py --global.developmentMode=false --server.port=8502 --browser.serverPort=8502
```

Open `http://localhost:8502` and keep the terminal running while you use the app.

### If you use the shared package folder from the earlier setup

The app can run without a virtual environment. Set the package folder for the current PowerShell session, then start Streamlit:

```powershell
$env:PYTHONPATH = "C:\EstateIQ_Packages"
py -3.12 -m streamlit run app.py --global.developmentMode=false --server.port=8502 --browser.serverPort=8502
```

The `requirements.txt` pins PyArrow to a NumPy-1.x-compatible version to avoid the earlier `pyarrow requires NumPy 2.0 or newer` conflict.

## Room Photo Checker update

The CNN model and the inference preprocessing have been updated together. The model expects an RGB image resized to **96 × 96**, with raw pixel values from 0 to 255; the saved model performs its own 1/255 scaling. The app sends images in that same format.

The output order is fixed as:

1. Bathroom
2. Bedroom
3. Dinning (shown in the interface as Dining room)
4. Kitchen
5. Livingroom (shown as Living room)

A room label is confirmed only when the top model score is at least 62% and leads the second choice by at least 10 percentage points. Otherwise the app displays **Uncertain · possible [room]**. Model scores are not calibrated probabilities.

The US renovation input is a real calendar year: select **Never renovated** for homes that have not been renovated, or choose a year between construction and sale. Do not enter `1` as a Yes/No value.

See `Property Image Classification Using CNN/outputs/TRAINING_REPORT.md` and `MODEL_CARD.md` for the dataset audit and independent test results. The current model is an academic/demo classifier, not a guaranteed room recognizer. For example, the held-out `living_31.jpg` sample is misclassified as Dining at 53.1%, so the app intentionally labels it uncertain rather than pretending to be sure.

To run the local structural/inference check after installing dependencies:

```powershell
py -3.12 validate_model_assets.py
```

## Project folders

- `Property Image Classification Using CNN/` — room images, trained model, notebook and test outputs.
- `Property prize prediction/` — general house-price prediction module.
- `Location Based Price Prediction using ANN/` — India location-based price estimation module.
- `Property demand forecast/` — demand forecasting module and sample data.
- `Customer_Review_Sentiment_Analysis_RNN/` — customer review sentiment module.

## Latest application safeguards

- Model and preprocessor caches are keyed by file path, modification time and file size, so replacing a model does not silently keep using the old cached object.
- Small-batch inference uses direct Keras calls instead of the slower `model.predict()` data-adapter path.
- US price output is checked against the training-data range; implausible values are blocked instead of shown as astronomical prices.
- The India ANN is clearly marked as experimental because its held-out R² is negative and it underperforms a median baseline.

See `APP_REPAIR_REPORT.md` for the root-cause checks and validation results.

## Important note

The project is educational. Historical/synthetic data and saved model scores are not professional real-estate valuation advice. Predictions and classifications should be interpreted with the limitations documented in each module.
