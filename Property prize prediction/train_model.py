from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "House Prices.csv"
MODEL_DIR, OUTPUT_DIR = ROOT / "models", ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True); OUTPUT_DIR.mkdir(exist_ok=True)
TARGET = "Price"

def prepare_dataframe(raw):
    df = raw.copy().drop_duplicates()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    parsed = pd.to_datetime(df["Date"].astype(str), format="%Y%m%dT%H%M%S", errors="coerce")
    parsed = parsed.fillna(pd.to_datetime(df["Date"], errors="coerce"))
    df["sale_year"] = parsed.dt.year
    df["sale_month"] = parsed.dt.month
    df["house_age_at_sale"] = (df["sale_year"] - pd.to_numeric(df["Yr_built"], errors="coerce")).clip(lower=0)
    renovated = pd.to_numeric(df["Yr_renovated"], errors="coerce").fillna(0)
    df["has_renovation"] = (renovated > 0).astype(int)
    df["years_since_renovation"] = (df["sale_year"] - renovated.where(renovated > 0)).clip(lower=0)
    return df.drop(columns=["ID", "Date", TARGET], errors="ignore"), df[TARGET].astype(float)

def main():
    raw = pd.read_csv(DATA_PATH)
    X, y = prepare_dataframe(raw)
    X = X.select_dtypes(include=["number"]).copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    Xtr = np.asarray(preprocessor.fit_transform(X_train), dtype=np.float32)
    Xte = np.asarray(preprocessor.transform(X_test), dtype=np.float32)
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Xtr.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(Xtr, y_train.to_numpy(), validation_split=0.2, epochs=150, batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)],
        verbose=1)
    pred = model.predict(Xte, verbose=0).reshape(-1)
    metrics = {"rows_used": int(len(X)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
        "note": "Historical dataset; not a current market valuation."}
    model.save(MODEL_DIR / "house_price_ann.keras")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plt.figure(figsize=(8,5)); plt.plot(history.history["loss"], label="Train"); plt.plot(history.history["val_loss"], label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Training Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150); plt.close()
    plt.figure(figsize=(7,7)); plt.scatter(y_test, pred, alpha=.5)
    low=min(float(y_test.min()), float(pred.min())); high=max(float(y_test.max()), float(pred.max()))
    plt.plot([low,high],[low,high],"r--"); plt.xlabel("Actual Price"); plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted"); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150); plt.close()
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
