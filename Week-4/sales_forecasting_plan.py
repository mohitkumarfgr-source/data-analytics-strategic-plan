"""
Advanced Predictive Analytics Prototype
Sales forecasting with lag/rolling features and Gradient Boosting.
This is a prototype for the Week 4 model-plan deliverable.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42

def make_sample_data(n_days=420):
    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    weekly = 18 * np.sin(2*np.pi*t/7)
    yearly = 10 * np.sin(2*np.pi*t/365)
    trend = 0.06 * t
    promotion = rng.binomial(1, 0.18, n_days)
    price = 100 - 0.4*promotion + rng.normal(0, 1.5, n_days)
    noise = rng.normal(0, 8, n_days)
    sales = 140 + trend + weekly + yearly + 22*promotion - 0.8*(price-100) + noise
    sales = np.maximum(sales, 1)
    return pd.DataFrame({
        "date": dates,
        "sales": sales.round(2),
        "promotion": promotion,
        "price": price.round(2)
    })

def add_features(df):
    x = df.copy().sort_values("date")
    x["dow"] = x["date"].dt.dayofweek
    x["month"] = x["date"].dt.month
    x["weekofyear"] = x["date"].dt.isocalendar().week.astype(int)
    x["dayofyear"] = x["date"].dt.dayofyear
    for lag in [1, 7, 14, 28]:
        x[f"lag_{lag}"] = x["sales"].shift(lag)
    for window in [7, 14, 28]:
        x[f"roll_mean_{window}"] = x["sales"].shift(1).rolling(window).mean()
        x[f"roll_std_{window}"] = x["sales"].shift(1).rolling(window).std()
    x["trend_index"] = np.arange(len(x))
    return x.dropna()

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    return np.mean(2*np.abs(y_pred-y_true)/np.where(denom == 0, 1, denom))*100

def main():
    data = make_sample_data()
    data.to_csv(OUT / "sample_sales_generated.csv", index=False)

    df = add_features(data)
    feature_cols = [
        "promotion", "price", "dow", "month", "weekofyear", "dayofyear",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_std_14",
        "roll_mean_28", "roll_std_28", "trend_index"
    ]

    # Chronological split: never shuffle time-series observations.
    split = int(len(df) * 0.80)
    train, test = df.iloc[:split], df.iloc[split:]
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.04, max_depth=3,
        min_samples_leaf=4, random_state=RANDOM_STATE, loss="huber"
    )
    model.fit(train[feature_cols], train["sales"])
    pred = model.predict(test[feature_cols])

    mae = mean_absolute_error(test["sales"], pred)
    rmse = np.sqrt(mean_squared_error(test["sales"], pred))
    smape_value = smape(test["sales"].to_numpy(), pred)

    result = test[["date", "sales"]].copy()
    result["prediction"] = pred
    result.to_csv(OUT / "predictions.csv", index=False)

    metrics = pd.DataFrame({
        "metric": ["MAE", "RMSE", "SMAPE (%)"],
        "value": [mae, rmse, smape_value]
    })
    metrics.to_csv(OUT / "metrics.csv", index=False)
    dump(model, OUT / "gradient_boosting_sales_model.joblib")

    print("Model evaluation")
    print(metrics.to_string(index=False))
    print(f"\nSaved outputs to: {OUT}")

if __name__ == "__main__":
    main()
