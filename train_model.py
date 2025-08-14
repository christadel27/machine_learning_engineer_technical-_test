import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

# ---------- FUNGSI ----------
def load_and_add_features(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df["Close"] = df["Close"].astype(float)
    df["sma_5"] = SMAIndicator(df["Close"], window=5).sma_indicator()
    df["sma_10"] = SMAIndicator(df["Close"], window=10).sma_indicator()
    df["ema_5"] = EMAIndicator(df["Close"], window=5).ema_indicator()
    df["ema_10"] = EMAIndicator(df["Close"], window=10).ema_indicator()
    df["rsi_14"] = RSIIndicator(df["Close"], window=14).rsi()
    macd = MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    bb = BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_high"] - df["bb_low"]
    df = df.dropna()
    return df

def prepare_data(df):
    df["target_return"] = df["Close"].shift(-1) / df["Close"] - 1
    df = df.dropna()
    X = df.drop(columns=["target_return"])
    y = df["target_return"]
    return train_test_split(X, y, shuffle=False, test_size=0.5)

def train_and_evaluate_with_mlflow(X_train, X_test, y_train, y_test):
    # Hyperparameters
    params = {
        "n_estimators": 300,
        "learning_rate": 0.001,
        "max_depth": 5,
        "random_state": 32
    }

    mlflow.set_experiment("Stock Forecasting XGBoost")
    with mlflow.start_run():
        # Log parameters
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Prediction
        y_pred_return = model.predict(X_test)
        close_test_actual = X_test["Close"]
        pred_price = close_test_actual * (1 + y_pred_return)

        # Metrics (return)
        rmse_return = np.sqrt(mean_squared_error(y_test, y_pred_return))
        mae_return = mean_absolute_error(y_test, y_pred_return)
        r2_return = r2_score(y_test, y_pred_return)

        # Metrics (price)
        rmse_price = np.sqrt(mean_squared_error(close_test_actual, pred_price))
        mae_price = mean_absolute_error(close_test_actual, pred_price)
        r2_price = r2_score(close_test_actual, pred_price)

        # Log metrics
        mlflow.log_metric("rmse_return", rmse_return)
        mlflow.log_metric("mae_return", mae_return)
        mlflow.log_metric("r2_return", r2_return)
        mlflow.log_metric("rmse_price", rmse_price)
        mlflow.log_metric("mae_price", mae_price)
        mlflow.log_metric("r2_price", r2_price)

        # Save model
        mlflow.sklearn.log_model(model, "xgb_model")

        # Plot and save as artifact
        plt.figure(figsize=(12,6))
        plt.plot(close_test_actual.index, close_test_actual, label="Actual Close")
        plt.plot(close_test_actual.index, pred_price, label="Predicted Close")
        plt.title("Actual vs Predicted Closing Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig("pred_vs_actual.png")
        mlflow.log_artifact("pred_vs_actual.png")

        # Save model to local file as well
        joblib.dump(model, "xgb_stock_model2.joblib")

    return model

# ---------- MAIN ----------
if __name__ == "__main__":
    df = load_and_add_features("BBCA.JK", "2010-01-01", "2025-08-01")
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_and_evaluate_with_mlflow(X_train, X_test, y_train, y_test)
