from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import plotly.graph_objects as go
from plotly.offline import plot

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model
model = joblib.load("xgb_stock_model2.joblib")

expected_features = [
    'Close', 'High', 'Low', 'Open', 'Volume',
    'sma_5', 'sma_10', 'ema_5', 'ema_10',
    'rsi_14', 'macd', 'macd_signal',
    'bb_high', 'bb_low', 'bb_width'
]

def prepare_data_for_prediction(df):
    if 'Adj Close' in df.columns:
        df = df.drop(columns=['Adj Close'])
    return df[expected_features]

def get_features(df):
    df["sma_5"] = SMAIndicator(df["Close"], window=5).sma_indicator()
    df["sma_10"] = SMAIndicator(df["Close"], window=10).sma_indicator()
    df["ema_5"] = EMAIndicator(df["Close"], window=5).ema_indicator()
    df["ema_10"] = EMAIndicator(df["Close"], window=10).ema_indicator()
    df["rsi_14"] = RSIIndicator(df["Close"], window=14).rsi()
    macd = MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    bb = BollingerBands(df["Close"], window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_high"] - df["bb_low"]
    df = df.dropna()
    return df

def plot_stock_and_prediction(df_features, predicted_price, next_date):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_features.index, y=df_features["Close"], 
                             mode="lines", name="Harga Close"))
    fig.add_trace(go.Scatter(
        x=[next_date], y=[predicted_price], mode="markers+text", 
        name="Prediksi Harga Besok", text=[f"{predicted_price:.2f}"], 
        textposition="top center", marker=dict(color="red", size=10)
    ))
    fig.update_layout(title="Grafik Harga Saham dan Prediksi",
                      xaxis_title="Tanggal", yaxis_title="Harga")
    return plot(fig, output_type="div", include_plotlyjs=False)

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "error": None, "graph": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, ticker: str = Form(...)):
    try:
        # Ambil data dengan harga sudah disesuaikan
        df_raw = yf.download(ticker, period="6mo", auto_adjust=True)
        
        # Bersihkan data
        df_clean = df_raw[~df_raw.index.duplicated(keep="last")].sort_index()
        if isinstance(df_clean.columns, pd.MultiIndex):
            df_clean.columns = [col[0] for col in df_clean.columns]

        # Buat fitur teknikal
        df_features = get_features(df_clean.copy())
        if df_features.empty:
            return templates.TemplateResponse("index.html", {
                "request": request, "result": None,
                "error": "Data tidak cukup untuk prediksi.", "graph": None
            })

        # Data terakhir untuk prediksi
        X_new = prepare_data_for_prediction(df_features).iloc[-1:]
        if X_new.empty:
            return templates.TemplateResponse("index.html", {
                "request": request, "result": None,
                "error": "Data terakhir tidak cukup untuk prediksi.", "graph": None
            })

        # Prediksi
        y_pred_return = model.predict(X_new)
        predicted_price = X_new["Close"].values[0] * (1 + y_pred_return[0])

        # Tanggal prediksi (sesuai data terakhir df_features)
        last_date = df_features.index[-1]
        next_date = last_date + pd.Timedelta(days=1)

        # Grafik
        graph_html = plot_stock_and_prediction(df_features, predicted_price, next_date)

        # Hasil teks
        result_text = (
            f"Prediksi harga penutupan {ticker.upper()} untuk tanggal "
            f"{next_date.date()} adalah: Rp{predicted_price:,.2f}"
        )

        return templates.TemplateResponse("index.html", {
            "request": request, "result": result_text,
            "error": None, "graph": graph_html
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, "result": None,
            "error": str(e), "graph": None
        })
