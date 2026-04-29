import os
import tempfile

BUCKET_NAME = "gabe-jay_stock"
DATA_FILE = "data/yfinance_data.csv"
TRADE_LOG_FILE = "data/trade_log.csv"

TICKERS = [
    "AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "AVGO", "TSLA",
    "BRK-B", "GOOG", "JPM", "LLY", "V", "XOM", "COST", "MA", "UNH",
    "NFLX", "WMT", "PG", "JNJ", "HD", "ABBV", "BAC", "CRM", "KO",
    "ORCL", "CVX", "WFC", "CSCO", "IBM", "PM", "ABT", "ACN", "MRK",
    "MCD", "LIN", "GE", "ISRG", "PEP", "PLTR", "TMO", "DIS", "GS",
    "ADBE", "NOW", "T", "TXN", "QCOM", "VZ", "AMD", "SPGI", "UBER",
    "BKNG", "AXP", "CAT", "RTX", "MS", "AMGN", "INTU", "PGR", "BSX",
    "C", "PFE", "UNP", "NEE", "BLK", "AMAT", "CMCSA", "HON", "SCHW",
    "GILD", "TJX", "LOW", "DHR", "BA", "SYK", "TMUS", "COP",
    "SBUX", "ADP", "PANW", "VRTX", "DE", "ADI", "ETN", "MDT", "BX",
    "BMY", "PLD", "LRCX", "MU", "INTC", "ANET", "KLAC", "CB",
    "SO", "ICE",
]

FEATURE_COLS = ["Close", "Return", "SMA_20", "Momentum", "BollingerWidth"]
TARGET_COL = "FutureReturn"

HOLD_SECONDS = 15 * 60 - 17

# Wire up GCS credentials from env var if needed
if "GCLOUD_CREDENTIALS" in os.environ and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    creds = os.environ["GCLOUD_CREDENTIALS"]
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json")
    tmp.write(creds)
    tmp.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
