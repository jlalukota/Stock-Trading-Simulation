import pandas as pd
import yfinance as yf
from google.cloud import storage

from config import BUCKET_NAME, DATA_FILE, TICKERS


def _gcs_client():
    return storage.Client()


def download_blob(blob_name: str) -> str:
    client = _gcs_client()
    return client.bucket(BUCKET_NAME).blob(blob_name).download_as_text()


def upload_blob(blob_name: str, data: str, content_type: str = "text/csv") -> None:
    client = _gcs_client()
    client.bucket(BUCKET_NAME).blob(blob_name).upload_from_string(
        data,
        content_type=content_type,
    )


def to_long_format(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Converts yfinance MultiIndex OHLCV data into long format:
    Datetime, Ticker, Close, High, Low, Open, Volume
    """
    if raw.empty:
        raise ValueError("yfinance returned an empty DataFrame.")

    raw.index.name = "Datetime"

    # yfinance returns columns like: (Price, Ticker)
    df = raw.stack(level=1, future_stack=True).reset_index()

    # Depending on yfinance version, ticker column may be named differently
    if "Ticker" not in df.columns:
        possible_ticker_cols = ["level_1", "Ticker", "Price"]
        for col in possible_ticker_cols:
            if col in df.columns:
                df.rename(columns={col: "Ticker"}, inplace=True)
                break

    required = {"Datetime", "Ticker", "Close", "High", "Low", "Open", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns after reshape: {missing}. Got: {list(df.columns)}")

    df = df[["Datetime", "Ticker", "Close", "High", "Low", "Open", "Volume"]]
    df.dropna(subset=["Close"], inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    df.sort_values(["Ticker", "Datetime"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def fetch_and_upload_ohlcv(period: str = "60d", interval: str = "15m") -> None:
    print("Downloading data from yfinance...")

    raw = yf.download(
        TICKERS,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=True,
    )

    df_long = to_long_format(raw)

    upload_blob(DATA_FILE, df_long.to_csv(index=False))
    print(f"Data uploaded to gs://{BUCKET_NAME}/{DATA_FILE}")


def load_ohlcv(refresh: bool = True) -> pd.DataFrame:
    if refresh:
        fetch_and_upload_ohlcv()

    csv_text = download_blob(DATA_FILE)
    df = pd.read_csv(pd.io.common.StringIO(csv_text), parse_dates=["Datetime"])

    df.sort_values(["Ticker", "Datetime"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df