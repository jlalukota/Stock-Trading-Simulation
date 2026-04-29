import io
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
    client.bucket(BUCKET_NAME).blob(blob_name).upload_from_string(data, content_type=content_type)


def fetch_and_upload_ohlcv(period: str = "5d", interval: str = "15m") -> None:
    print("Downloading data from yfinance...")
    data = yf.download(TICKERS, period=period, interval=interval)
    upload_blob(DATA_FILE, data.to_csv(index=True))
    print(f"Data uploaded to gs://{BUCKET_NAME}/{DATA_FILE}")


def parse_wide_csv(csv_text: str, n_tickers: int = 100) -> pd.DataFrame:
    """
    Converts the multi-level-header CSV that yfinance produces into a long-format
    DataFrame with columns: Datetime, Ticker, Close, High, Low, Open, Volume.

    yfinance CSV layout:
      Row 0 — "Price", Close×N, High×N, Low×N, Open×N, Volume×N
      Row 1 — "Ticker", t1…tN  (repeated for each metric block)
      Row 2 — "Datetime" label (no values)
      Row 3+ — data rows
    """
    lines = csv_text.splitlines()
    if len(lines) < 4:
        raise ValueError("CSV has fewer than 4 lines — unexpected format.")

    header2 = lines[1].split(",")
    tickers = [t.strip() for t in header2[1 : 1 + n_tickers]]

    metrics = ["Close", "High", "Low", "Open", "Volume"]
    col_names = ["Datetime"]
    for metric in metrics:
        for t in tickers:
            col_names.append(f"{metric}_{t}")

    df = pd.read_csv(
        io.StringIO("\n".join(lines[3:])),
        header=None,
        names=col_names,
        parse_dates=["Datetime"],
    )

    frames = []
    for t in tickers:
        sub = df[["Datetime", f"Close_{t}", f"High_{t}", f"Low_{t}", f"Open_{t}", f"Volume_{t}"]].copy()
        sub["Ticker"] = t
        sub.rename(
            columns={
                f"Close_{t}": "Close",
                f"High_{t}": "High",
                f"Low_{t}": "Low",
                f"Open_{t}": "Open",
                f"Volume_{t}": "Volume",
            },
            inplace=True,
        )
        frames.append(sub)

    df_long = pd.concat(frames, ignore_index=True)
    df_long.sort_values(["Ticker", "Datetime"], inplace=True)
    df_long.reset_index(drop=True, inplace=True)
    return df_long


def load_ohlcv(refresh: bool = True) -> pd.DataFrame:
    if refresh:
        fetch_and_upload_ohlcv()
    csv_text = download_blob(DATA_FILE)
    return parse_wide_csv(csv_text)
