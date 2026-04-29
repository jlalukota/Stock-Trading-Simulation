import csv
import io
import time
from datetime import datetime as dt

import pytz
import yfinance as yf

from config import BUCKET_NAME, TRADE_LOG_FILE, HOLD_SECONDS
from data.ingestion import download_blob, upload_blob


def get_current_price(ticker: str) -> float:
    stock = yf.Ticker(ticker)
    try:
        price = stock.fast_info.last_price
        if price is not None:
            return float(price)
    except Exception:
        pass
    try:
        price = stock.info.get("regularMarketPrice")
        if price is not None:
            return float(price)
    except Exception:
        pass
    df = stock.history(period="1d", interval="1m")
    return float(df["Close"].iloc[-1])


def execute_cycle(top_trades_df, hold_seconds: int = HOLD_SECONDS) -> list[dict]:
    """
    Buys at current price, waits hold_seconds, sells at current price.
    Returns list of completed trade dicts.
    """
    now_utc = lambda: dt.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")

    buy_time = now_utc()
    buy_trades = []
    print("\n-- Buy Phase --")
    for _, row in top_trades_df.iterrows():
        ticker = row["Ticker"]
        price = get_current_price(ticker)
        buy_trades.append({"Ticker": ticker, "BuyTime": buy_time, "BuyPrice": price, "PredictedReturn": row["PredictedReturn"]})
        print(f"  {ticker}: bought at {price:.4f}")

    print(f"\nHolding for {hold_seconds}s...")
    time.sleep(hold_seconds)

    sell_time = now_utc()
    completed = []
    print("\n-- Sell Phase --")
    for trade in buy_trades:
        ticker = trade["Ticker"]
        sell_price = get_current_price(ticker)
        profit = sell_price - trade["BuyPrice"]
        completed.append({
            "Ticker": ticker,
            "BuyTime": trade["BuyTime"],
            "BuyPrice": trade["BuyPrice"],
            "SellTime": sell_time,
            "SellPrice": sell_price,
            "Profit": profit,
            "PredictedReturn": trade["PredictedReturn"],
        })
        print(f"  {ticker}: sold at {sell_price:.4f} | profit: {profit:.4f}")

    return completed


def append_trade_log(trades: list[dict]) -> None:
    header = ["Ticker", "BuyTime", "BuyPrice", "SellTime", "SellPrice", "Profit", "PredictedReturn"]

    try:
        existing = download_blob(TRADE_LOG_FILE)
    except Exception:
        existing = ""

    buf = io.StringIO()
    if not existing.strip():
        writer = csv.writer(buf)
        writer.writerow(header)
    else:
        buf.write(existing)
        if not existing.endswith("\n"):
            buf.write("\n")
        writer = csv.writer(buf)

    for t in trades:
        writer.writerow([t.get(col) for col in header])

    upload_blob(TRADE_LOG_FILE, buf.getvalue())
    print(f"Logged {len(trades)} trade(s) to {TRADE_LOG_FILE}")
