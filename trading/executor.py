"""
Trade execution: separated into open and close operations.

open_positions(top_df, capital, strategy)
    Called at bar start. Buys selected tickers at current market price.
    Records open positions in state file.

close_positions()
    Called at the NEXT bar start, before opening new positions.
    Sells all positions recorded by the previous open_positions call.
    Returns completed trade records and realised P&L.
    Clears positions from state file.

This separation eliminates time.sleep(). Holding time = one bar interval,
enforced by the scheduler calling close before open on each cycle.
"""

import csv
import io
import logging
from datetime import datetime as dt

import pandas as pd
import pytz
import yfinance as yf

import state
from config import TRADE_LOG_FILE
from data.ingestion import download_blob, upload_blob
from portfolio import allocate_capital

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

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
    if df.empty:
        raise RuntimeError(f"No price data available for {ticker}")
    return float(df["Close"].iloc[-1])


# ---------------------------------------------------------------------------
# Open / close operations
# ---------------------------------------------------------------------------

def open_positions(
    top_df: pd.DataFrame,
    strategy: str = "combined",
    max_position_frac: float = 0.40,
    min_position_frac: float = 0.05,
) -> list[dict]:
    """
    Buys positions for the selected tickers. Uses capital from state.
    Returns list of position records (also persisted to state file).
    """
    capital = state.get_capital()
    allocations = allocate_capital(
        top_df, capital,
        strategy=strategy,
        max_position_frac=max_position_frac,
        min_position_frac=min_position_frac,
    )

    buy_time = dt.now(pytz.utc).isoformat()
    positions = []

    for _, row in top_df.iterrows():
        ticker = row["Ticker"]
        dollar_alloc = allocations.get(ticker, 0.0)
        if dollar_alloc <= 0:
            continue

        try:
            buy_price = get_current_price(ticker)
        except Exception as exc:
            logger.warning("Could not fetch price for %s: %s — skipping", ticker, exc)
            continue

        shares = dollar_alloc / buy_price
        positions.append({
            "Ticker":          ticker,
            "BuyTime":         buy_time,
            "BuyPrice":        buy_price,
            "Shares":          shares,
            "Allocation":      dollar_alloc,
            "PredictedReturn": float(row.get("PredictedReturn", 0.0)),
        })
        logger.info("Opened %s: %.4f shares @ %.4f  ($%.2f)", ticker, shares, buy_price, dollar_alloc)

    # Capital is committed to open positions; record remaining uninvested cash
    invested = sum(p["Allocation"] for p in positions)
    uninvested = capital - invested
    state.save_open_positions(positions, capital=uninvested)

    return positions


def close_positions() -> list[dict]:
    """
    Sells all open positions from the previous cycle.
    Updates capital with realised P&L.
    Logs completed trades to GCS.
    Returns list of completed trade dicts.
    """
    positions = state.get_open_positions()
    if not positions:
        return []

    sell_time = dt.now(pytz.utc).isoformat()
    completed = []
    realised_pnl = 0.0

    for pos in positions:
        ticker = pos["Ticker"]
        try:
            sell_price = get_current_price(ticker)
        except Exception as exc:
            logger.warning("Could not fetch sell price for %s: %s — position left open", ticker, exc)
            continue

        shares  = pos["Shares"]
        buy_price = pos["BuyPrice"]
        pnl     = shares * (sell_price - buy_price)
        realised_pnl += pnl

        trade = {
            "Ticker":          ticker,
            "BuyTime":         pos["BuyTime"],
            "BuyPrice":        buy_price,
            "SellTime":        sell_time,
            "SellPrice":       sell_price,
            "Shares":          shares,
            "PnL":             round(pnl, 4),
            "PredictedReturn": pos["PredictedReturn"],
            "ActualReturn":    round((sell_price - buy_price) / buy_price, 8),
        }
        completed.append(trade)
        logger.info("Closed %s: sold @ %.4f  PnL=$%.4f", ticker, sell_price, pnl)

    # Restore uninvested cash + realised P&L
    uninvested_cash = state.get_capital()  # was stored as uninvested during open
    total_invested_back = sum(pos["Shares"] * c["SellPrice"]
                               for pos, c in zip(positions, completed)
                               if c["Ticker"] == pos["Ticker"])
    new_capital = uninvested_cash + total_invested_back
    state.clear_positions(capital=new_capital)
    logger.info("Capital after close: $%.2f  (P&L this cycle: $%.4f)", new_capital, realised_pnl)

    if completed:
        append_trade_log(completed)

    return completed


# ---------------------------------------------------------------------------
# Trade logging (GCS)
# ---------------------------------------------------------------------------

def append_trade_log(trades: list[dict]) -> None:
    header = ["Ticker", "BuyTime", "BuyPrice", "SellTime", "SellPrice",
              "Shares", "PnL", "PredictedReturn", "ActualReturn"]
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
        writer.writerow([t.get(col, "") for col in header])

    upload_blob(TRADE_LOG_FILE, buf.getvalue())
    logger.info("Logged %d trade(s) to %s", len(trades), TRADE_LOG_FILE)
