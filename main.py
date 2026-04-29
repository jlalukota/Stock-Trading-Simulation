"""
Legacy blocking entry point. Preserved for reference and testing.

For production use, run scheduler.py instead:
    python3 scheduler.py --strategy combined --top-n 3

This file runs one cycle per 15-minute bar using time.sleep(), which
blocks the process and cannot recover cleanly from a crash mid-sleep.
scheduler.py eliminates this with APScheduler + persistent state.
"""
import time
#ddddd
from datetime import datetime as dt

import pytz

from data.features import build_feature_matrix
from data.ingestion import load_ohlcv
from models.trainer import train
from trading.executor import close_positions, open_positions
from trading.selector import select_top_trades
from utils.market import is_market_open


def run_cycle() -> None:
    print("\n--- New Trading Cycle ---")
    close_positions()
    df_raw = load_ohlcv(refresh=True)
    df = build_feature_matrix(df_raw.copy())
    model = train(df)
    top_trades = select_top_trades(model, df_raw)
    open_positions(top_trades)
    print("--- Cycle complete ---\n")


def main() -> None:
    eastern = pytz.timezone("US/Eastern")
    while True:
        if is_market_open():
            print(f"Market open at {dt.now(eastern).isoformat()}")
            run_cycle()
        else:
            print(f"Market closed at {dt.now(eastern).isoformat()}. Sleeping 15s...")
            time.sleep(15)


if __name__ == "__main__":
    main()
