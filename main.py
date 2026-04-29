import time
from datetime import datetime as dt
#ddddd
import pytz

from data.ingestion import load_ohlcv
from data.features import build_feature_matrix
from models.trainer import train
from trading.selector import select_top_trades
from trading.executor import execute_cycle, append_trade_log
from utils.market import is_market_open

def run_cycle() -> None:
    print("\n--- New Trading Cycle ---")
    df_raw = load_ohlcv(refresh=True)
    df = build_feature_matrix(df_raw.copy())
    model = train(df)
    top_trades = select_top_trades(model, df_raw)
    completed = execute_cycle(top_trades)
    append_trade_log(completed)
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
