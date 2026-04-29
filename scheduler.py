"""
Event-driven trading scheduler.

Entry point:
    python3 scheduler.py
    python3 scheduler.py --capital 50000 --strategy combined --top-n 3
    python3 scheduler.py --reset            # wipe state and start fresh

Architecture
------------
One job fires every 15 minutes during NYSE market hours.
Within each firing:
    1. close()   — sell positions opened the previous cycle
    2. ingest()  — pull fresh OHLCV from yfinance / GCS
    3. train()   — retrain RandomForest on fresh data
    4. open()    — allocate capital, buy new positions

Each step is isolated. If ingest or train fails, the cycle is skipped
and existing positions are left open (they close on the next cycle).
The scheduler itself never crashes from a job exception.

State
-----
Capital and open positions are persisted to state/positions.json.
On restart, any positions found in that file are closed before new ones
are opened. This prevents zombie positions surviving a crash.

Tradeoffs vs asyncio
--------------------
APScheduler runs jobs in threads. All existing code (yfinance, sklearn,
GCS) is synchronous — threading fits naturally. asyncio would require
run_in_executor() wrappers everywhere for no latency benefit at 15-min
resolution. The tradeoff is that a hanging job blocks its thread, so
each job has a max_instances=1 guard to prevent pile-up.
"""

import argparse
import logging
import sys
from datetime import datetime as dt

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

import state
from data.features import build_feature_matrix
from data.ingestion import load_ohlcv
from models.trainer import train
from trading.executor import close_positions, open_positions
from trading.selector import select_top_trades
from utils.market import is_market_open

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scheduler")


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

def _cycle(top_n: int, strategy: str, max_pos: float, min_pos: float) -> None:
    """
    One complete market cycle. Runs inside the scheduler thread.
    Any unhandled exception is caught by the caller so the scheduler
    continues running.
    """
    eastern = pytz.timezone("US/Eastern")
    now_et = dt.now(eastern)

    if not is_market_open():
        logger.info("Market closed at %s — skipping cycle", now_et.strftime("%H:%M ET"))
        return

    logger.info("=== Cycle start %s ===", now_et.strftime("%Y-%m-%d %H:%M ET"))
    capital = state.get_capital()
    logger.info("Capital: $%.2f", capital)

    # 1. Close previous positions (safe to call even if none are open)
    closed = close_positions()
    if closed:
        logger.info("Closed %d position(s) this cycle", len(closed))
        capital = state.get_capital()
        logger.info("Capital after close: $%.2f", capital)

    # 2. Ingest fresh data
    logger.info("Ingesting data...")
    try:
        df_raw = load_ohlcv(refresh=True)
    except Exception as exc:
        logger.error("Ingest failed: %s — cycle aborted, positions stay open", exc)
        return

    # 3. Train model
    logger.info("Training model...")
    try:
        df = build_feature_matrix(df_raw.copy())
        model = train(df)
    except Exception as exc:
        logger.error("Training failed: %s — cycle aborted", exc)
        return

    # 4. Select and open new positions
    logger.info("Selecting top-%d trades...", top_n)
    try:
        top = select_top_trades(model, df_raw, num_trades=top_n)
        opened = open_positions(
            top,
            strategy=strategy,
            max_position_frac=max_pos,
            min_position_frac=min_pos,
        )
        logger.info("Opened %d position(s)", len(opened))
    except Exception as exc:
        logger.error("Trade execution failed: %s", exc)
        return

    logger.info("=== Cycle complete ===")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Event-driven trading scheduler")
    parser.add_argument("--capital",          type=float, default=None,
                        help="Starting capital (only used on first run or with --reset)")
    parser.add_argument("--top-n",            type=int,   default=3)
    parser.add_argument("--strategy",         default="combined",
                        choices=["equal", "vol_adjusted", "confidence", "combined"])
    parser.add_argument("--max-position-frac",type=float, default=0.40)
    parser.add_argument("--min-position-frac",type=float, default=0.05)
    parser.add_argument("--interval-minutes", type=int,   default=15,
                        help="Bar interval in minutes (default: 15)")
    parser.add_argument("--reset",            action="store_true",
                        help="Wipe state/positions.json and start fresh")
    args = parser.parse_args()

    if args.reset:
        capital = args.capital or 100_000.0
        state.reset(initial_capital=capital)
        logger.info("State reset. Starting capital: $%.2f", capital)
    elif args.capital is not None and not state.has_open_positions():
        # Only set capital from CLI if there are no open positions (fresh start)
        state.set_capital(args.capital)
        logger.info("Capital initialised: $%.2f", args.capital)

    current_capital = state.get_capital()
    logger.info(
        "Scheduler starting | capital=$%.2f | strategy=%s | top_n=%d | interval=%dmin",
        current_capital, args.strategy, args.top_n, args.interval_minutes,
    )

    if state.has_open_positions():
        logger.warning(
            "Open positions found in state file — they will be closed on the first cycle."
        )

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        func=_cycle,
        trigger=IntervalTrigger(minutes=args.interval_minutes),
        kwargs={
            "top_n":    args.top_n,
            "strategy": args.strategy,
            "max_pos":  args.max_position_frac,
            "min_pos":  args.min_position_frac,
        },
        id="market_cycle",
        name="15-min market cycle",
        max_instances=1,        # prevent overlap if a cycle runs long
        coalesce=True,          # skip missed fires rather than pile them up
        misfire_grace_time=60,  # allow up to 60s late before considering it missed
    )

    logger.info("Scheduler running. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")
        if state.has_open_positions():
            logger.warning(
                "Positions are still open. Run scheduler again to close them, "
                "or inspect state/positions.json manually."
            )


if __name__ == "__main__":
    main()
