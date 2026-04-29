"""
Historical backtesting engine.

Usage:
    python3 backtest.py --start 2024-01-01 --end 2024-12-31
    python3 backtest.py --start 2024-01-01 --end 2024-12-31 --interval 1d --top-n 5
    python3 backtest.py --start 2024-01-01 --end 2024-12-31 --cost-bps 5 --slippage-bps 3

Note on intervals:
    yfinance restricts 15m intraday data to the last 60 calendar days.
    For ranges beyond that, use --interval 1d (default) or 1h (last ~730 days).

Execution model (no look-ahead):
    Bar t:   features computed, model predicts, top-N selected
    Entry:   close[t] × (1 + slippage_bps/10000)
    Exit:    close[t+1] × (1 - slippage_bps/10000)
    Cost:    cost_bps/10000 applied to notional on each side (entry + exit)

Model discipline:
    At bar t the model is trained on all bars strictly before t.
    No data from t or later enters the training set.
    Model is retrained every `retrain_every` periods.
"""

import argparse
import math
import os
from dataclasses import dataclass, field
from datetime import datetime as dt
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf

from config import FEATURE_COLS, TARGET_COL, TICKERS
from data.features import build_feature_matrix
from models.trainer import train

ANNUALIZATION = {
    "1d":  math.sqrt(252),
    "1h":  math.sqrt(252 * 7),
    "15m": math.sqrt(252 * 26),
}


# ---------------------------------------------------------------------------
# Configuration and result containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    start: str            = "2024-01-01"
    end: str              = "2025-01-01"
    interval: str         = "1d"
    top_n: int            = 3
    capital: float        = 100_000.0
    cost_bps: float       = 5.0       # commission per side, in basis points
    slippage_bps: float   = 3.0       # market-impact per side, in basis points
    min_train_periods: int = 60        # bars needed before first trade
    retrain_every: int    = 20         # retrain model every N bars
    cache_dir: str        = "cache"    # local CSV cache to avoid re-downloading


@dataclass
class BacktestResult:
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict           = field(default_factory=dict)
    config: BacktestConfig  = field(default_factory=BacktestConfig)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_backtest_data(config: BacktestConfig) -> pd.DataFrame:
    """
    Downloads OHLCV for all TICKERS over [start, end] at the given interval.
    Results are cached to disk to avoid repeated API calls during development.
    Returns long-format DataFrame: Datetime, Ticker, Close, High, Low, Open, Volume.
    """
    os.makedirs(config.cache_dir, exist_ok=True)
    cache_key = f"{config.cache_dir}/bt_{config.start}_{config.end}_{config.interval}.parquet"

    if os.path.exists(cache_key):
        print(f"Loading cached data from {cache_key}")
        return pd.read_parquet(cache_key)

    print(f"Downloading {config.interval} data from {config.start} to {config.end}...")
    raw = yf.download(
        TICKERS,
        start=config.start,
        end=config.end,
        interval=config.interval,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for interval={config.interval} "
            f"over [{config.start}, {config.end}]. "
            "For 15m data yfinance restricts to the last 60 days; use 1d for longer ranges."
        )

    # yfinance returns a MultiIndex DataFrame: (metric, ticker)
    # Reshape to long format.
    raw.index.name = "Datetime"
    raw = raw.stack(level=1, future_stack=True).reset_index()
    raw.rename(columns={"level_1": "Ticker", "Price": "Ticker"}, inplace=True)

    # Normalise column names that yfinance may capitalise differently
    raw.columns = [c.strip().capitalize() if c not in ("Datetime", "Ticker") else c for c in raw.columns]

    needed = {"Close", "High", "Low", "Open", "Volume"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"Expected columns missing after reshape: {missing}. Got: {list(raw.columns)}")

    raw = raw.dropna(subset=["Close"])
    raw["Datetime"] = pd.to_datetime(raw["Datetime"])
    raw.sort_values(["Ticker", "Datetime"], inplace=True)
    raw.reset_index(drop=True, inplace=True)

    raw.to_parquet(cache_key, index=False)
    print(f"Cached to {cache_key}")
    return raw


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough decline as a positive fraction (e.g. 0.25 = 25% drawdown)."""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(-drawdown.min())


def compute_backtest_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    ann: float,
) -> dict:
    period_returns = equity.pct_change().dropna()
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

    std = period_returns.std(ddof=1)
    sharpe = float(period_returns.mean() / std * ann) if std > 0 else float("nan")

    n_trades = len(trades)
    win_rate = float((trades["PnL"] > 0).mean()) if n_trades > 0 else float("nan")
    avg_pnl  = float(trades["PnL"].mean()) if n_trades > 0 else float("nan")
    total_costs = float(trades["Cost"].sum()) if "Cost" in trades.columns and n_trades > 0 else float("nan")

    return {
        "start_capital":   float(equity.iloc[0]),
        "end_capital":     float(equity.iloc[-1]),
        "total_return":    float(total_return),
        "annualised_return": float((1 + total_return) ** (ann ** 2 / len(period_returns)) - 1)
                              if len(period_returns) > 0 else float("nan"),
        "sharpe_ratio":    sharpe,
        "max_drawdown":    max_drawdown(equity),
        "n_trades":        n_trades,
        "win_rate":        win_rate,
        "avg_pnl":         avg_pnl,
        "total_costs":     total_costs,
    }


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def _equal_weight(top_df: pd.DataFrame, capital: float) -> dict[str, float]:
    """Returns {ticker: dollar_allocation}. Phase 4 replaces this."""
    n = len(top_df)
    alloc = capital / n if n > 0 else 0.0
    return {row["Ticker"]: alloc for _, row in top_df.iterrows()}


def run_backtest(
    config: BacktestConfig,
    allocate: Callable = _equal_weight,
) -> BacktestResult:
    """
    Main backtest loop.

    `allocate` is a function (top_df, capital) -> {ticker: dollar_amount}.
    Swap it out in Phase 4 without touching this function.
    """
    df_raw = fetch_backtest_data(config)

    print("Engineering features...")
    df = build_feature_matrix(df_raw)

    times = np.sort(df["Datetime"].unique())
    n_times = len(times)
    print(f"Backtest period: {times[0]} → {times[-1]} | {n_times} bars | {df['Ticker'].nunique()} tickers")

    slip  = config.slippage_bps / 10_000
    cost  = config.cost_bps    / 10_000

    # Build fast close-price lookup: (Datetime, Ticker) → Close
    close_lookup: dict[tuple, float] = {
        (row.Datetime, row.Ticker): row.Close
        for row in df[["Datetime", "Ticker", "Close"]].itertuples(index=False)
    }

    capital = config.capital
    equity_curve: dict = {}
    trade_records: list[dict] = []
    model = None
    last_trained_idx = -config.retrain_every  # force train on first eligible bar

    for i in range(n_times - 1):   # -1: we need t+1 for the exit price
        t      = times[i]
        t_next = times[i + 1]

        # --- Training ---
        # Strictly less-than: bar t is not in the training set.
        df_train = df[df["Datetime"] < t]
        n_train_periods = df_train["Datetime"].nunique()

        should_train = (
            n_train_periods >= config.min_train_periods
            and (i - last_trained_idx) >= config.retrain_every
        )
        if should_train:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                model = train(df_train)
            last_trained_idx = i

        if model is None:
            equity_curve[t] = capital
            continue

        # --- Prediction ---
        df_bar = df[df["Datetime"] == t].copy()
        if df_bar.empty:
            equity_curve[t] = capital
            continue

        df_bar["PredictedReturn"] = model.predict(df_bar[FEATURE_COLS])
        top = df_bar.nlargest(config.top_n, "PredictedReturn")

        # --- Execution ---
        allocations = allocate(top, capital)
        period_pnl = 0.0

        for _, row in top.iterrows():
            ticker = row["Ticker"]
            dollar_alloc = allocations.get(ticker, 0.0)
            if dollar_alloc <= 0:
                continue

            entry_close = close_lookup.get((t, ticker))
            exit_close  = close_lookup.get((t_next, ticker))
            if entry_close is None or exit_close is None:
                continue  # no data at one of the bars — skip, capital stays uninvested

            buy_price  = entry_close * (1 + slip)
            sell_price = exit_close  * (1 - slip)
            shares     = dollar_alloc / buy_price

            entry_cost = dollar_alloc * cost
            exit_cost  = (shares * sell_price) * cost
            trade_cost = entry_cost + exit_cost

            trade_pnl = shares * (sell_price - buy_price) - trade_cost
            period_pnl += trade_pnl

            trade_records.append({
                "EntryTime":       t,
                "ExitTime":        t_next,
                "Ticker":          ticker,
                "EntryPrice":      round(buy_price, 6),
                "ExitPrice":       round(sell_price, 6),
                "Shares":          round(shares, 6),
                "Allocation":      round(dollar_alloc, 4),
                "PnL":             round(trade_pnl, 4),
                "Cost":            round(trade_cost, 4),
                "PredictedReturn": round(row["PredictedReturn"], 8),
                "ActualReturn":    round((exit_close - entry_close) / entry_close, 8),
            })

        capital += period_pnl
        equity_curve[t] = capital

    equity_curve[times[-1]] = capital  # record final value

    equity = pd.Series(equity_curve, name="Equity")
    equity.index.name = "Datetime"
    trades_df = pd.DataFrame(trade_records)

    ann = ANNUALIZATION.get(config.interval, math.sqrt(252))
    metrics = compute_backtest_metrics(equity, trades_df, ann)

    return BacktestResult(
        equity_curve=equity,
        trade_log=trades_df,
        metrics=metrics,
        config=config,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(result: BacktestResult) -> None:
    cfg = result.config
    m   = result.metrics
    sep = "=" * 60

    print(f"\n{sep}")
    print("BACKTEST SUMMARY")
    print(sep)
    print(f"  Period:           {cfg.start} -> {cfg.end}  ({cfg.interval} bars)")
    print(f"  Tickers:          {len(TICKERS)}  |  Top-N per bar: {cfg.top_n}")
    print(f"  Cost:             {cfg.cost_bps} bps/side  |  Slippage: {cfg.slippage_bps} bps/side")
    print(f"\n  Starting capital: ${m['start_capital']:>14,.2f}")
    print(f"  Ending capital:   ${m['end_capital']:>14,.2f}")
    print(f"  Total return:     {m['total_return']:>+.2%}")
    print(f"  Sharpe ratio:     {m['sharpe_ratio']:>+.4f}  (annualised)")
    print(f"  Max drawdown:     {m['max_drawdown']:.2%}")
    print(f"\n  Trades executed:  {m['n_trades']:,}")
    print(f"  Win rate:         {m['win_rate']:.2%}")
    print(f"  Avg PnL/trade:    ${m['avg_pnl']:.4f}")
    print(f"  Total costs paid: ${m['total_costs']:.2f}")
    print(sep)


def save_results(result: BacktestResult, output_dir: str = "backtest_output") -> None:
    os.makedirs(output_dir, exist_ok=True)

    equity_path = os.path.join(output_dir, "equity_curve.csv")
    result.equity_curve.to_csv(equity_path, header=True)
    print(f"Equity curve saved to {equity_path}")

    if not result.trade_log.empty:
        trades_path = os.path.join(output_dir, "trade_log.csv")
        result.trade_log.to_csv(trades_path, index=False)
        print(f"Trade log saved to {trades_path}")

        # Per-ticker P&L summary
        summary = (
            result.trade_log.groupby("Ticker")
            .agg(
                Trades=("PnL", "count"),
                TotalPnL=("PnL", "sum"),
                WinRate=("PnL", lambda x: (x > 0).mean()),
                AvgActualReturn=("ActualReturn", "mean"),
                AvgPredictedReturn=("PredictedReturn", "mean"),
            )
            .sort_values("TotalPnL", ascending=False)
        )
        summary_path = os.path.join(output_dir, "ticker_summary.csv")
        summary.to_csv(summary_path)
        print(f"Ticker summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run historical backtest.")
    parser.add_argument("--start",            default="2024-01-01")
    parser.add_argument("--end",              default="2025-01-01")
    parser.add_argument("--interval",         default="1d",
                        choices=["1d", "1h", "15m"],
                        help="Bar size. yfinance limits: 15m=last 60d, 1h=last 730d, 1d=unlimited")
    parser.add_argument("--top-n",            type=int,   default=3)
    parser.add_argument("--capital",          type=float, default=100_000.0)
    parser.add_argument("--cost-bps",         type=float, default=5.0,
                        help="Commission per side in basis points")
    parser.add_argument("--slippage-bps",     type=float, default=3.0,
                        help="Slippage per side in basis points")
    parser.add_argument("--min-train-periods",type=int,   default=60)
    parser.add_argument("--retrain-every",    type=int,   default=20)
    parser.add_argument("--output-dir",       default="backtest_output")
    parser.add_argument("--no-cache",         action="store_true",
                        help="Ignore local cache and re-download data")
    args = parser.parse_args()

    config = BacktestConfig(
        start=args.start,
        end=args.end,
        interval=args.interval,
        top_n=args.top_n,
        capital=args.capital,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        min_train_periods=args.min_train_periods,
        retrain_every=args.retrain_every,
    )

    if args.no_cache:
        cache_key = f"{config.cache_dir}/bt_{config.start}_{config.end}_{config.interval}.parquet"
        if os.path.exists(cache_key):
            os.remove(cache_key)
            print(f"Cache cleared: {cache_key}")

    result = run_backtest(config)
    print_summary(result)
    save_results(result, args.output_dir)


if __name__ == "__main__":
    main()
