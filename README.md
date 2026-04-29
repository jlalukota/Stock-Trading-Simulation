# GabeJay Stock — Quantitative Trading Simulation

A production-structured quantitative trading simulation. Pulls intraday data via yfinance, engineers features, trains a RandomForestRegressor, and executes 15-minute hold-period trades with realistic transaction costs and slippage. Includes a backtesting engine, out-of-sample evaluation pipeline, and an event-driven scheduler that replaces blocking sleep with APScheduler.

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │      scheduler.py            │
                        │  APScheduler, 15-min interval│
                        └────────────┬────────────────-┘
                                     │ each bar
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                       ▼
     close_positions()          load_ohlcv()          open_positions()
     trading/executor.py        data/ingestion.py     trading/executor.py
                                      │                      │
                                 features.py           portfolio.py
                                 models/trainer.py     allocate_capital()
                                      │                      │
                                      └──────────┬───────────┘
                                                 ▼
                                            state.py
                                       state/positions.json
```

**Cycle order per bar**: close → ingest → train → open. Closing before opening means each position is held for exactly one bar, and sell proceeds fund the next round of buys.

---

## Module Structure

| File | Responsibility |
|---|---|
| `scheduler.py` | Event-driven entry point. APScheduler fires every 15 min. |
| `main.py` | Legacy blocking entry point. Preserved for local testing. |
| `config.py` | Tickers list, GCS paths, feature/target column names. |
| `state.py` | Persists `{capital, positions}` to `state/positions.json`. Survives restarts. |
| `data/ingestion.py` | yfinance download, GCS upload/download, wide-to-long reshape. |
| `data/features.py` | Return, SMA\_20, Momentum, BollingerWidth, Volatility. |
| `models/trainer.py` | RandomForestRegressor fit + predict. |
| `trading/selector.py` | Selects top-N tickers by predicted return at the latest bar. |
| `trading/executor.py` | `open_positions` / `close_positions` with GCS trade logging. |
| `portfolio.py` | Four allocation strategies with position limits. |
| `evaluation.py` | Time-split and walk-forward out-of-sample evaluation. |
| `backtest.py` | Historical simulation with costs, slippage, equity curve output. |
| `utils/market.py` | NYSE market hours check. |

---

## Setup

```bash
pip install -r requirements.txt
export GCLOUD_CREDENTIALS='<json key contents>'   # or set GOOGLE_APPLICATION_CREDENTIALS
```

---

## Usage

Live trading (event-driven)

```bash
# Fresh start with $100k capital
python3 scheduler.py --reset --capital 100000

# Resume after a restart (closes any open positions on first cycle)
python3 scheduler.py

# Different strategy and position count
python3 scheduler.py --strategy vol_adjusted --top-n 5 --max-position-frac 0.25
```

**Sample output:**
```
2024-03-15 09:30:01  INFO      scheduler  Scheduler starting | capital=$100000.00 | strategy=combined | top_n=3 | interval=15min
2024-03-15 09:30:02  INFO      scheduler  === Cycle start 2024-03-15 09:30 ET ===
2024-03-15 09:30:02  INFO      scheduler  Capital: $100000.00
2024-03-15 09:30:02  INFO      scheduler  Ingesting data...
2024-03-15 09:30:18  INFO      scheduler  Training model...
2024-03-15 09:30:31  INFO      models.trainer  Model trained on 12,450 rows. In-sample MSE: 0.000021
2024-03-15 09:30:31  INFO      scheduler  Selecting top-3 trades...
Selected trades:
 Ticker      Close  PredictedReturn  Volatility
   NVDA  874.150000         0.003241    0.018432
   META  502.340000         0.002987    0.014221
   MSFT  415.780000         0.002104    0.011038
2024-03-15 09:30:32  INFO      trading.executor  Opened NVDA: 19.1823 shares @ 874.1500  ($16,775.43)
2024-03-15 09:30:32  INFO      trading.executor  Opened META: 26.6102 shares @ 502.3400  ($13,368.29)
2024-03-15 09:30:32  INFO      trading.executor  Opened MSFT: 32.1847 shares @ 415.7800  ($13,390.21)
2024-03-15 09:30:32  INFO      scheduler  Opened 3 position(s)
2024-03-15 09:30:32  INFO      scheduler  === Cycle complete ===
```

```bash
# Inspect open positions without running
cat state/positions.json
```
```json
{
  "capital": 56466.07,
  "positions": [
    {"Ticker": "NVDA", "BuyTime": "2024-03-15T09:30:32+00:00", "BuyPrice": 874.15, "Shares": 19.1823, "Allocation": 16775.43, "PredictedReturn": 0.003241},
    {"Ticker": "META", "BuyTime": "2024-03-15T09:30:32+00:00", "BuyPrice": 502.34, "Shares": 26.6102, "Allocation": 13368.29, "PredictedReturn": 0.002987},
    {"Ticker": "MSFT", "BuyTime": "2024-03-15T09:30:32+00:00", "BuyPrice": 415.78, "Shares": 32.1847, "Allocation": 13390.21, "PredictedReturn": 0.002104}
  ]
}
```

---

Backtesting

```bash
# Full year, daily bars
python3 backtest.py --start 2024-01-01 --end 2025-01-01

# Last 60 days, 15-minute bars
python3 backtest.py --start 2024-11-01 --end 2025-01-01 --interval 15m

# Compare allocation strategies
python3 backtest.py --start 2024-01-01 --end 2025-01-01 --allocation-strategy equal
python3 backtest.py --start 2024-01-01 --end 2025-01-01 --allocation-strategy combined

# Higher costs, more positions
python3 backtest.py --start 2024-01-01 --end 2025-01-01 --cost-bps 10 --slippage-bps 5 --top-n 5
```

**Sample output:**
```
============================================================
BACKTEST SUMMARY
============================================================
  Period:           2024-01-01 -> 2025-01-01  (1d bars)
  Tickers:          98  |  Top-N per bar: 3
  Allocation:       combined  (max 40%/pos, min 5%/pos)
  Cost:             5.0 bps/side  |  Slippage: 3.0 bps/side

  Starting capital: $    100,000.00
  Ending capital:   $    107,423.15
  Total return:     +7.42%
  Sharpe ratio:     +0.8341  (annualised)
  Max drawdown:     12.34%

  Trades executed:  714
  Win rate:         52.38%
  Avg PnL/trade:    $10.39
  Total costs paid: $6,821.44
============================================================
Equity curve saved to backtest_output/equity_curve.csv
Trade log saved to backtest_output/trade_log.csv
Ticker summary saved to backtest_output/ticker_summary.csv
```

**Output files:**

| File | Contents |
|---|---|
| `backtest_output/equity_curve.csv` | `Datetime, Equity` — one row per bar |
| `backtest_output/trade_log.csv` | Every trade: entry/exit price, P&L, actual vs predicted return |
| `backtest_output/ticker_summary.csv` | Per-ticker: trade count, total P&L, win rate, avg return |

> **Note on intervals**: yfinance restricts 15m data to the last 60 calendar days. Use `--interval 1d` for longer date ranges.

---

Out-of-sample evaluation

```bash
# Default: 60% train / 20% val / 20% test + walk-forward
python3 evaluation.py --no-refresh

# Tune window sizes
python3 evaluation.py --wf-train-periods 80 --wf-step-periods 26 --top-n 5
```

**Sample output:**
```
============================================================
EVALUATION PIPELINE
============================================================
Total rows: 12,450  |  Tickers: 98  |  Periods: 127

[1/3] In-sample (training set only — not a real metric)

  In-sample:
    Samples:              12,450
    Periods:              127
    MSE:                  0.00000021
    RMSE:                 0.00045826
    Directional accuracy: 81.34%
    Strategy Sharpe:      42.1823  (annualised, 15-min bars)

[2/3] Hold-out test split (train 60% / val 20% / test 20%)
  Train periods: 76  Val periods: 25  Test periods: 26

  Validation:
    Samples:              2,548
    Periods:              25
    MSE:                  0.00000389
    RMSE:                 0.00197358
    Directional accuracy: 51.23%
    Strategy Sharpe:      0.3241  (annualised, 15-min bars)

  Test (OOS):
    Samples:              2,562
    Periods:              26
    MSE:                  0.00000412
    RMSE:                 0.00202977
    Directional accuracy: 50.78%
    Strategy Sharpe:      0.1892  (annualised, 15-min bars)

[3/3] Walk-forward validation (train_window=60 bars, step=13 bars)

  Walk-forward OOS:
    Samples:              3,185
    Periods:              65
    MSE:                  0.00000401
    RMSE:                 0.00200250
    Directional accuracy: 50.91%
    Strategy Sharpe:      0.2714  (annualised, 15-min bars)

============================================================
SUMMARY — why in-sample numbers are misleading
============================================================
  In-sample MSE is always lower than OOS MSE because the model
  memorises training data. The only numbers that predict live
  performance are the hold-out test and walk-forward metrics.
  A high directional_accuracy (>0.5) matters more than low MSE:
  correctly predicting the direction of a move is what drives P&L.
```

**How to read the numbers:**

| Metric | What it means | Good sign |
|---|---|---|
| MSE | Mean squared prediction error | Lower OOS — but only vs baseline |
| Directional accuracy | `sign(pred) == sign(actual)` | > 50% after costs |
| Strategy Sharpe | Annualised Sharpe of long-top-N strategy | > 0.5 OOS |
| In-sample vs OOS gap | How much the model overfits | Small gap |

---

Capital allocation comparison

```python
from portfolio import compare_allocations
# top_df needs: Ticker, PredictedReturn, Volatility
print(compare_allocations(top_df, capital=100_000))
```

```
        PredictedReturn  Volatility  pred/vol    equal  vol_adjusted  confidence  combined
Ticker
NVDA           0.003241    0.018432    0.1759  33333.3      25841.12    35012.44  28164.31
META           0.002987    0.014221    0.2100  33333.3      33471.08    32279.63  40000.00
MSFT           0.002104    0.011038    0.1906  33333.3      40687.80    22707.93  31835.69
```

---

Design decisions

**No data leakage**: rolling features (`SMA_20`, `BollingerWidth`, `Volatility`) are computed per-ticker in time order — `pandas.rolling()` is a causal operation. No global normalization is applied across the train/test boundary.

**Walk-forward over static split**: the model is retrained on each window rather than once on all training data. A static backtest trains on the future relative to its first evaluation bar if you're not careful; rolling retrain enforces the same discipline as live operation.

**APScheduler over asyncio**: all I/O (yfinance, GCS, sklearn) is synchronous. asyncio would require `run_in_executor` wrappers with no latency benefit at 15-minute resolution. APScheduler's thread-per-job model fits the existing call stack directly.

**Crash-safe positions**: `open_positions` writes to `state/positions.json` before returning. On restart, the scheduler reads the file and closes any open positions before opening new ones. No manual intervention needed after a crash.

**Allocation default (`combined`)**: weights by `predicted_return / volatility` — the predicted per-trade Sharpe ratio. This rewards high conviction *and* penalises high-risk tickers. Pure confidence weighting without the volatility denominator can over-concentrate in volatile names when the model happens to predict large moves.

---

Known limitations

- **yfinance 15m restriction**: intraday data is limited to the last 60 calendar days. Backtests beyond that must use `--interval 1d`.
- **RandomForest for returns**: tree-based models don't extrapolate well out of their training distribution. A market regime shift (e.g. 2020 COVID crash) will degrade OOS performance sharply.
- **No short selling**: all positions are long-only. The allocators treat negative predicted returns as zero weight.
- **Single-bar hold**: positions are held for exactly one 15-minute bar. This matches the original system's design but ignores momentum effects that might favour longer holds.
- **No portfolio-level risk limit**: each bar allocates the full capital. A drawdown stop (e.g. reduce position size after 5% loss) is not implemented.
