"""
Out-of-sample evaluation pipeline.

Run:
    python3 evaluation.py
    python3 evaluation.py --train-frac 0.6 --val-frac 0.2 --top-n 3

Design:
  - All splits are time-ordered (no shuffling, no leakage).
  - Rolling features are computed on the full dataset before splitting;
    this is safe because pandas rolling() is a causal operation — each
    row's value depends only on its own past.
  - Walk-forward retrains the model from scratch on each window, mirroring
    live operation where the model is periodically retrained on fresh data.
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from config import FEATURE_COLS, TARGET_COL
from data.features import build_feature_matrix
from data.ingestion import load_ohlcv
from models.trainer import train

BARS_PER_DAY = 26          # 15-min bars in a 6.5-hour session
ANNUALIZATION = math.sqrt(252 * BARS_PER_DAY)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits df into (train, val, test) by unique Datetime values.
    test_frac is implied: 1 - train_frac - val_frac.

    Splitting by timestamp (not row index) guarantees all tickers at the
    same bar land in the same split, preventing cross-ticker leakage.
    """
    times = np.sort(df["Datetime"].unique())
    n = len(times)
    i_val = int(n * train_frac)
    i_test = int(n * (train_frac + val_frac))

    t_train = set(times[:i_val])
    t_val = set(times[i_val:i_test])
    t_test = set(times[i_test:])

    return (
        df[df["Datetime"].isin(t_train)].copy(),
        df[df["Datetime"].isin(t_val)].copy(),
        df[df["Datetime"].isin(t_test)].copy(),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions where sign(pred) == sign(actual)."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


def strategy_sharpe(
    df_oos: pd.DataFrame,
    predicted_col: str = "PredictedReturn",
    actual_col: str = TARGET_COL,
    top_n: int = 3,
) -> float:
    """
    Simulates a long-only strategy: at each bar, go long the top_n tickers
    by predicted return, record their actual return.
    Returns annualised Sharpe ratio (mean / std * sqrt(bars_per_year)).

    This measures whether the MODEL'S RANKING is useful, not just whether
    predictions are numerically close to actuals.
    """
    period_returns = []
    for ts, group in df_oos.groupby("Datetime"):
        if len(group) < top_n:
            continue
        selected = group.nlargest(top_n, predicted_col)[actual_col]
        period_returns.append(selected.mean())

    if len(period_returns) < 2:
        return float("nan")

    r = np.array(period_returns)
    std = r.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(r.mean() / std * ANNUALIZATION)


def compute_metrics(
    df_oos: pd.DataFrame,
    predicted_col: str = "PredictedReturn",
    top_n: int = 3,
) -> dict:
    y_true = df_oos[TARGET_COL].values
    y_pred = df_oos[predicted_col].values
    return {
        "n_samples": len(y_true),
        "n_periods": df_oos["Datetime"].nunique(),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "strategy_sharpe": strategy_sharpe(df_oos, predicted_col=predicted_col, top_n=top_n),
    }


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward(
    df: pd.DataFrame,
    train_periods: int,
    step_periods: int,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Rolls a training window of `train_periods` unique timestamps across df,
    retraining the model each step and collecting OOS predictions.

    Returns a DataFrame of all OOS rows with an added PredictedReturn column.

    train_periods: how many bars to train on each fold
    step_periods:  how many bars to predict (and advance by) each fold
    """
    times = np.sort(df["Datetime"].unique())
    n = len(times)

    if train_periods >= n:
        raise ValueError(
            f"train_periods ({train_periods}) must be < total periods ({n})."
        )

    all_oos: list[pd.DataFrame] = []
    fold = 0

    for start in range(0, n - train_periods, step_periods):
        train_times = set(times[start : start + train_periods])
        oos_times = set(times[start + train_periods : start + train_periods + step_periods])

        if not oos_times:
            break

        df_train = df[df["Datetime"].isin(train_times)]
        df_oos = df[df["Datetime"].isin(oos_times)].copy()

        if df_train.empty or df_oos.empty:
            continue

        model = train(df_train)
        df_oos["PredictedReturn"] = model.predict(df_oos[FEATURE_COLS])

        all_oos.append(df_oos)
        fold += 1

    print(f"Walk-forward complete: {fold} folds, "
          f"{sum(len(f) for f in all_oos):,} OOS rows total.")
    return pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def run_evaluation(
    df_features: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    top_n: int = 3,
    wf_train_periods: int = 60,
    wf_step_periods: int = 13,
) -> None:
    """
    Runs and prints three evaluation tiers:
      1. In-sample  — what the original code reported (misleading baseline)
      2. Hold-out   — simple time-split test set
      3. Walk-forward — rolling retrain, most realistic of the three
    """
    print("\n" + "=" * 60)
    print("EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Total rows: {len(df_features):,}  |  "
          f"Tickers: {df_features['Ticker'].nunique()}  |  "
          f"Periods: {df_features['Datetime'].nunique()}")

    # ---- 1. In-sample baseline (what the old code showed) ----------------
    print("\n[1/3] In-sample (training set only — not a real metric)")
    model_full = train(df_features)
    df_features["PredictedReturn"] = model_full.predict(df_features[FEATURE_COLS])
    m_in = compute_metrics(df_features, top_n=top_n)
    _print_metrics(m_in, label="In-sample")

    # ---- 2. Hold-out test split ------------------------------------------
    print("\n[2/3] Hold-out test split "
          f"(train {train_frac:.0%} / val {val_frac:.0%} / test {1-train_frac-val_frac:.0%})")
    train_df, val_df, test_df = time_split(df_features, train_frac, val_frac)
    print(f"  Train periods: {train_df['Datetime'].nunique()}  "
          f"Val periods: {val_df['Datetime'].nunique()}  "
          f"Test periods: {test_df['Datetime'].nunique()}")

    model_holdout = train(train_df)
    val_df = val_df.copy()
    test_df = test_df.copy()
    val_df["PredictedReturn"] = model_holdout.predict(val_df[FEATURE_COLS])
    test_df["PredictedReturn"] = model_holdout.predict(test_df[FEATURE_COLS])

    _print_metrics(compute_metrics(val_df, top_n=top_n), label="Validation")
    _print_metrics(compute_metrics(test_df, top_n=top_n), label="Test (OOS)")

    # ---- 3. Walk-forward -------------------------------------------------
    print(f"\n[3/3] Walk-forward validation "
          f"(train_window={wf_train_periods} bars, step={wf_step_periods} bars)")
    print("  Suppressing per-fold training output...")

    import logging
    logging.disable(logging.CRITICAL)
    _suppress_print()
    wf_df = walk_forward(df_features, wf_train_periods, wf_step_periods, top_n)
    _restore_print()
    logging.disable(logging.NOTSET)

    if not wf_df.empty:
        _print_metrics(compute_metrics(wf_df, top_n=top_n), label="Walk-forward OOS")
    else:
        print("  Not enough data for walk-forward with these parameters.")

    # ---- Summary comparison ----------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY — why in-sample numbers are misleading")
    print("=" * 60)
    print(
        "  In-sample MSE is always lower than OOS MSE because the model\n"
        "  memorises training data. The only numbers that predict live\n"
        "  performance are the hold-out test and walk-forward metrics.\n"
        "  A high directional_accuracy (>0.5) matters more than low MSE:\n"
        "  correctly predicting the direction of a move is what drives P&L."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_metrics(m: dict, label: str) -> None:
    print(f"\n  {label}:")
    print(f"    Samples:              {m['n_samples']:,}")
    print(f"    Periods:              {m['n_periods']}")
    print(f"    MSE:                  {m['mse']:.8f}")
    print(f"    RMSE:                 {m['rmse']:.8f}")
    print(f"    Directional accuracy: {m['directional_accuracy']:.2%}")
    print(f"    Strategy Sharpe:      {m['strategy_sharpe']:.4f}  (annualised, 15-min bars)")


_original_stdout = sys.stdout

def _suppress_print():
    import io
    sys.stdout = io.StringIO()

def _restore_print():
    sys.stdout = _original_stdout


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run out-of-sample evaluation.")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--wf-train-periods", type=int, default=60,
                        help="Number of bars per walk-forward training window")
    parser.add_argument("--wf-step-periods", type=int, default=13,
                        help="Number of bars to advance each fold (~half day)")
    parser.add_argument("--no-refresh", action="store_true",
                        help="Use cached GCS data instead of re-downloading")
    args = parser.parse_args()

    print("Loading data...")
    df_raw = load_ohlcv(refresh=not args.no_refresh)
    print("Engineering features...")
    df = build_feature_matrix(df_raw)

    run_evaluation(
        df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        top_n=args.top_n,
        wf_train_periods=args.wf_train_periods,
        wf_step_periods=args.wf_step_periods,
    )


if __name__ == "__main__":
    main()
