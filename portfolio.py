"""
Capital allocation strategies.

All strategies share the same interface:
    allocate_capital(top_df, capital, strategy, max_position_frac, min_position_frac)
    -> dict[ticker, dollar_amount]

top_df must contain columns: Ticker, PredictedReturn, Volatility.
All dollar amounts sum to <= capital (no leverage).

Strategies
----------
equal        : capital / N per position. Naive baseline.
vol_adjusted : weight ∝ 1/σ. Equal risk contribution per position.
               Favours low-volatility tickers.
confidence   : weight ∝ |predicted_return|. Larger position when model
               conviction is higher. Ignores risk — use with limits.
combined     : weight ∝ predicted_return / σ. Risk-adjusted conviction
               (predicted per-trade Sharpe). Default strategy.

Position limits
---------------
max_position_frac : no single position exceeds this fraction of capital.
min_position_frac : positions below this threshold are zeroed out.
Weights are clipped then renormalised in one pass.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Weight normalization
# ---------------------------------------------------------------------------

def _normalize(raw_weights: dict[str, float], max_frac: float, min_frac: float) -> dict[str, float]:
    """
    Clip weights to [min_frac, max_frac] and renormalise to sum to 1.0.
    Weights below min_frac are dropped (zeroed out), not raised.
    One-pass clip + renorm is exact for small N (top-3 to top-10 portfolios).
    """
    total = sum(raw_weights.values())
    if total <= 0:
        return {t: 0.0 for t in raw_weights}

    normed = {t: w / total for t, w in raw_weights.items()}

    # Drop positions below minimum
    normed = {t: w for t, w in normed.items() if w >= min_frac}
    if not normed:
        return {}

    # Clip maximum and renormalise once
    clipped = {t: min(w, max_frac) for t, w in normed.items()}
    total_clipped = sum(clipped.values())
    if total_clipped <= 0:
        return {}

    return {t: w / total_clipped for t, w in clipped.items()}


def _to_dollars(weights: dict[str, float], capital: float) -> dict[str, float]:
    return {t: w * capital for t, w in weights.items()}


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _equal(top_df: pd.DataFrame) -> dict[str, float]:
    n = len(top_df)
    if n == 0:
        return {}
    w = 1.0 / n
    return {row["Ticker"]: w for _, row in top_df.iterrows()}


def _vol_adjusted(top_df: pd.DataFrame) -> dict[str, float]:
    """
    Inverse-volatility weighting. Tickers with lower historical volatility
    receive a proportionally larger allocation.
    Falls back to equal weight for any ticker with zero/missing volatility.
    """
    weights = {}
    for _, row in top_df.iterrows():
        vol = row.get("Volatility", np.nan)
        if pd.isna(vol) or vol <= 0:
            vol = top_df["Volatility"].median()  # fallback
        if pd.isna(vol) or vol <= 0:
            vol = 1.0
        weights[row["Ticker"]] = 1.0 / vol
    return weights


def _confidence(top_df: pd.DataFrame) -> dict[str, float]:
    """
    Weight proportional to predicted return magnitude.
    Only positive predicted returns are allocated; negatives are zeroed.
    """
    weights = {}
    for _, row in top_df.iterrows():
        pred = row.get("PredictedReturn", 0.0)
        weights[row["Ticker"]] = max(float(pred), 0.0)
    return weights


def _combined(top_df: pd.DataFrame) -> dict[str, float]:
    """
    Risk-adjusted conviction: weight ∝ predicted_return / volatility.
    This is the predicted per-trade Sharpe ratio.
    Only positive scores are allocated.
    """
    weights = {}
    median_vol = top_df["Volatility"].median()
    for _, row in top_df.iterrows():
        pred = float(row.get("PredictedReturn", 0.0))
        vol  = float(row.get("Volatility", np.nan))
        if pd.isna(vol) or vol <= 0:
            vol = median_vol if (not pd.isna(median_vol) and median_vol > 0) else 1.0
        score = pred / vol
        weights[row["Ticker"]] = max(score, 0.0)
    return weights


_STRATEGIES = {
    "equal":        _equal,
    "vol_adjusted": _vol_adjusted,
    "confidence":   _confidence,
    "combined":     _combined,
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def allocate_capital(
    top_df: pd.DataFrame,
    capital: float,
    strategy: str = "combined",
    max_position_frac: float = 0.40,
    min_position_frac: float = 0.05,
) -> dict[str, float]:
    """
    Returns {ticker: dollar_amount} for the given top_df.

    Parameters
    ----------
    top_df            : DataFrame with Ticker, PredictedReturn, Volatility.
    capital           : Total capital available for this period.
    strategy          : One of 'equal', 'vol_adjusted', 'confidence', 'combined'.
    max_position_frac : Hard cap per position as fraction of capital.
    min_position_frac : Positions below this fraction are excluded.

    Returns
    -------
    dict[str, float] where values sum to <= capital.
    Empty dict if no valid positions (e.g. all predictions negative).
    """
    if top_df.empty:
        return {}

    strategy_fn = _STRATEGIES.get(strategy)
    if strategy_fn is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(_STRATEGIES)}")

    raw_weights = strategy_fn(top_df)

    if not raw_weights or sum(raw_weights.values()) <= 0:
        # Fall back to equal weight if strategy produces no positive weights
        raw_weights = _equal(top_df)

    final_weights = _normalize(raw_weights, max_position_frac, min_position_frac)
    return _to_dollars(final_weights, capital)


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_allocations(
    top_df: pd.DataFrame,
    capital: float = 100_000.0,
    max_position_frac: float = 0.40,
    min_position_frac: float = 0.05,
) -> pd.DataFrame:
    """
    Returns a DataFrame comparing all strategies side-by-side.
    Useful for spot-checking allocation behaviour on a given bar's predictions.
    """
    rows = []
    for ticker in top_df["Ticker"]:
        row = {"Ticker": ticker}
        for name in _STRATEGIES:
            alloc = allocate_capital(
                top_df, capital,
                strategy=name,
                max_position_frac=max_position_frac,
                min_position_frac=min_position_frac,
            )
            row[name] = round(alloc.get(ticker, 0.0), 2)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Ticker")

    # Add metadata columns for context
    meta = top_df.set_index("Ticker")[["PredictedReturn", "Volatility"]].copy()
    meta["pred/vol"] = (meta["PredictedReturn"] / meta["Volatility"]).round(4)
    return pd.concat([meta, df], axis=1)
