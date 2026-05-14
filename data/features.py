import pandas as pd
from config import FEATURE_COLS, TARGET_COL


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features to a long-format OHLCV DataFrame in-place.
    Requires columns: Datetime, Ticker, Close.
    Adds: Return, FutureReturn, SMA_20, Momentum, BollingerWidth, Volatility.
    """
    g = df.groupby("Ticker")["Close"]

    df["Return"] = g.pct_change()
    df["FutureReturn"] = g.pct_change(5).shift(-5)

    df["SMA_20"] = g.transform(lambda x: x.rolling(20).mean())
    df["SMA_5"] = g.transform(lambda x: x.rolling(5).mean())
    df["SMA_60"] = g.transform(lambda x: x.rolling(60).mean())
    df["price/SMA_5"] = df["Close"] / df["SMA_5"]
    df["price/SMA_20"] = df["Close"] / df["SMA_20"]
    df["price/SMA_60"] = df["Close"] / df["SMA_60"]
    df["SMA_20/SMA_60"] = df["SMA_20"] / df["SMA_60"]

    # Multi-horizon momentum
    df["Momentum_10"] = g.transform(lambda x: x.pct_change(10))
    df["Momentum_20"] = g.transform(lambda x: x.pct_change(20))
    df["Momentum_60"] = g.transform(lambda x: x.pct_change(60))

    # Bollinger band width = upper_band - lower_band = 4 × rolling_std
    rolling_mean = g.transform(lambda x: x.rolling(30).mean())
    rolling_std = g.transform(lambda x: x.rolling(30).std())
    df["BollingerWidth"] = (4 * rolling_std) / rolling_mean

    # Volatility regime
    df["Volatility_5"] = df.groupby("Ticker")["Return"].transform(lambda x: x.rolling(5).std())
    df["Volatility_60"] = df.groupby("Ticker")["Return"].transform(lambda x: x.rolling(60).std())
    df["VolatilityRatio"] = df["Volatility_5"] / df["Volatility_60"]

    df["VWAP_20"] = df.groupby("Ticker").apply(
        lambda x: (x["Close"] * x["Volume"]).rolling(20).sum() / x["Volume"].rolling(20).sum()
    ).reset_index(level=0, drop=True)
    df["price/VWAP_20"] = df["Close"] / df["VWAP_20"]

    df["RVOL"] = df.groupby("Ticker")["Volume"].transform(
        lambda x: x / x.rolling(20).mean()
    )

    # Volume anomaly
    df["Volume_Z"] = df.groupby("Ticker")["Volume"].transform(
        lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()
    )

    # Rolling Sharpe-like feature
    df["RollingReturn_20"] = df.groupby("Ticker")["Return"].transform(lambda x: x.rolling(20).mean())
    df["RollingSharpe_20"] = df["RollingReturn_20"] / df["Volatility_5"]


    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies features and drops rows with any NaN in the feature or target columns.
    Returns a clean DataFrame ready for model consumption.
    """
    df = add_features(df.copy())
    df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)
    return df
