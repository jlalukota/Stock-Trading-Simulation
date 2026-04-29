import pandas as pd
from config import FEATURE_COLS, TARGET_COL


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features to a long-format OHLCV DataFrame in-place.
    Requires columns: Datetime, Ticker, Close.
    Adds: Return, FutureReturn, SMA_20, Momentum, BollingerWidth.
    """
    g = df.groupby("Ticker")["Close"]

    df["Return"] = g.pct_change()
    df["FutureReturn"] = g.pct_change().shift(-1)
    df["SMA_20"] = g.transform(lambda x: x.rolling(20).mean())
    df["Momentum"] = g.transform(lambda x: x.diff(5))
    # Bollinger band width = upper_band - lower_band = 4 × rolling_std
    df["BollingerWidth"] = g.transform(lambda x: x.rolling(30).std() * 4)

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies features and drops rows with any NaN in the feature or target columns.
    Returns a clean DataFrame ready for model consumption.
    """
    df = add_features(df.copy())
    df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)
    return df
