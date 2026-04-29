import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data.features import build_feature_matrix
from models.trainer import predict
from config import FEATURE_COLS


def select_top_trades(model: RandomForestRegressor, df_raw: pd.DataFrame, num_trades: int = 3) -> pd.DataFrame:
    """
    Computes features on the full OHLCV history so rolling windows are valid,
    then returns the top-N tickers by predicted return at the latest timestamp.
    """
    df = build_feature_matrix(df_raw)

    latest_time = df["Datetime"].max()
    latest = df[df["Datetime"] == latest_time].copy()

    if latest.empty:
        raise ValueError("No rows remain after feature computation for the latest timestamp.")

    latest["PredictedReturn"] = predict(model, latest)

    top = latest.nlargest(num_trades, "PredictedReturn")[["Ticker", "Close", "PredictedReturn", "Volatility"]]
    print("Selected trades:")
    print(top.to_string(index=False))
    return top
