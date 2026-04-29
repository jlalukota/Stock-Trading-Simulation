import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from config import FEATURE_COLS, TARGET_COL


def train(df: pd.DataFrame, n_estimators: int = 100, random_state: int = 42) -> RandomForestRegressor:
    """
    Fits a RandomForestRegressor on the provided feature matrix.
    df must already have FEATURE_COLS and TARGET_COL present (no NaNs).
    Prints in-sample MSE as a sanity check only — do not use it as a
    performance estimate; it measures memorization, not generalization.
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X, y)

    in_sample_mse = mean_squared_error(y, model.predict(X))
    print(f"Model trained on {len(X):,} rows. In-sample MSE: {in_sample_mse:.6f} (training set only — not a generalization metric)")
    return model


def predict(model: RandomForestRegressor, df: pd.DataFrame) -> pd.Series:
    """Returns a Series of predicted returns indexed like df."""
    return pd.Series(model.predict(df[FEATURE_COLS]), index=df.index, name="PredictedReturn")
