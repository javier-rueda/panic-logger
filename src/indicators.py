import pandas as pd
import numpy as np

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # Classic Wilder's RSI
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def zscore(series: pd.Series, window: int) -> pd.Series:
    m = sma(series, window)
    s = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - m) / s

def pct_drop_over_n(series: pd.Series, n: int) -> pd.Series:
    """Percent change over the last n bars (negative means drop)."""
    return (series / series.shift(n) - 1.0) * 100.0