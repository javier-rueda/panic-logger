import pandas as pd
import yfinance as yf

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Handle MultiIndex columns (e.g., ('Close','AAPL')) by keeping the first level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def get_history(ticker: str, period: str = "220d", interval: str = "1d", asof: str | None = None, lookback_days: int = 400) -> pd.DataFrame:
    """
    Download price history from Yahoo Finance.
    If `asof` is given, fetch data up to and including that date (no future data).
    """
    if asof:
        asof_date = pd.to_datetime(asof).normalize()
        start = (asof_date - pd.Timedelta(days=lookback_days)).date().isoformat()
        end = (asof_date + pd.Timedelta(days=1)).date().isoformat()  # yfinance end is exclusive
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    else:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

    if df is None or df.empty:
        return pd.DataFrame()

    # flatten and normalize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns=lambda s: str(s).strip().title())

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(subset=["Close"])
    return df