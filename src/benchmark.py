# src/benchmark.py
# Plot your backtest NAV vs a benchmark (e.g., SPY) and save a PNG.
#
# Usage A (existing NAV):
#   python -m src.benchmark --nav storage/signals_short_2023_2025_nav.csv --benchmark SPY --png out/short_vs_spy.png
#
# Usage B (run backtest inline, then plot):
#   python -m src.benchmark --signals storage/signals_short_2023_2025.csv --config configs/short_config.yaml \
#       --cash 10000 --max-pos 10 --entry next_open --start 2023-01-01 --end 2025-01-01 \
#       --benchmark SPY --png out/short_vs_spy.png
#
# Requirements: pandas yfinance matplotlib pyyaml

from __future__ import annotations

import argparse
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Optional imports (only needed for inline backtest mode)
try:
    from src.backtest import load_yaml_config, load_signals, run_backtest
except Exception:
    load_yaml_config = None
    load_signals = None
    run_backtest = None

import yfinance as yf


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure single-level string column names (yfinance can return MultiIndex)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) if isinstance(c, tuple) and len(c) else str(c) for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def _load_nav(nav_csv: str) -> pd.DataFrame:
    df = pd.read_csv(nav_csv)
    df = _flatten_columns(df)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "nav" not in cols:
        raise ValueError("NAV csv must contain columns 'date' and 'nav'")
    df[cols["date"]] = pd.to_datetime(df[cols["date"]])
    df = df.rename(columns={cols["date"]: "date", cols["nav"]: "nav"})
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "nav"]]


def _max_drawdown(series: pd.Series) -> float:
    peak = -1e18
    mdd = 0.0
    for v in series:
        if v > peak:
            peak = v
        dd = v / peak - 1.0
        if dd < mdd:
            mdd = dd
    return float(mdd)


def _cagr(series: pd.Series, dates: pd.Series) -> float:
    if len(series) < 2:
        return float("nan")
    s0 = float(series.iloc[0])
    s1 = float(series.iloc[-1])
    if s0 <= 0 or s1 <= 0:
        return float("nan")
    y = max((pd.to_datetime(dates.iloc[-1]) - pd.to_datetime(dates.iloc[0])).days / 365.25, 1e-9)
    return float((s1 / s0) ** (1.0 / y) - 1.0)


def _sharpe(daily_ret: pd.Series, rf_daily: float = 0.0) -> float:
    excess = daily_ret - rf_daily
    mu = excess.mean()
    sd = excess.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((mu / sd) * math.sqrt(252))


def _normalize_to_1(df: pd.DataFrame, col: str, newcol: str) -> pd.DataFrame:
    out = df.copy()
    base = float(out[col].iloc[0])  # .iloc[0] avoids the FutureWarning
    out[newcol] = out[col] / base if base != 0 else np.nan
    return out


def _download_benchmark(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        start=(pd.to_datetime(start) - pd.Timedelta(days=3)).date().isoformat(),
        end=(pd.to_datetime(end) + pd.Timedelta(days=3)).date().isoformat(),
        interval="1d",
        progress=False,
        auto_adjust=True,
        threads=True,
        group_by="column",
    )
    if df is None or df.empty:
        raise RuntimeError(f"Failed to download benchmark {ticker}")
    df = _flatten_columns(df)
    # handle both single-ticker and multiindex-like cases
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    if "Close" not in df.columns:
        # In some cases, yfinance returns columns like f"{ticker} Close"
        close_cols = [c for c in df.columns if c.lower().endswith("close")]
        if not close_cols:
            raise RuntimeError(f"Benchmark data for {ticker} missing 'Close' column")
        df = df.rename(columns={close_cols[0]: "Close"})
    df = df[["Close"]].dropna()
    df = df.reset_index()
    # the index column could be 'Date' or already named
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "index" in df.columns:
        df = df.rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "Close"]].sort_values("date").reset_index(drop=True)
    return df


def plot_vs_benchmark(nav_df: pd.DataFrame, bench_df: pd.DataFrame, name: str, bench_ticker: str, out_png: str) -> None:
    left = nav_df.copy()
    right = _flatten_columns(bench_df.copy())

    # Align date range
    start = max(left["date"].min(), right["date"].min())
    end = min(left["date"].max(), right["date"].max())
    left = left[(left["date"] >= start) & (left["date"] <= end)].reset_index(drop=True)
    right = right[(right["date"] >= start) & (right["date"] <= end)].reset_index(drop=True)

    # Resample benchmark to business days, forward-fill
    b = (right.set_index("date")
               .reindex(pd.date_range(start, end, freq="B"))
               .ffill())
    # After reindex, the index is DatetimeIndex without a column name → restore 'date'
    b.index.name = "date"
    b = b.reset_index()
    b = _flatten_columns(b)

    # Normalize both series to 1.0
    left = _normalize_to_1(left, "nav", "nav_n")
    b = _normalize_to_1(b, "Close", "bench_n")

    # Merge on date
    tmp = (left[["date", "nav_n"]]
           .merge(b[["date", "bench_n"]], on="date", how="inner"))

    # Stats
    tmp["ret_nav"] = tmp["nav_n"].pct_change().fillna(0.0)
    tmp["ret_bmk"] = tmp["bench_n"].pct_change().fillna(0.0)

    nav_cagr = _cagr(tmp["nav_n"], tmp["date"])
    bmk_cagr = _cagr(tmp["bench_n"], tmp["date"])
    nav_mdd = _max_drawdown(tmp["nav_n"])
    bmk_mdd = _max_drawdown(tmp["bench_n"])
    nav_sh = _sharpe(tmp["ret_nav"])
    bmk_sh = _sharpe(tmp["ret_bmk"])

    # Plot
    plt.figure(figsize=(10, 6), dpi=110)
    plt.plot(tmp["date"], tmp["nav_n"], label=f"{name} (CAGR {nav_cagr*100:.1f}%, MDD {nav_mdd*100:.1f}%, Sharpe {nav_sh:.2f})")
    plt.plot(tmp["date"], tmp["bench_n"], label=f"{bench_ticker} (CAGR {bmk_cagr*100:.1f}%, MDD {bmk_mdd*100:.1f}%, Sharpe {bmk_sh:.2f})", linestyle="--")
    plt.title(f"{name} vs {bench_ticker}")
    plt.ylabel("Growth (normalized to 1.0)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _ensure_dir(out_png)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    print(f"Saved PNG → {out_png}")
    print(f"\n{name}:   CAGR {nav_cagr*100:.2f}% | MaxDD {nav_mdd*100:.2f}% | Sharpe {nav_sh:.2f}")
    print(f"{bench_ticker}: CAGR {bmk_cagr*100:.2f}% | MaxDD {bmk_mdd*100:.2f}% | Sharpe {bmk_sh:.2f}")


def main():
    ap = argparse.ArgumentParser(description="Plot NAV vs benchmark (e.g., SPY) and save a PNG.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--nav", type=str, help="Path to NAV csv (from src.backtest)")
    src.add_argument("--signals", type=str, help="Signals CSV to backtest inline")
    ap.add_argument("--config", type=str, help="YAML config (required if --signals is used)")

    ap.add_argument("--cash", type=float, default=100000.0, help="Starting cash (inline backtest)")
    ap.add_argument("--max-pos", type=int, default=10, help="Max positions (inline backtest)")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="Round-trip cost in bps (inline)")
    ap.add_argument("--entry", choices=["next_open","close"], default="next_open", help="Entry timing (inline)")
    ap.add_argument("--start", type=str, default=None, help="Backtest start (inline)")
    ap.add_argument("--end", type=str, default=None, help="Backtest end (inline)")

    ap.add_argument("--benchmark", type=str, default="SPY", help="Benchmark ticker, e.g., SPY")
    ap.add_argument("--png", type=str, default="benchmark.png", help="Output PNG path")
    ap.add_argument("--name", type=str, default="Strategy", help="Label for your NAV curve")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    # Build or load NAV
    if args.nav:
        nav_df = _load_nav(args.nav)
        if args.start:
            nav_df = nav_df[nav_df["date"] >= pd.to_datetime(args.start)]
        if args.end:
            nav_df = nav_df[nav_df["date"] <= pd.to_datetime(args.end)]
        name = args.name or os.path.splitext(os.path.basename(args.nav))[0]
    else:
        if run_backtest is None or load_yaml_config is None or load_signals is None:
            raise SystemExit("Backtest module not available; use --nav or ensure src.backtest is importable.")
        if not args.config:
            raise SystemExit("--config is required when using --signals")

        cfg = load_yaml_config(args.config, debug=args.debug)
        sig = load_signals(args.signals, debug=args.debug)
        nav_df, trades_df, _ = run_backtest(
            sig,
            start=args.start,
            end=args.end,
            cash0=args.cash,
            max_pos=args.max_pos,
            cost_bps=args.cost_bps,
            entry=args.entry,
            config=cfg,
            debug=args.debug,
            pyramid=False,
        )
        name = args.name or os.path.splitext(os.path.basename(args.signals))[0]

    if nav_df is None or nav_df.empty:
        raise SystemExit("No NAV data to plot.")

    # Benchmark on same span as NAV
    start = nav_df["date"].min()
    end = nav_df["date"].max()
    bench_df = _download_benchmark(args.benchmark, start, end)

    plot_vs_benchmark(nav_df, bench_df, name=name, bench_ticker=args.benchmark, out_png=args.png)


if __name__ == "__main__":
    main()
