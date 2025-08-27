# src/build_signals_range.py — FAST signals builder (strict tickers, with market_cap & confidence)
# - One download per ticker for the whole range (+lookback buffer)
# - Optional parquet cache (.cache/prices)
# - Vectorized indicators once per ticker
# - Evaluates rules day-by-day (no look-ahead) in memory
# - Prints an intro banner and elapsed time
# - Uses tickers EXACTLY as in watchlist (no normalization / punctuation changes)

from __future__ import annotations

import argparse
import os
import json
import time
from typing import Dict, List, Tuple, Optional

import pandas as pd
import yfinance as yf
import yaml

from src.util import read_watchlist
from src.indicators import rsi, zscore, pct_drop_over_n
from src.rules import evaluate_entry, compute_confidence  # confidence same as engine.py

SIGNAL_HEADER = [
    "timestamp",
    "ticker",
    "market",
    "signal_type",
    "price",
    "reason",
    "values",
    "rule_id",
    "notes",
    "market_cap",
    "confidence",  # NEW: top-level column for easy consumption
]

# ---------- helpers ----------

def _ensure_dir(p: str) -> None:
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

def _infer_lookback_days(cfg: dict, default: int = 250, extra: int = 10) -> int:
    """
    Minimal bars required for indicators, inferred from config entry rule knobs.
    """
    entry = cfg.get("rules", {}).get("entry", {})
    rsi_p = int(entry.get("rsi_period", 14) or 14)
    sma_w = int(entry.get("sma_window", 200) or 200)
    drop_n = int(entry.get("drop_lookback_days", 10) or 10)
    need = max(rsi_p, sma_w, drop_n)
    # Keep at least 'default' unless we need more; then add 'extra'
    need_plus = need + extra
    return max(default, need_plus)

def _read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _download_batch(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Batch download via yfinance. May return MultiIndex columns (group_by='ticker').
    """
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        progress=False,
        auto_adjust=True,
        threads=True,
        group_by="ticker",
    )

def _split_multiindex_batch(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Convert yfinance batch result to dict[ticker] -> OHLCV DataFrame.
    """
    out: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out

    if isinstance(df.columns, pd.MultiIndex):
        tickers = sorted({k for k, _ in df.columns})
        for t in tickers:
            sub = df[t].copy()
            sub.index = pd.to_datetime(sub.index)
            cols = {c: str(c).title() for c in sub.columns}
            sub = sub.rename(columns=cols)
            if "Adj Close" in sub.columns and "Close" not in sub.columns:
                sub = sub.rename(columns={"Adj Close": "Close"})
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in sub.columns]
            sub = sub[keep].dropna(subset=["Close"])
            if not sub.empty:
                out[t] = sub
    else:
        # single ticker returned
        sub = df.copy()
        sub.index = pd.to_datetime(sub.index)
        cols = {c: str(c).title() for c in sub.columns}
        sub = sub.rename(columns=cols)
        if "Adj Close" in sub.columns and "Close" not in sub.columns:
            sub = sub.rename(columns={"Adj Close": "Close"})
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in sub.columns]
        sub = sub[keep].dropna(subset=["Close"])
        out["__single__"] = sub
    return out

def _cache_path(cache_dir: str, ticker: str, start: str, end: str) -> str:
    return os.path.join(cache_dir, f"{ticker}_{start}_{end}_1d.parquet")

def _get_prices_for_range(
    tickers: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str,
    use_cache: bool,
    chunk: int,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for all tickers once, possibly from cache; otherwise via batched yfinance.
    Returns dict[ticker] -> DataFrame with: Open, High, Low, Close, Volume.
    """
    out: Dict[str, pd.DataFrame] = {}
    tickers = [t for t in tickers if t]

    start_iso = start.date().isoformat()
    end_iso = end.date().isoformat()

    # Cache load
    if use_cache:
        for t in tickers:
            pth = _cache_path(cache_dir, t, start_iso, end_iso)
            if os.path.exists(pth):
                try:
                    df = pd.read_parquet(pth)
                    df.index = pd.to_datetime(df.index)
                    out[t] = df
                except Exception:
                    pass

    missing = [t for t in tickers if t not in out]
    if not missing:
        return out

    # Batch download in chunks
    for i in range(0, len(missing), max(1, chunk)):
        group = missing[i : i + chunk]
        try:
            dfb = _download_batch(group, start_iso, end_iso)
            split = _split_multiindex_batch(dfb)
            for t in group:
                df = split.get(t)
                if df is not None and not df.empty:
                    out[t] = df
        except Exception:
            # Per-ticker fallback
            for t in group:
                try:
                    dfi = yf.download(
                        t,
                        start=start_iso,
                        end=end_iso,
                        interval="1d",
                        progress=False,
                        auto_adjust=True,
                        threads=False,
                    )
                    if dfi is not None and not dfi.empty:
                        dfi.index = pd.to_datetime(dfi.index)
                        cols = {c: str(c).title() for c in dfi.columns}
                        dfi = dfi.rename(columns=cols)
                        if "Adj Close" in dfi.columns and "Close" not in dfi.columns:
                            dfi = dfi.rename(columns={"Adj Close": "Close"})
                        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in dfi.columns]
                        dfi = dfi[keep].dropna(subset=["Close"])
                        out[t] = dfi
                except Exception:
                    pass

    # Save to cache
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        for t, df in out.items():
            try:
                df.to_parquet(_cache_path(cache_dir, t, start_iso, end_iso), index=True)
            except Exception:
                pass

    return out

def _compute_indicators_inplace(df: pd.DataFrame, entry_cfg: dict) -> None:
    """
    Compute the handful of indicators your evaluate_entry() typically needs.
    (Adds columns; initial rows will be NaN until windows warm up.)
    """
    rsi_p = int(entry_cfg.get("rsi_period", 14) or 14)
    sma_w = int(entry_cfg.get("sma_window", 200) or 200)
    drop_n = int(entry_cfg.get("drop_lookback_days", 10) or 10)
    df[f"RSI{rsi_p}"] = rsi(df["Close"], period=rsi_p)
    df[f"Z{sma_w}"] = zscore(df["Close"], window=sma_w)
    df[f"DROP{drop_n}"] = pct_drop_over_n(df["Close"], n=drop_n)

def _get_market_caps(tickers: List[str]) -> Dict[str, Optional[float]]:
    """
    Fetch market cap once per ticker (strict symbols). Uses yfinance info when available,
    falls back to .info['marketCap'].
    """
    out: Dict[str, Optional[float]] = {}
    for t in tickers:
        mc = None
        try:
            tk = yf.Ticker(t)
            fi = getattr(tk, "info", None)
            if isinstance(fi, dict) and fi.get("market_cap"):
                mc = float(fi["market_cap"])
            else:
                info = tk.info or {}
                if "marketCap" in info and info["marketCap"]:
                    mc = float(info["marketCap"])
        except Exception:
            mc = None
        out[t] = mc
    return out

# ---------- main fast builder ----------

def main():
    t0 = time.time()

    ap = argparse.ArgumentParser(description="Fast builder: generate signals for a date range (no look-ahead).")
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", type=str, default=None, help="Output CSV (default: storage/signals_<start>_<end>.csv)")
    ap.add_argument("--freq", type=str, default="B", help="Pandas freq (default B=business days)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    ap.add_argument("--echo", action="store_true", help="Print '[YYYY-MM-DD] processed' per day")
    ap.add_argument("--cache-dir", type=str, default=os.path.join(".cache", "prices"),
                    help="Local Parquet cache directory")
    ap.add_argument("--no-cache", action="store_true", help="Disable local price cache")
    ap.add_argument("--chunk", type=int, default=64, help="Batch size for yfinance downloads")
    ap.add_argument("--extra-lookback", type=int, default=10, help="Extra days above inferred indicator lookback")
    args = ap.parse_args()

    # Read config and entry rule
    cfg = _read_config(args.config)
    entry_cfg = cfg.get("rules", {}).get("entry", {})
    conf_cfg  = cfg.get("confidence", {})  # NEW: confidence knobs (weights, caps, penalties)

    # Universe (strict: do NOT touch/normalize tickers)
    watchlist_path = cfg.get("universe", {}).get("watchlist_csv", "watchlist.csv")
    wl = read_watchlist(watchlist_path)
    tickers_meta: List[Tuple[str, str, str]] = []  # (ticker, orig, market)
    seen = set()
    for row in wl:
        orig = str(row.get("ticker", "")).strip()
        if not orig or orig in seen:
            continue
        seen.add(orig)
        tickers_meta.append((orig, orig, row.get("market", "NA")))

    # Dates and output path
    start = pd.to_datetime(args.start).normalize()
    end = pd.to_datetime(args.end).normalize()
    if end < start:
        raise SystemExit("--end must be >= --start")

    out_path = args.out or os.path.join("storage", f"signals_{start.date()}_{end.date()}.csv")
    _ensure_dir(out_path)
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"Output exists: {out_path} (use --overwrite)")

    # Lookback and history window (yfinance end is exclusive)
    lookback = _infer_lookback_days(cfg, extra=args.extra_lookback)
    hist_start = (start - pd.Timedelta(days=lookback)).normalize()
    hist_end = (end + pd.Timedelta(days=1)).normalize()

    # Intro banner
    days = pd.date_range(start, end, freq=args.freq)
    print("=== Panic Logger — Build Signals Range ===")
    print(f"Config file : {args.config}")
    print(f"Date range  : {start.date()} → {end.date()}  ({len(days)} days)")
    print(f"Universe    : {len(tickers_meta)} tickers (from {watchlist_path})")
    print(f"Indicators  : lookback {lookback} days (history starts {hist_start.date()})")
    print(f"Output file : {out_path}")
    print("==========================================\n")

    # Price data once for the whole range (cache-aware)
    tickers = [t for t, _, _ in tickers_meta]
    prices = _get_prices_for_range(
        tickers=tickers,
        start=hist_start,
        end=hist_end,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        chunk=args.chunk,
    )

    # Market caps once per ticker
    market_caps = _get_market_caps(tickers)

    # Pre-compute indicators once per ticker
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty:
            continue
        _compute_indicators_inplace(df, entry_cfg)

    # Iterate days in-memory (no look-ahead)
    rows = []
    for d in days:
        d_norm = pd.to_datetime(d).normalize()
        date_stamp = f"{d_norm.date().isoformat()} 00:00:00"

        for ticker, orig, market in tickers_meta:
            df = prices.get(ticker)
            if df is None or df.empty:
                continue

            # Use only bars up to and including d; require a bar ON d
            df_d = df[df.index.normalize() <= d_norm]
            if df_d.empty or df_d.index[-1].normalize() != d_norm:
                continue

            ok, info = evaluate_entry(df_d, entry_cfg)
            if not ok:
                continue

            last = df_d.iloc[-1]
            close = float(last["Close"])

            # --- Confidence (same philosophy as engine.py; no look-ahead) ---
            try:
                C, comps = compute_confidence(df_d, entry_cfg, conf_cfg)
                C_val = float(C)
            except Exception:
                C_val = float("nan")
                comps = {}

            # values blob stays compact but includes metrics & confidence
            values_blob = {}
            try:
                rsi_p = int(entry_cfg.get("rsi_period", 14) or 14)
                drop_n = int(entry_cfg.get("drop_lookback_days", 10) or 10)
                sma_w = int(entry_cfg.get("sma_window", 200) or 200)
                values_blob = {
                    "rsi": float(last.get(f"RSI{rsi_p}", float("nan"))),
                    "drop_pct": float(last.get(f"DROP{drop_n}", float("nan"))),
                    "zscore": float(last.get(f"Z{sma_w}", float("nan"))),
                    "confidence": round(C_val, 3),
                    "components": comps,
                }
            except Exception:
                # safe fallback
                values_blob = {"confidence": round(C_val, 3)}

            rows.append({
                "timestamp": date_stamp,
                "ticker": ticker,           # exact as in watchlist
                "market": market,
                "signal_type": "BUY",
                "price": f"{close:.4f}",
                "reason": info.get("reason", "entry_ok"),
                "values": json.dumps(values_blob, separators=(",", ":")) if values_blob else "{}",
                "rule_id": "ENTRY_V1",
                "notes": "",
                "market_cap": market_caps.get(ticker),
                "confidence": round(C_val, 4),
            })

        if args.echo:
            print(f"[{d_norm.date()}] processed")

    # Write all at once
    df_out = pd.DataFrame(rows, columns=SIGNAL_HEADER)
    _ensure_dir(out_path)
    if os.path.exists(out_path) and args.overwrite:
        os.remove(out_path)
    df_out.to_csv(out_path, index=False)

    dt = time.time() - t0
    print(f"\nSaved signals → {out_path}  (rows={len(df_out)})")
    print(f"Finished in {dt:.1f}s ({len(rows)} signals, {len(tickers_meta)} tickers, {len(days)} days)")

if __name__ == "__main__":
    main()
