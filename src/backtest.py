# src/backtest.py
# Backtest a signals CSV created by build_signals_range / engine --asof
# - Canonicalizes tickers (trim/upper + dot<->hyphen handling)
# - Batch-downloads prices via yfinance (more reliable) with fallback per-ticker
# - De-dupes signals to one (date, ticker) row
# - Ignores repeated BUY signals while a position is open (unless --pyramid)
# - Verbose debug phases
#
# Example (Windows cmd.exe):
#   python -m src.backtest ^
#     --signals storage\signals_2025-06_07.csv ^
#     --out-prefix storage\bt_2025-06_07 ^
#     --cash 100000 --max-pos 10 --hold-days 20 --cost-bps 10 ^
#     --stop-pct 12 --take-pct 25 ^
#     --entry close ^
#     --start 2025-06-01 --end 2025-08-05 ^
#     --debug

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

@dataclass
class Trade:
    open_date: pd.Timestamp
    close_date: Optional[pd.Timestamp]
    ticker: str
    qty: int
    entry_px: float
    exit_px: Optional[float]
    reason: str
    pnl: Optional[float]

def load_signals(path: str, debug: bool = False) -> pd.DataFrame:
    if debug: print(f"[DEBUG] phase: load_signals → {path}")
    df = pd.read_csv(path)

    # date column
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        src_date_col = "timestamp"
    else:
        if "date" not in df.columns:
            raise ValueError("signals CSV must contain 'timestamp' or 'date'")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        src_date_col = "date"

    if "ticker" not in df.columns:
        raise ValueError("signals CSV must contain 'ticker'")

    # only BUYs if present
    if "signal_type" in df.columns:
        before = len(df)
        df = df[df["signal_type"].astype(str).str.upper() == "BUY"].copy()
        if debug: print(f"[DEBUG] filtered BUYs: {before} → {len(df)}")

    # strict: do NOT alter tickers
    df["ticker"] = df["ticker"].astype(str)

    # basic cleanup + de‑dupe per (date,ticker)
    df = df.dropna(subset=["date", "ticker"])
    df.sort_values(["date", "ticker"], inplace=True)
    before = len(df)
    df = df.drop_duplicates(subset=["date", "ticker"], keep="first").reset_index(drop=True)
    if debug and len(df) != before:
        print(f"[DEBUG] dedup (date,ticker): {before} → {len(df)}")

    if debug and not df.empty:
        print(f"[DEBUG] rows: {len(df)} | unique tickers: {df['ticker'].nunique()}")
        print(f"[DEBUG] date col: {src_date_col} | range: {df['date'].min()} → {df['date'].max()}")
    return df

def _split_multiindex_batch(df: pd.DataFrame, debug: bool = False) -> Dict[str, pd.DataFrame]:
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
            sub = sub[keep].dropna(subset=["Open", "Close"])
            if not sub.empty:
                out[t] = sub
            elif debug:
                print(f"[DEBUG] batch: empty after cleaning for {t}")
    else:
        # single ticker returned
        sub = df.copy()
        sub.index = pd.to_datetime(sub.index)
        cols = {c: str(c).title() for c in sub.columns}
        sub = sub.rename(columns=cols)
        if "Adj Close" in sub.columns and "Close" not in sub.columns:
            sub = sub.rename(columns={"Adj Close": "Close"})
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in sub.columns]
        sub = sub[keep].dropna(subset=["Open", "Close"])
        out["__single__"] = sub
    return out

def load_prices_batch(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp, debug: bool = False) -> Dict[str, pd.DataFrame]:
    if debug:
        print(f"[DEBUG] phase: load_prices_batch | tickers={len(tickers)} | range={start.date()}→{end.date()}")
    prices: Dict[str, pd.DataFrame] = {}

    # Batch first (strict tickers, as-is)
    try:
        dfb = yf.download(
            tickers=tickers,
            start=(start - pd.Timedelta(days=5)).date().isoformat(),
            end=(end + pd.Timedelta(days=5)).date().isoformat(),
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=True,
            group_by="ticker",
        )
        if dfb is not None and not dfb.empty:
            split = _split_multiindex_batch(dfb, debug=debug)
            for t in tickers:
                if t in split and not split[t].empty:
                    prices[t] = split[t]
    except Exception as e:
        if debug: print(f"[DEBUG] batch download error: {e}")

    # Fill any missing with single calls (still strict tickers)
    missing = [t for t in tickers if t not in prices]
    if debug:
        print(f"[DEBUG] batch fetched: {len(prices)} / {len(tickers)} | fallback singles: {len(missing)}")
    for t in missing:
        try:
            dfi = yf.download(
                t,
                start=(start - pd.Timedelta(days=5)).date().isoformat(),
                end=(end + pd.Timedelta(days=5)).date().isoformat(),
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
                dfi = dfi[keep].dropna(subset=["Open", "Close"])
                if not dfi.empty:
                    prices[t] = dfi
        except Exception as e:
            if debug: print(f"[DEBUG] single download error for {t}: {e}")

    if debug:
        got = sum(1 for t in tickers if t in prices)
        print(f"[DEBUG] price datasets available: {got} / {len(tickers)}")
    return prices

def next_open(df: pd.DataFrame, asof_day: pd.Timestamp) -> Optional[float]:
    if df is None or df.empty: return None
    idx = df.index.searchsorted(asof_day + pd.Timedelta(days=1))
    if idx < len(df): return float(df.iloc[idx]["Open"])
    return None

def latest_close(df: pd.DataFrame, day: pd.Timestamp) -> Optional[float]:
    if df is None or df.empty: return None
    idx = df.index.searchsorted(day, side="right") - 1
    if idx >= 0: return float(df.iloc[idx]["Close"])
    return None

def run_backtest(
    signals: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    cash0: float = 100_000.0,
    max_pos: int = 10,
    hold_days: int = 20,
    cost_bps: float = 10.0,
    stop_pct: Optional[float] = 12.0,
    take_pct: Optional[float] = 25.0,
    entry: str = "next_open",
    debug: bool = False,
    pyramid: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Trade]]:
    bt_start = pd.to_datetime(start).normalize() if start else signals["date"].min()
    bt_end   = pd.to_datetime(end).normalize()   if end   else signals["date"].max()
    if debug:
        print(f"[DEBUG] phase: select_range | bt_start={bt_start.date()} | bt_end={bt_end.date()}")

    sig = signals[(signals["date"] >= bt_start) & (signals["date"] <= bt_end)].copy()
    if sig.empty: raise ValueError("No BUY signals in the selected date range.")

    tickers = sorted(sig["ticker"].unique().tolist())
    prices = load_prices_batch(tickers, bt_start, bt_end, debug=debug)

    if debug:
        missing = [t for t in tickers if t not in prices]
        print(f"[DEBUG] phase: price_check | have={len(tickers)-len(missing)} | missing={len(missing)}")
        if missing:
            print(f"[DEBUG] missing: {', '.join(missing[:25])}{' ...' if len(missing)>25 else ''}")

    days = pd.date_range(bt_start, bt_end, freq="B")
    cash = cash0
    positions: Dict[str, Dict] = {}  # ticker -> {qty, entry_px, entry_day}
    trades: List[Trade] = []
    nav_rows = []

    # per-day signals, strict ticker
    by_day: Dict[pd.Timestamp, List[pd.Series]] = {}
    for r in sig.itertuples(index=False):
        d = pd.to_datetime(r.date).normalize()
        by_day.setdefault(d, [])
        if not any(x.ticker == r.ticker for x in by_day[d]):  # de-duplicate same-day ticker
            by_day[d].append(r)
        elif debug:
            print(f"[DEBUG {d.date()}] drop duplicate signal for {r.ticker}")

    half_cost = cost_bps / 2.0 / 1e4
    if debug:
        print(f"[DEBUG] phase: simulate | days={len(days)} | max_pos={max_pos} | entry={entry} | pyramid={pyramid}")

    for d in days:
        # exits
        to_close: List[str] = []
        for t, p in positions.items():
            dfp = prices.get(t)

            # time exit
            if (d.date() - p["entry_day"].date()).days >= hold_days:
                px = next_open(dfp, d - pd.Timedelta(days=1)) if entry == "next_open" else latest_close(dfp, d)
                if px is not None:
                    cash += p["qty"] * px * (1 - half_cost)
                    pnl = p["qty"] * (px - p["entry_px"])
                    trades.append(Trade(p["entry_day"], d, t, p["qty"], p["entry_px"], px, "time", pnl))
                    to_close.append(t)
                    if debug: print(f"[DEBUG {d.date()}] exit {t} by time @ {px:.2f}, pnl={pnl:.2f}")
                    continue

            cl = latest_close(dfp, d)
            if cl is None:
                if debug: print(f"[DEBUG {d.date()}] no close for {t}, skip stop/take")
                continue

            if stop_pct and cl <= p["entry_px"] * (1 - stop_pct/100):
                px = next_open(dfp, d) if entry == "next_open" else cl
                if px is not None:
                    cash += p["qty"] * px * (1 - half_cost)
                    pnl = p["qty"] * (px - p["entry_px"])
                    trades.append(Trade(p["entry_day"], d + pd.Timedelta(days=1), t, p["qty"], p["entry_px"], px, "stop", pnl))
                    to_close.append(t)
                    if debug: print(f"[DEBUG {d.date()}] exit {t} by stop @ {px:.2f}, pnl={pnl:.2f}")
                    continue

            if take_pct and cl >= p["entry_px"] * (1 + take_pct/100):
                px = next_open(dfp, d) if entry == "next_open" else cl
                if px is not None:
                    cash += p["qty"] * px * (1 - half_cost)
                    pnl = p["qty"] * (px - p["entry_px"])
                    trades.append(Trade(p["entry_day"], d + pd.Timedelta(days=1), t, p["qty"], p["entry_px"], px, "take", pnl))
                    to_close.append(t)
                    if debug: print(f"[DEBUG {d.date()}] exit {t} by take @ {px:.2f}, pnl={pnl:.2f}")
                    continue

        for t in to_close:
            positions.pop(t, None)

        # entries
        todays = by_day.get(d, [])
        free_slots = max(0, max_pos - len(positions))
        if todays and free_slots > 0:
            # equal-weight on current equity
            mtm = 0.0
            for tt, pp in positions.items():
                cl = latest_close(prices.get(tt), d)
                if cl is not None:
                    mtm += cl * pp["qty"]
            equity = cash + mtm
            alloc = equity / max_pos

            candidates: List[pd.Series] = []
            seen = set()
            for r in todays:
                if r.ticker in seen: continue
                seen.add(r.ticker)
                if (not pyramid) and (r.ticker in positions):
                    if debug: print(f"[DEBUG {d.date()}] skip {r.ticker}: already in positions")
                    continue
                candidates.append(r)

            for r in candidates[:free_slots]:
                dfp = prices.get(r.ticker)
                if dfp is None or dfp.empty:
                    if debug: print(f"[DEBUG {d.date()}] skip {r.ticker}: no price data")
                    continue
                px = next_open(dfp, d) if entry == "next_open" else latest_close(dfp, d)
                if px is None:
                    if debug: print(f"[DEBUG {d.date()}] skip {r.ticker}: no {entry} price")
                    continue

                qty = math.floor((alloc * (1 - half_cost)) / px)
                if qty <= 0:
                    if debug: print(f"[DEBUG {d.date()}] skip {r.ticker}: qty=0 (alloc≈{alloc:.2f}, px≈{px:.2f})")
                    continue
                cost = qty * px * (1 + half_cost)
                if cost > cash:
                    if debug: print(f"[DEBUG {d.date()}] skip {r.ticker}: not enough cash (need {cost:.2f}, have {cash:.2f})")
                    continue

                cash -= cost
                positions[r.ticker] = {"qty": qty, "entry_px": px, "entry_day": d}
                if debug: print(f"[DEBUG {d.date()}] enter {r.ticker}: qty={qty}, px={px:.2f}, cash→{cash:.2f}")

        # nav
        mtm = 0.0
        for t, p in positions.items():
            cl = latest_close(prices.get(t), d)
            if cl is not None:
                mtm += cl * p["qty"]
        nav_rows.append({"date": d.date(), "nav": cash + mtm, "cash": cash, "n_pos": len(positions)})

    # end-of-period mark
    for t, p in list(positions.items()):
        px = latest_close(prices.get(t), bt_end)
        if px is not None:
            pnl = p["qty"] * (px - p["entry_px"])
            trades.append(Trade(p["entry_day"], bt_end, t, p["qty"], p["entry_px"], px, "eop", pnl))
            if debug: print(f"[DEBUG {bt_end.date()}] mark {t} @ {px:.2f} (EOP), pnl={pnl:.2f}")

    nav_df = pd.DataFrame(nav_rows)
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    return nav_df, trades_df, trades

def summarize(nav_df: pd.DataFrame, trades_df: pd.DataFrame) -> str:
    if nav_df.empty: return "No NAV data."
    nav_ret = nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1
    nav_df["ret"] = nav_df["nav"].pct_change().fillna(0.0)
    avg = nav_df["ret"].mean()
    vol = nav_df["ret"].std()
    sharpe = (avg / vol * (252 ** 0.5)) if vol > 0 else float("nan")
    max_dd = 0.0; peak = -1e18
    for v in nav_df["nav"]:
        if v > peak: peak = v
        dd = (v / peak) - 1.0
        if dd < max_dd: max_dd = dd
    wins = (trades_df["pnl"] > 0).sum() if "pnl" in trades_df else 0
    total = len(trades_df)
    winrate = wins / total if total else float("nan")
    return (
        f"Total Return: {nav_ret*100:.2f}%\n"
        f"Max Drawdown: {max_dd*100:.2f}%\n"
        f"Sharpe (daily→annualized): {sharpe:.2f}\n"
        f"Trades: {total} | Win rate: {winrate*100:.1f}%\n"
        f"Avg Daily Return: {avg*100:.3f}% | Daily Vol: {vol*100:.3f}%"
    )

def main():
    ap = argparse.ArgumentParser(description="Backtest signals CSV (strict tickers).")
    ap.add_argument("--signals", required=True)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--cash", type=float, default=100000.0)
    ap.add_argument("--max-pos", type=int, default=10)
    ap.add_argument("--hold-days", type=int, default=20)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--stop-pct", type=float, default=12.0)
    ap.add_argument("--take-pct", type=float, default=25.0)
    ap.add_argument("--entry", choices=["next_open", "close"], default="next_open")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--pyramid", action="store_true")
    args = ap.parse_args()

    sig = load_signals(args.signals, debug=args.debug)
    nav_df, trades_df, _ = run_backtest(
        sig,
        start=args.start,
        end=args.end,
        cash0=args.cash,
        max_pos=args.max_pos,
        hold_days=args.hold_days,
        cost_bps=args.cost_bps,
        stop_pct=args.stop_pct,
        take_pct=args.take_pct,
        entry=args.entry,
        debug=args.debug,
        pyramid=args.pyramid,
    )
    base = args.out_prefix or os.path.splitext(args.signals)[0]
    nav_path = f"{base}_nav.csv"
    trades_path = f"{base}_trades.csv"
    os.makedirs(os.path.dirname(nav_path) or ".", exist_ok=True)
    nav_df.to_csv(nav_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    print("\n=== Backtest Summary ===")
    print(summarize(nav_df, trades_df))
    print(f"\nSaved NAV → {nav_path}")
    print(f"Saved trades → {trades_path}")

if __name__ == "__main__":
    main()