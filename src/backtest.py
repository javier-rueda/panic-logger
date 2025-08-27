# src/backtest.py
# Backtest BUY signals; exits & risk from YAML; sizing modes:
#   allocation_mode: fixed_cash | confidence | confidence_normalized
#   allocation_reference: portfolio | equity
#
# Example:
#   python -m src.backtest --signals storage/short_signals.csv ^
#     --config configs/short_config.yaml --max-pos 10 --entry next_open ^
#     --start 2023-01-01 --end 2025-01-01 --cash 10000 --debug
#
# Requires: pip install pyyaml pandas yfinance

from __future__ import annotations
import argparse, math, os, json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf
import yaml

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

# ---------------- YAML / IO ----------------

def load_yaml_config(path: str, debug: bool = False) -> dict:
    if debug: print(f"[DEBUG] load_config → {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_signals(path: str, debug: bool = False) -> pd.DataFrame:
    if debug: print(f"[DEBUG] load_signals → {path}")
    df = pd.read_csv(path)

    # normalize date
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    else:
        if "date" not in df.columns:
            raise ValueError("signals CSV must contain 'timestamp' or 'date'")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    if "ticker" not in df.columns:
        raise ValueError("signals CSV must contain 'ticker'")

    # Keep only BUYs if a type column exists
    if "signal_type" in df.columns:
        df = df[df["signal_type"].astype(str).str.upper() == "BUY"].copy()

    df["ticker"] = df["ticker"].astype(str)

    # optional: confidence column; fallback parse from JSON 'values'
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    elif "values" in df.columns:
        def _extract_conf(s):
            try:
                v = json.loads(s) if isinstance(s, str) else {}
                return float(v.get("confidence"))
            except Exception:
                return float("nan")
        df["confidence"] = df["values"].apply(_extract_conf)

    # clean & dedupe
    df = (df.dropna(subset=["date","ticker"])
            .sort_values(["date","ticker"])
            .drop_duplicates(subset=["date","ticker"], keep="first")
            .reset_index(drop=True))
    return df

# ---------------- Prices ----------------

def _split_multiindex_batch(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty: return out
    if isinstance(df.columns, pd.MultiIndex):
        tickers = sorted({k for k,_ in df.columns})
        for t in tickers:
            sub = df[t].copy(); sub.index = pd.to_datetime(sub.index)
            sub.columns = [str(c).title() for c in sub.columns]
            if "Adj Close" in sub.columns and "Close" not in sub.columns:
                sub = sub.rename(columns={"Adj Close": "Close"})
            keep = [c for c in ["Open","High","Low","Close","Volume"] if c in sub.columns]
            sub = sub[keep].dropna(subset=["Open","Close"])
            if not sub.empty: out[t] = sub
    else:
        sub = df.copy(); sub.index = pd.to_datetime(sub.index)
        sub.columns = [str(c).title() for c in sub.columns]
        if "Adj Close" in sub.columns and "Close" not in sub.columns:
            sub = sub.rename(columns={"Adj Close":"Close"})
        keep = [c for c in ["Open","High","Low","Close","Volume"] if c in sub.columns]
        out["__single__"] = sub[keep].dropna(subset=["Open","Close"])
    return out

def load_prices_batch(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp, debug: bool = False) -> Dict[str, pd.DataFrame]:
    prices: Dict[str, pd.DataFrame] = {}
    try:
        dfb = yf.download(
            tickers=tickers,
            start=(start - pd.Timedelta(days=5)).date().isoformat(),
            end=(end + pd.Timedelta(days=5)).date().isoformat(),
            interval="1d", progress=False, auto_adjust=True, threads=True, group_by="ticker",
        )
        if dfb is not None and not dfb.empty:
            prices.update(_split_multiindex_batch(dfb))
    except Exception as e:
        if debug: print(f"[DEBUG] batch error: {e}")
    missing = [t for t in tickers if t not in prices]
    for t in missing:
        try:
            dfi = yf.download(
                t,
                start=(start - pd.Timedelta(days=5)).date().isoformat(),
                end=(end + pd.Timedelta(days=5)).date().isoformat(),
                interval="1d", progress=False, auto_adjust=True, threads=False,
            )
            if dfi is not None and not dfi.empty:
                dfi.index = pd.to_datetime(dfi.index)
                dfi.columns = [str(c).title() for c in dfi.columns]
                if "Adj Close" in dfi.columns and "Close" not in dfi.columns:
                    dfi = dfi.rename(columns={"Adj Close":"Close"})
                keep = [c for c in ["Open","High","Low","Close","Volume"] if c in dfi.columns]
                dfi = dfi[keep].dropna(subset=["Open","Close"])
                if not dfi.empty: prices[t] = dfi
        except Exception as e:
            if debug: print(f"[DEBUG] single error {t}: {e}")
    return prices

def next_open(df: pd.DataFrame, asof_day: pd.Timestamp) -> Optional[float]:
    if df is None or df.empty: return None
    idx = df.index.searchsorted(asof_day + pd.Timedelta(days=1))
    return float(df.iloc[idx]["Open"]) if idx < len(df) else None

def latest_close(df: pd.DataFrame, day: pd.Timestamp) -> Optional[float]:
    if df is None or df.empty: return None
    idx = df.index.searchsorted(day, side="right") - 1
    return float(df.iloc[idx]["Close"]) if idx >= 0 else None

# ---------------- Indicators (exits) ----------------

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0); down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0.0, 1e-12))
    return 100 - (100 / (1 + rs))

def precompute_indicators(prices: Dict[str, pd.DataFrame], exit_cfg: dict) -> Dict[str, Dict[str, pd.Series]]:
    out: Dict[str, Dict[str, pd.Series]] = {}
    sma_window = exit_cfg.get("sma_window")
    rsi_period = exit_cfg.get("rsi_period")
    for t, df in prices.items():
        ind: Dict[str, pd.Series] = {}
        if sma_window: ind["sma"] = compute_sma(df["Close"], int(sma_window))
        if rsi_period: ind["rsi"] = compute_rsi(df["Close"], int(rsi_period))
        out[t] = ind
    return out

# ---------------- Core backtest ----------------

def run_backtest(
    signals: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    cash0: float,
    max_pos: int,
    cost_bps: float,
    entry: str,
    config: dict,
    debug: bool = False,
    pyramid: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Trade]]:
    rules = (config.get("rules") or {})
    exit_cfg = (rules.get("exit") or {})
    rg = (config.get("risk_guards") or {})
    sizing = (config.get("sizing") or {})

    # Allocation mode from YAML (default = fixed_cash)
    alloc_mode = str(sizing.get("allocation_mode", "fixed_cash")).strip().lower()
    if alloc_mode not in {"fixed_cash", "confidence", "confidence_normalized"}:
        alloc_mode = "fixed_cash"

    # Allocation reference: portfolio (fixed initial slot) vs equity (slot scales with equity)
    alloc_ref = str(sizing.get("allocation_reference", "equity")).strip().lower()
    if alloc_ref not in {"portfolio", "equity"}:
        alloc_ref = "equity"

    # YAML exit/risk parameters
    tp_pct = exit_cfg.get("take_profit_pct", None)
    rsi_period = exit_cfg.get("rsi_period", None)
    rsi_min_to_sell = exit_cfg.get("rsi_min_to_sell", None)
    sma_window = exit_cfg.get("sma_window", None)
    hard_stop_pct = rg.get("hard_stop_pct", None)
    time_stop_days = rg.get("time_stop_days", None)

    bt_start = pd.to_datetime(start).normalize() if start else signals["date"].min()
    bt_end   = pd.to_datetime(end).normalize()   if end   else signals["date"].max()

    sig = signals[(signals["date"] >= bt_start) & (signals["date"] <= bt_end)].copy()
    if sig.empty: raise ValueError("No BUY signals in the selected date range.")

    tickers = sorted(sig["ticker"].unique().tolist())
    prices = load_prices_batch(tickers, bt_start, bt_end, debug=debug)
    indicators = precompute_indicators(prices, exit_cfg)

    days = pd.date_range(bt_start, bt_end, freq="B")
    cash = cash0
    positions: Dict[str, Dict] = {}
    trades: List[Trade] = []
    nav_rows = []

    # group signals by day
    by_day: Dict[pd.Timestamp, List[pd.Series]] = {}
    for r in sig.itertuples(index=False):
        d = pd.to_datetime(r.date).normalize()
        by_day.setdefault(d, [])
        if not any(x.ticker == r.ticker for x in by_day[d]):
            by_day[d].append(r)

    half_cost = cost_bps / 2.0 / 1e4
    initial_slot = cash0 / max_pos  # fixed slot size if alloc_ref == "portfolio"

    if debug:
        print(f"[DEBUG] simulate | exits={exit_cfg} | guards={rg} | alloc_mode={alloc_mode} | alloc_ref={alloc_ref}")

    for d in days:
        # -------- exits --------
        to_close: List[str] = []
        for t, p in positions.items():
            dfp = prices.get(t)
            cl = latest_close(dfp, d)
            if cl is None: continue

            # time stop
            if time_stop_days is not None and (d.date() - p["entry_day"].date()).days >= int(time_stop_days):
                px = next_open(dfp, d - pd.Timedelta(days=1)) if entry == "next_open" else cl
                if px is not None:
                    cash += p["qty"] * px * (1 - half_cost)
                    trades.append(Trade(p["entry_day"], d, t, p["qty"], p["entry_px"], px, "time", p["qty"]*(px-p["entry_px"])))
                    to_close.append(t); continue

            # hard stop (% vs entry)
            if hard_stop_pct is not None and cl <= p["entry_px"] * (1 + float(hard_stop_pct)/100.0):
                px = next_open(dfp, d) if entry == "next_open" else cl
                if px is not None:
                    cash += p["qty"] * px * (1 - half_cost)
                    trades.append(Trade(p["entry_day"], d + pd.Timedelta(days=1), t, p["qty"], p["entry_px"], px, "stop", p["qty"]*(px-p["entry_px"])))
                    to_close.append(t); continue

            # take profit
            if tp_pct is not None and cl >= p["entry_px"] * (1 + float(tp_pct)/100.0):
                px = next_open(dfp, d) if entry == "next_open" else cl
                if px is not None:
                    cash += p["qty"] * px * (1 - half_cost)
                    trades.append(Trade(p["entry_day"], d + pd.Timedelta(days=1), t, p["qty"], p["entry_px"], px, "take", p["qty"]*(px-p["entry_px"])))
                    to_close.append(t); continue

            # RSI exit
            if (rsi_period and rsi_min_to_sell is not None) and ("rsi" in indicators.get(t, {})):
                rsi_series = indicators[t]["rsi"]
                idx = rsi_series.index.searchsorted(d, side="right") - 1
                if idx >= 0 and float(rsi_series.iloc[idx]) >= float(rsi_min_to_sell):
                    px = next_open(dfp, d) if entry == "next_open" else cl
                    if px is not None:
                        cash += p["qty"] * px * (1 - half_cost)
                        trades.append(Trade(p["entry_day"], d + pd.Timedelta(days=1), t, p["qty"], p["entry_px"], px, "rsi_exit", p["qty"]*(px-p["entry_px"])))
                        to_close.append(t); continue

            # SMA exit: exit when Close ≥ SMA (recovery/mean reversion)
            if sma_window and ("sma" in indicators.get(t, {})):
                sma_series = indicators[t]["sma"]
                idx = sma_series.index.searchsorted(d, side="right") - 1
                if idx >= 0 and cl >= float(sma_series.iloc[idx]):
                    px = next_open(dfp, d) if entry == "next_open" else cl
                    if px is not None:
                        cash += p["qty"] * px * (1 - half_cost)
                        trades.append(Trade(p["entry_day"], d + pd.Timedelta(days=1), t, p["qty"], p["entry_px"], px, "sma_exit", p["qty"]*(px-p["entry_px"])))
                        to_close.append(t); continue

        for t in to_close:
            positions.pop(t, None)

        # -------- entries (three sizing modes) --------
        todays = by_day.get(d, [])
        free = max(0, max_pos - len(positions))
        if todays and free > 0:
            # mark-to-market equity and slot size
            mtm = sum((latest_close(prices.get(tt), d) or 0.0) * pp["qty"] for tt, pp in positions.items())
            equity = cash + mtm
            slot_size = initial_slot if alloc_ref == "portfolio" else (equity / max_pos)

            # collect candidates (dedupe) and their confidence
            seen = set()
            cands = []
            for r in todays:
                if r.ticker in seen: 
                    continue
                seen.add(r.ticker)
                if (not pyramid) and (r.ticker in positions):
                    continue
                # confidence default 1.0 if missing/NaN
                c = 1.0
                if hasattr(r, "confidence"):
                    try:
                        c = float(getattr(r, "confidence"))
                        if not (c == c):  # NaN
                            c = 1.0
                    except Exception:
                        c = 1.0
                c = max(0.0, min(1.0, c))  # clamp
                cands.append((r, c))

            # sort by confidence descending (applies to all modes)
            cands.sort(key=lambda rc: rc[1], reverse=True)
            selected = cands[:free]

            if alloc_mode == "fixed_cash":
                # Equal-weight per-slot allocation (ignores confidence)
                for (r, c) in selected:
                    dfp = prices.get(r.ticker)
                    if dfp is None or dfp.empty: continue
                    px = next_open(dfp, d) if entry == "next_open" else latest_close(dfp, d)
                    if px is None: continue
                    alloc_i = slot_size
                    qty = math.floor((alloc_i * (1 - half_cost)) / px)
                    if qty <= 0: continue
                    cost = qty * px * (1 + half_cost)
                    if cost > cash:  # skip if insufficient cash
                        continue
                    cash -= cost
                    positions[r.ticker] = {"qty": qty, "entry_px": px, "entry_day": d}
                    if debug:
                        print(f"[DEBUG {d.date()}] enter {r.ticker}: mode=fixed_cash, ref={alloc_ref}, alloc≈{alloc_i:.2f}, qty={qty}, px={px:.2f}, cash→{cash:.2f}")

            elif alloc_mode == "confidence":
                # alloc = slot_size * Ci  (conservative; may leave cash idle)
                for (r, c) in selected:
                    dfp = prices.get(r.ticker)
                    if dfp is None or dfp.empty: continue
                    px = next_open(dfp, d) if entry == "next_open" else latest_close(dfp, d)
                    if px is None: continue
                    alloc_i = slot_size * c
                    qty = math.floor((alloc_i * (1 - half_cost)) / px)
                    if qty <= 0: continue
                    cost = qty * px * (1 + half_cost)
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[r.ticker] = {"qty": qty, "entry_px": px, "entry_day": d}
                    if debug:
                        print(f"[DEBUG {d.date()}] enter {r.ticker}: mode=confidence, ref={alloc_ref}, conf={c:.3f}, alloc≈{alloc_i:.2f}, qty={qty}, px={px:.2f}, cash→{cash:.2f}")

            else:  # alloc_mode == "confidence_normalized"
                # Day budget ≈ slot_size * k, weights ~ Ci / ΣC
                k = len(selected)
                sum_c = sum(c for _, c in selected) or float(k)
                for (r, c) in selected:
                    dfp = prices.get(r.ticker)
                    if dfp is None or dfp.empty: continue
                    px = next_open(dfp, d) if entry == "next_open" else latest_close(dfp, d)
                    if px is None: continue
                    weight = (c / sum_c)
                    alloc_i = slot_size * k * weight
                    qty = math.floor((alloc_i * (1 - half_cost)) / px)
                    if qty <= 0: continue
                    cost = qty * px * (1 + half_cost)
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[r.ticker] = {"qty": qty, "entry_px": px, "entry_day": d}
                    if debug:
                        print(f"[DEBUG {d.date()}] enter {r.ticker}: mode=confidence_normalized, ref={alloc_ref}, conf={c:.3f}, w={weight:.3f}, alloc≈{alloc_i:.2f}, qty={qty}, px={px:.2f}, cash→{cash:.2f}")

        # NAV
        mtm = sum((latest_close(prices.get(t), d) or 0.0) * p["qty"] for t, p in positions.items())
        nav_rows.append({"date": d.date(), "nav": cash + mtm, "cash": cash, "n_pos": len(positions)})

    # EOP mark
    for t, p in list(positions.items()):
        px = latest_close(prices.get(t), bt_end)
        if px is not None:
            trades.append(Trade(p["entry_day"], bt_end, t, p["qty"], p["entry_px"], px, "eop", p["qty"]*(px-p["entry_px"])))

    nav_df = pd.DataFrame(nav_rows)
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    return nav_df, trades_df, trades

# ---------------- Summary ----------------

def summarize(nav_df: pd.DataFrame, trades_df: pd.DataFrame) -> str:
    if nav_df.empty: return "No NAV data."
    nav_ret = nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1
    nav_df["ret"] = nav_df["nav"].pct_change().fillna(0.0)
    avg = nav_df["ret"].mean(); vol = nav_df["ret"].std()
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
    ap = argparse.ArgumentParser(description="Backtest BUY signals using YAML exits; allocation modes + reference.")
    ap.add_argument("--signals", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--cash", type=float, default=100000.0)
    ap.add_argument("--max-pos", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--entry", choices=["next_open","close"], default="next_open")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--pyramid", action="store_true")
    args = ap.parse_args()

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
        pyramid=args.pyramid,
    )
    base = args.out_prefix or os.path.splitext(args.signals)[0]
    nav_path = f"{base}_nav.csv"; trades_path = f"{base}_trades.csv"
    os.makedirs(os.path.dirname(nav_path) or ".", exist_ok=True)
    nav_df.to_csv(nav_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    print("\n=== Backtest Summary ===")
    print(summarize(nav_df, trades_df))
    print(f"\nSaved NAV → {nav_path}")
    print(f"Saved trades → {trades_path}")

if __name__ == "__main__":
    main()

