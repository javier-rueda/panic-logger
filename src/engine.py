from __future__ import annotations

import json
from typing import Optional
import pandas as pd
import yfinance as yf

from src.util import (
    read_watchlist,
    read_positions,
    append_signal_row,
    has_signal_today,
    now_eu,
)
from src.data_loader import get_history
from src.indicators import rsi, zscore, pct_drop_over_n
from src.rules import evaluate_entry, compute_confidence


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
]


def _normalize_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def run_once(
    cfg: dict,
    asof: Optional[str] = None,
    quiet: bool = False,
    override_log_path: Optional[str] = None,
):
    """
    Run the scanner once.

    - If `asof` is provided (YYYY-MM-DD), we "time-travel":
        * truncate each ticker's history to <= asof (no look-ahead)
        * only generate signals if a bar exists ON `asof`
        * do NOT apply today's open-positions / max-open-trades filters
        * stamp the log with "YYYY-MM-DD 00:00:00"
    - If `asof` is None, it's the normal live/today run.

    `quiet=True` suppresses per-ticker prints and summaries (useful for range builds).
    `override_log_path` writes logs to a custom CSV instead of cfg["files"]["signals_log_csv"].
    """
    wl_path = cfg["universe"]["watchlist_csv"]
    pos_path = cfg["files"]["positions_csv"]
    log_path = override_log_path if override_log_path else cfg["files"]["signals_log_csv"]

    watchlist = read_watchlist(wl_path)
    positions = read_positions(pos_path)

    if asof:
        held = set()
        open_trades = 0
        header = f"=== Panic Logger — AS OF {asof} ==="
    else:
        held = {p["ticker"] for p in positions}
        open_trades = len(positions)
        header = "=== Panic Logger (live) ==="

    if not quiet:
        print(header)
        print(f"Watchlist tickers: {len(watchlist)}\n")

    hist_cfg = cfg["data_source"]["history"]
    period = hist_cfg.get("period", "220d")
    interval = hist_cfg.get("interval", "1d")

    entry_cfg = cfg["rules"]["entry"]
    sizing_cfg = cfg.get("sizing", {})
    conf_cfg = cfg.get("confidence", {})

    # Sizing (used only for reporting; nothing is executed)
    portfolio_value = float(sizing_cfg.get("portfolio_value", 0.0))
    available_cash_pct = float(sizing_cfg.get("available_cash_pct", 0.0))
    base_risk_per_trade_pct = float(sizing_cfg.get("base_risk_per_trade_pct", 0.0075))
    max_cash_per_trade_pct = float(sizing_cfg.get("max_cash_per_trade_pct", 0.08))
    stop_pct = float(sizing_cfg.get("stop_pct", -12.0))
    max_open_trades = int(sizing_cfg.get("max_open_trades", 8))

    available_cash = portfolio_value * available_cash_pct

    asof_ts = _normalize_date(asof) if asof else None
    log_stamp = f"{asof} 00:00:00" if asof else now_eu().strftime("%Y-%m-%d %H:%M:%S")

    buy_summary = []

    for w in watchlist:
        tkr = (w.get("ticker") or "").strip()
        market = w.get("market", "NA")
        if not tkr:
            continue

        df = get_history(tkr, period=period, interval=interval, asof=asof)
        if df.empty or "Close" not in df:
            if not quiet:
                print(f" - {tkr}: data not available")
            continue

        # --- ASOF truncation (no look-ahead) ---
        if asof_ts is not None:
            df = df[df.index.normalize() <= asof_ts]
            if df.empty:
                if not quiet:
                    print(f" - {tkr}: no data up to {asof}")
                continue
            last_bar_day = df.index[-1].normalize()
            if last_bar_day != asof_ts:
                # Non-trading day for this ticker at `asof` → show metrics (for transparency) and skip signals
                if not quiet:
                    last_close = float(df["Close"].iloc[-1])
                    rsi_val = float(rsi(df["Close"], period=entry_cfg["rsi_period"]).iloc[-1])
                    drop_val = float(pct_drop_over_n(df["Close"], n=entry_cfg["drop_lookback_days"]).iloc[-1])
                    z_val = float(zscore(df["Close"], window=entry_cfg["sma_window"]).iloc[-1])
                    print(
                        f" - {tkr}: (no bar on {asof}, last {last_bar_day.date()}) "
                        f"Close={last_close:.2f} | RSI({entry_cfg['rsi_period']})={rsi_val:.1f} | "
                        f"Drop{entry_cfg['drop_lookback_days']}d={drop_val:.1f}% | z{entry_cfg['sma_window']}={z_val:.2f}"
                    )
                continue
        # ---------------------------------------

        close = float(df["Close"].iloc[-1])
        rsi_val = float(rsi(df["Close"], period=entry_cfg["rsi_period"]).iloc[-1])
        drop_val = float(pct_drop_over_n(df["Close"], n=entry_cfg["drop_lookback_days"]).iloc[-1])
        z_val = float(zscore(df["Close"], window=entry_cfg["sma_window"]).iloc[-1])

        if not quiet:
            print(
                f" - {tkr}: Close={close:.2f} | "
                f"RSI({entry_cfg['rsi_period']})={rsi_val:.1f} | "
                f"Drop{entry_cfg['drop_lookback_days']}d={drop_val:.1f}% | "
                f"z{entry_cfg['sma_window']}={z_val:.2f}"
            )

        ok, info = evaluate_entry(df, entry_cfg)
        if not ok:
            continue

        # Skip duplicate/constraints checks only in live mode
        if not asof:
            if tkr in held:
                if not quiet:
                    print("     -> BUY candidate ✅ but SKIPPED (already holding this ticker).")
                continue
            if open_trades >= max_open_trades:
                if not quiet:
                    print(f"     -> BUY candidate ✅ but SKIPPED (max open trades {max_open_trades} reached).")
                continue
            if has_signal_today(log_path, tkr, "BUY"):
                if not quiet:
                    print("     -> BUY candidate ✅ but SKIPPED (already logged today).")
                continue

        # Confidence & sizing (for reporting/log enrichment)
        C, comps = compute_confidence(df, entry_cfg, conf_cfg)
        risk_multiplier = 0.6 + 0.8 * C
        risk_per_trade = portfolio_value * base_risk_per_trade_pct * risk_multiplier

        stop_fraction = abs(stop_pct) / 100.0 if stop_pct != 0 else 0.12
        cash_needed = risk_per_trade / stop_fraction if stop_fraction > 0 else 0.0
        cap_per_trade = portfolio_value * max_cash_per_trade_pct
        cash_alloc = min(cash_needed, cap_per_trade, available_cash) if available_cash > 0 else 0.0
        cash_pct_of_liquidity = (cash_alloc / available_cash) if available_cash > 0 else 0.0

        # Market cap ONLY for BUY candidates (slow path, best-effort)
        mc_val = None
        mc_str = "N/A"
        try:
            ti = yf.Ticker(tkr).info
            mc = ti.get("marketCap")
            mc_val = float(mc) if mc else None
            mc_str = f"{mc/1e9:.1f}B" if mc else "N/A"
        except Exception:
            pass

        if not quiet:
            print("     -> BUY candidate ✅ (meets rule)")
            print(
                f"        CONFIDENCE C={C:.2f}  "
                f"[drop_s={comps['drop_score']}, rsi_s={comps['rsi_score']}, "
                f"z_s={comps['z_score']}, vol_s={comps['vol_score']}]"
            )
            print(f"        Suggested allocation: ~{cash_pct_of_liquidity*100:.1f}% of available cash")
            print(f"        MarketCap: {mc_str}")

        # Log to CSV
        values_blob = {
            "drop_pct": round(drop_val, 2),
            "rsi": round(rsi_val, 2),
            "zscore": round(z_val, 2),
            "confidence": round(C, 3),
            "components": comps,
            "alloc_pct_of_liquidity": round(cash_pct_of_liquidity, 4),
            "stop_pct_assumed": stop_pct,
            "market_cap": mc_val,
        }
        row = {
            "timestamp": log_stamp,
            "ticker": tkr,
            "market": mc_val,
            "signal_type": "BUY",
            "price": f"{close:.4f}",
            "reason": info["reason"],
            "values": json.dumps(values_blob, separators=(",", ":")),
            "rule_id": "ENTRY_V1",
            "notes": "",
        }
        append_signal_row(log_path, row, SIGNAL_HEADER)

        # For end-of-run summary (only if not quiet)
        buy_summary.append(
            {
                "ticker": tkr,
                "close": close,
                "confidence": C,
                "alloc_pct_liq": cash_pct_of_liquidity,
                "market_cap_str": mc_str,
                "drop": values_blob["drop_pct"],
                "rsi": values_blob["rsi"],
                "z": values_blob["zscore"],
            }
        )

        if quiet and asof:
            print(f"[{asof}] processed")    

    # Summary
    if not quiet:
        if buy_summary:
            print("\n=== BUY CANDIDATES SUMMARY ===")
            for b in buy_summary:
                print(
                    f" {b['ticker']}: Close={b['close']:.2f} | "
                    f"C={b['confidence']:.2f} | Alloc~{b['alloc_pct_liq']*100:.1f}% cash | "
                    f"MC={b['market_cap_str']} | Drop={b['drop']:.1f}% | RSI={b['rsi']:.1f} | z={b['z']:.2f}"
                )
        else:
            print("\nNo BUY candidates.")

        # Positions printout (skip in ASOF because they're today's holdings)
        if not asof:
            print(f"\nOpen positions: {len(positions)}")
            for p in positions:
                print(f" - {p['ticker']} @ {p['entry_price']} on {p['entry_date']}")
        print("Run complete.")