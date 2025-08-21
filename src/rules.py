from __future__ import annotations
import pandas as pd
from src.indicators import rsi, zscore, pct_drop_over_n

def evaluate_entry(df: pd.DataFrame, entry_cfg: dict) -> tuple[bool, dict]:
    """
    Check the entry rule:
      - 10-day drop <= -min_drop_pct
      - RSI(rsi_period) <= rsi_max_to_buy
      - zscore vs SMA(sma_window) <= zscore_max_to_buy
    Returns (ok, info_dict)
    """
    # Ensure we have enough history
    needed = max(
        entry_cfg["rsi_period"],
        entry_cfg["sma_window"],
        entry_cfg["drop_lookback_days"]
    ) + 5
    if len(df) < needed or "Close" not in df:
        return False, {"reason": "insufficient_history", "values": {}}

    close = df["Close"]

    drop_lookback = entry_cfg["drop_lookback_days"]
    min_drop_pct  = float(entry_cfg["min_drop_pct"])
    rsi_period    = entry_cfg["rsi_period"]
    rsi_max_buy   = float(entry_cfg["rsi_max_to_buy"])
    sma_window    = entry_cfg["sma_window"]
    zscore_max    = float(entry_cfg["zscore_max_to_buy"])

    # Compute metrics (use last available values)
    drop_val = pct_drop_over_n(close, n=drop_lookback).iloc[-1]
    rsi_val  = rsi(close, period=rsi_period).iloc[-1]
    z_val    = zscore(close, window=sma_window).iloc[-1]

    # Conditions
    cond_drop = drop_val <= -abs(min_drop_pct)
    cond_rsi  = rsi_val  <= rsi_max_buy
    cond_z    = z_val    <= zscore_max

    ok = bool(cond_drop and cond_rsi and cond_z)

    info = {
        "reason": "entry_ok" if ok else "entry_no_match",
        "values": {
            f"drop{drop_lookback}d_pct": round(float(drop_val), 2),
            f"rsi{rsi_period}": round(float(rsi_val), 2),
            f"z{sma_window}": round(float(z_val), 2),
            "last_close": round(float(close.iloc[-1]), 4),
        },
        "thresholds": {
            "min_drop_pct": -abs(min_drop_pct),
            "rsi_max_to_buy": rsi_max_buy,
            "zscore_max_to_buy": zscore_max,
        }
    }
    return ok, info



import math
from src.indicators import sma

def compute_confidence(df: pd.DataFrame, entry_cfg: dict, conf_cfg: dict) -> tuple[float, dict]:
    """
    Build a confidence score C in [0,1] from how 'extreme' the signal is.
    Components: DropScore, RSIScore, ZScoreDepth, VolumeSpike (+ regime penalty).
    Returns (C, components_dict).
    """
    close = df["Close"]
    rsi_period    = entry_cfg["rsi_period"]
    sma_window    = entry_cfg["sma_window"]
    drop_lookback = entry_cfg["drop_lookback_days"]

    # Last values used in evaluate_entry (already computed there)
    from src.indicators import rsi, zscore, pct_drop_over_n
    drop_val = pct_drop_over_n(close, n=drop_lookback).iloc[-1]   # e.g., -18.3 (%)
    rsi_val  = rsi(close, period=rsi_period).iloc[-1]              # e.g., 24.5
    z_val    = zscore(close, window=sma_window).iloc[-1]           # e.g., -2.6

    # Volume spike (cap to avoid crazy values)
    vol = df["Volume"].iloc[-1] if "Volume" in df.columns else math.nan
    vol_ma50 = df["Volume"].rolling(50, min_periods=20).mean().iloc[-1] if "Volume" in df.columns else math.nan
    vol_ratio = float(vol / vol_ma50) if (vol_ma50 and not math.isnan(vol_ma50) and vol_ma50 > 0) else 1.0

    # Thresholds / caps from config
    w = conf_cfg["weights"]
    rsi_floor = float(conf_cfg.get("rsi_floor", 10))
    drop_hard_max = float(conf_cfg.get("drop_hard_max", -25))   # negative
    z_hard_min = float(conf_cfg.get("zscore_hard_min", -4))     # negative
    vol_cap = float(conf_cfg.get("volume_cap", 2.0))
    regime_penalty = float(conf_cfg.get("regime_penalty_below_200dma", 0.8))

    # ENTRY thresholds
    min_drop_pct  = float(entry_cfg["min_drop_pct"])
    rsi_max_buy   = float(entry_cfg["rsi_max_to_buy"])
    zscore_max    = float(entry_cfg["zscore_max_to_buy"])       # e.g., -2.0

    # 1) DropScore: more negative drop -> higher score
    # Map drop from [-min_drop_pct, drop_hard_max] to [0,1]
    d = -float(drop_val)  # positive magnitude
    d_th = abs(min_drop_pct)
    d_cap = abs(drop_hard_max)
    if drop_val <= -d_th:
        drop_score = (min(max(d, d_th), d_cap) - d_th) / (d_cap - d_th) if d_cap > d_th else 1.0
    else:
        drop_score = 0.0

    # 2) RSIScore: lower RSI below threshold -> higher score
    # Map rsi in [rsi_floor, rsi_max_buy] to [1,0], clamp to floor
    if rsi_val <= rsi_max_buy:
        rsi_clamped = max(rsi_floor, float(rsi_val))
        rsi_score = (rsi_max_buy - rsi_clamped) / (rsi_max_buy - rsi_floor) if rsi_max_buy > rsi_floor else 1.0
    else:
        rsi_score = 0.0

    # 3) ZScoreDepth: z <= zscore_max (e.g., -2) gets score, deeper (e.g., -3, -4) increases up to cap
    if z_val <= zscore_max:
        z_clamped = max(float(z_hard_min), float(z_val))
        z_score = (zscore_max - z_clamped) / (zscore_max - z_hard_min) if zscore_max > z_hard_min else 1.0
    else:
        z_score = 0.0

    # 4) VolumeSpike: >1 means heavier volume; cap to vol_cap, map [1, vol_cap] -> [0,1]
    vr = min(max(vol_ratio, 1.0), vol_cap)
    vol_score = (vr - 1.0) / (vol_cap - 1.0) if vol_cap > 1.0 else 0.0

    # Weighted sum
    raw = (
        w["drop"]   * drop_score +
        w["rsi"]    * rsi_score +
        w["zscore"] * z_score +
        w["volume"] * vol_score
    )

    # Regime penalty if below 200DMA
    s200 = sma(close, 200).iloc[-1]
    penalty = regime_penalty if float(close.iloc[-1]) < float(s200) else 1.0

    C = max(0.0, min(1.0, raw * penalty))

    components = {
        "drop_score": round(drop_score, 3),
        "rsi_score": round(rsi_score, 3),
        "z_score": round(z_score, 3),
        "vol_score": round(vol_score, 3),
        "penalty": penalty,
        "vol_ratio": round(vr, 2),
    }
    return C, components