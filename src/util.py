import csv
import os
from datetime import datetime, timezone, timedelta

def read_watchlist(path):
    """Read watchlist.csv into a list of dicts."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out

def read_positions(path):
    """Read positions.csv into a list of dicts."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out

def append_signal_row(path, row, header):
    """Append a signal row (dict) to signals_log.csv. Create file if missing."""
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def has_signal_today(path, ticker, signal_type):
    """Check if a signal for ticker+type already exists today in signals_log.csv."""
    if not os.path.exists(path):
        return False
    today = now_eu().strftime("%Y-%m-%d")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row.get("ticker") == ticker
                and row.get("signal_type") == signal_type
                and row.get("timestamp", "").startswith(today)
            ):
                return True
    return False

def now_eu():
    """Return current time in Central European Time (CET/CEST)."""
    return datetime.now(timezone.utc) + timedelta(hours=1)
