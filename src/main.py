import argparse
import yaml
from src.engine import run_once

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="panic-logger")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--asof", type=str, default=None, help="Run as of YYYY-MM-DD (no look-ahead)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-ticker prints (useful for range builds)")
    parser.add_argument("--log-to", type=str, default=None, help="Override output CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_once(cfg, asof=args.asof, quiet=args.quiet, override_log_path=args.log_to)