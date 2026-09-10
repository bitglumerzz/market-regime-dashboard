"""Live Pine ground-truth harvester via TV MCP bridge.

For each (ticker, timeframe), asks the bridge to:
  1. Switch TV Desktop to that chart
  2. Extract Pine indicator labels (e.g. Elliott Wave wave labels)
  3. Save JSON output to tv_ground_truth/{symbol}_{tf}_live.json
  4. Also save OHLCV as CSV so the labels are co-located with prices

This closes the main 4h calibration gap — currently almost all
tv_ground_truth/ files are 1d.

Usage:
    python harvest_pine_live.py --once               # one cycle
    python harvest_pine_live.py --loop               # daily refresh
    python harvest_pine_live.py --ticker BTC-USD --tf 4h  # one chart

Prerequisites (run on host where TV Desktop lives):
  1. TV Desktop launched: open -a "TradingView" --args --remote-debugging-port=9222
  2. tv_bridge.py running: python tv_bridge.py
  3. Your Pine wave indicator added to BTC/ETH/HYPE charts on 1d AND 4h
  4. (in Streamlit Docker:) this script can run inside the container — it
     submits to bridge via shared volume.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tv_bridge_client import (
    request_pine_labels, request_pine_lines, is_bridge_available,
)

GT_DIR = "tv_ground_truth"
DEFAULT_TICKERS = ["BTC-USD", "ETH-USD", "HYPE-USD"]
DEFAULT_TFS = ["1d", "4h"]
# Common names of Elliott Wave Pine indicators users tend to install.
# Pass --study-filter to override.
DEFAULT_STUDY_FILTERS = ["Elliott", "Wave", "EW"]


def _setup_logging() -> logging.Logger:
    log = logging.getLogger("harvest_pine_live")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(h)
    return log


def _harvest_one(ticker: str, tf: str, study_filter: str,
                  log: logging.Logger) -> dict:
    """Harvest Pine labels + lines for one (ticker, tf)."""
    os.makedirs(GT_DIR, exist_ok=True)
    # Sanitize ticker for filename
    safe_ticker = ticker.replace("/", "-").replace(":", "-")
    out_labels = os.path.abspath(os.path.join(GT_DIR,
                                                f"{safe_ticker}_{tf}_pine_labels.json"))
    out_lines  = os.path.abspath(os.path.join(GT_DIR,
                                                f"{safe_ticker}_{tf}_pine_lines.json"))

    log.info("→ %s %s (filter=%r)", ticker, tf, study_filter)
    t0 = time.time()
    labels_ok = request_pine_labels(ticker, tf, out_labels,
                                       study_filter=study_filter, timeout=45.0)
    lines_ok = request_pine_lines(ticker, tf, out_lines,
                                     study_filter=study_filter, timeout=45.0)
    elapsed = time.time() - t0

    n_labels = 0
    if labels_ok and os.path.isfile(out_labels):
        try:
            file_size = os.path.getsize(out_labels)
            if file_size > 0:
                with open(out_labels) as f:
                    content = f.read()
                if content.strip():
                    data = json.loads(content)
                    # Different MCP CLI versions may wrap differently
                    if isinstance(data, dict):
                        for v in data.values():
                            if isinstance(v, list):
                                n_labels += len(v)
                    elif isinstance(data, list):
                        n_labels = len(data)
        except json.JSONDecodeError as exc:
            log.warning("  empty/invalid JSON in %s: %s",
                         os.path.basename(out_labels), str(exc)[:80])
        except Exception as exc:
            log.warning("  count failed: %s", str(exc)[:80])

    log.info("  labels=%s lines=%s | n_labels=%d | %.1fs",
              "✓" if labels_ok else "✗",
              "✓" if lines_ok else "✗",
              n_labels, elapsed)
    return {
        "ticker": ticker, "tf": tf,
        "labels_ok": labels_ok, "lines_ok": lines_ok,
        "n_labels": n_labels,
        "labels_path": out_labels if labels_ok else None,
        "lines_path":  out_lines if lines_ok else None,
        "harvested_at": datetime.now(timezone.utc).isoformat(),
    }


def harvest_all(tickers: list[str], tfs: list[str],
                 study_filter: str,
                 log: logging.Logger) -> list[dict]:
    if not is_bridge_available():
        log.error("Bridge queue dir not available — is tv_bridge.py running "
                  "and bind-mounted? Set up Phase 8.1 first.")
        return []
    results = []
    for ticker in tickers:
        for tf in tfs:
            r = _harvest_one(ticker, tf, study_filter, log)
            results.append(r)
            time.sleep(0.5)  # gentle gap between requests
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", action="append",
                    help="single ticker (default: BTC-USD ETH-USD HYPE-USD)")
    p.add_argument("--tf", action="append",
                    help="single timeframe (default: 1d 4h)")
    p.add_argument("--study-filter", default="Elliott",
                    help="substring of Pine indicator name (default 'Elliott')")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--once", action="store_true",
                    help="harvest once and exit (default)")
    g.add_argument("--loop", action="store_true",
                    help="run forever, harvesting once a day")
    args = p.parse_args()

    log = _setup_logging()
    tickers = args.ticker or DEFAULT_TICKERS
    tfs = args.tf or DEFAULT_TFS

    if args.loop:
        log.info("Pine harvester in --loop mode (once a day)")
        while True:
            harvest_all(tickers, tfs, args.study_filter, log)
            time.sleep(24 * 3600)
    else:
        results = harvest_all(tickers, tfs, args.study_filter, log)
        ok_count = sum(1 for r in results if r["labels_ok"])
        print()
        print(f"Done: {ok_count}/{len(results)} (ticker, tf) pairs succeeded")
        for r in results:
            status = "✓" if r["labels_ok"] else "✗"
            print(f"  {status} {r['ticker']} {r['tf']:>3}: "
                  f"{r['n_labels']} labels, file={r['labels_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
