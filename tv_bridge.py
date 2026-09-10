"""Host bridge between Docker-bound Streamlit and TV MCP CLI on macOS.

Architecture:
   Streamlit (in Docker)             Mac host (this script)
        │                                  │
   writes request JSON                     │
   → tv_bridge_queue/<id>.req ────────────►│
                                           │
                              tv_bridge.py (this) — polls every 2s
                                           │
                              subprocess: node ~/Code/tradingview-mcp/src/cli/index.js
                                           │
                                  TV Desktop CDP :9222
                                           │
                              writes output to specified path
                                           │
   reads output ◄────────  predictions/screenshots/<id>.png  or
                           tv_ground_truth/<sym>_<tf>_live.csv

The queue dir is bind-mounted into Docker so Streamlit can write requests.
Output paths are also bind-mounted so Streamlit can read results.

Request file format (JSON):
{
  "type":   "screenshot" | "pine_labels" | "pine_lines" | "pine_boxes" | "ohlcv" | "state",
  "symbol": "BINANCE:BTCUSDT",        # required for non-state requests
  "interval": "240" | "1D" | "60",    # required for non-state requests
  "study_filter": "Elliott Wave",     # optional, for pine_*
  "region": "chart",                  # optional, for screenshot
  "out": "/abs/path/to/output.png"    # required — where to save
}

CLI: python tv_bridge.py [--queue-dir DIR] [--interval-sec N] [--once]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# -----------------------------------------------------------------------------
# Defaults — adjust for your machine
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_QUEUE_DIR = str(PROJECT_DIR / "tv_bridge_queue")
DEFAULT_TV_MCP_CLI = str(Path.home() / "Code/tradingview-mcp/src/cli/index.js")
DEFAULT_TV_SCREENSHOT_DIR = str(Path.home() / "Code/tradingview-mcp/screenshots")
DEFAULT_POLL_INTERVAL = 2.0     # seconds
DEFAULT_TIMEOUT = 90            # default per CLI call
# Screenshot needs more time — chart navigation + render + CDP capture
SCREENSHOT_TIMEOUT = 180        # seconds for screenshot-specific calls
LOG_PATH = str(PROJECT_DIR / "tv_bridge.log")


def _translate_path(p: str) -> str:
    """Translate Docker-side absolute paths to host-side paths.

    Streamlit container has cwd=/app, so requests carry paths like
    /app/predictions/screenshots/xxx.png. The bridge runs from the project
    root on host (the bind-mount source). We rewrite /app/<rest> → <rest>
    so it lands in the same shared directory on disk.
    """
    if not p:
        return p
    if p.startswith("/app/"):
        return p[len("/app/"):]
    if p == "/app":
        return "."
    return p


def _setup_logging() -> logging.Logger:
    log = logging.getLogger("tv_bridge")
    log.setLevel(logging.INFO)
    if log.handlers:
        return log
    # stdout AND file
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    try:
        fh = logging.FileHandler(LOG_PATH)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except Exception:
        pass
    return log


# -----------------------------------------------------------------------------
# CLI runner
# -----------------------------------------------------------------------------
def _run_tv_cli(args: list[str], cli_path: str,
                 timeout: int = DEFAULT_TIMEOUT) -> tuple[int, str, str]:
    """Run the TV MCP CLI. Returns (returncode, stdout, stderr)."""
    cmd = ["node", cli_path] + args
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except FileNotFoundError as exc:
        return 127, "", f"node not found: {exc}"


# How many days of history to show per TF — picked so the rightmost bars
# fill ~70% of screen at default TV chart size.
_RANGE_DAYS_PER_TF = {
    "1D":  365,  "D":   365,
    "240":  60,  "4H":   60,
    "60":   21,  "1H":   21,
    "15":    7,
    "5":     3,
}


def _norm_interval(s: str) -> str:
    """Normalize TV interval strings to a canonical form for comparison.
    '240' / '4H' / '4h' / 'H4' → '240'; '1D' / 'D' / 'd' / '1440' → '1D'.
    """
    if not s:
        return ""
    s = s.strip().upper().replace("H", "").replace("M", "").replace("D", "D")
    aliases = {
        "1D": "1D", "D": "1D", "1440": "1D",
        "240": "240", "4": "240", "H4": "240",
        "60":  "60",  "1":  "60", "H1": "60",
        "15":  "15",  "5":  "5",
    }
    return aliases.get(s, s)


def _autofit_chart(interval: str, cli_path: str, log: logging.Logger) -> None:
    """Reset chart range so the rightmost bars are visible after symbol/TF switch.

    TV chart can show the wrong period after symbol/timeframe change — e.g.
    you switch from BTC 1d to DOGE 4h and end up zoomed to 2024 data. We
    explicitly set the visible range via `tv range --from <ts> --to <ts>`.
    This is more reliable than keyboard shortcuts (which can have side
    effects — e.g. Alt+S in TV is NOT a standard autofit hotkey).

    Best-effort — silent failure doesn't break the flow.
    """
    days = _RANGE_DAYS_PER_TF.get(interval.upper(), 90)
    now_ts = int(datetime.now(timezone.utc).timestamp())
    from_ts = now_ts - days * 86400
    rc, _, err = _run_tv_cli(["range", "--from", str(from_ts),
                               "--to", str(now_ts)],
                              cli_path, timeout=10)
    if rc == 0:
        log.info("  ✓ autofit range=%dd", days)
        return
    # Fallback: scroll to today (X-axis only)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rc, _, _ = _run_tv_cli(["scroll", today], cli_path, timeout=8)
    if rc == 0:
        log.info("  ✓ autofit fallback: scroll→today")
        return
    log.warning("  ⚠ autofit didn't apply (%s)", err.strip()[:120])


def _set_chart_state(symbol: str, interval: str, cli_path: str,
                      log: logging.Logger) -> bool:
    """Switch the TV chart to (symbol, interval). VERIFIES the switch worked
    via `tv state` after — TV CLI may return rc=0 even when symbol didn't load."""
    if symbol:
        rc, _, err = _run_tv_cli(["symbol", symbol], cli_path)
        if rc != 0:
            log.error("  ! tv symbol %s failed: %s", symbol, err.strip()[:160])
            return False
    if interval:
        rc, _, err = _run_tv_cli(["timeframe", interval], cli_path)
        if rc != 0:
            log.error("  ! tv timeframe %s failed: %s", interval, err.strip()[:160])
            return False

    # Wait for TV to actually load new chart + redraw Pine indicators.
    # 4 seconds covers: symbol fetch (~1s) + chart redraw (~1s) +
    # Pine indicator recompute (~1.5s). Previous 1.2s was too tight.
    time.sleep(4.0)

    # Verify the symbol AND interval actually changed. TV CLI returns rc=0
    # even for invalid switches sometimes — without this check we screenshot
    # the PREVIOUS chart and save it under the new prediction's ID.
    if symbol or interval:
        rc, stdout, _ = _run_tv_cli(["state"], cli_path, timeout=15)
        if rc == 0:
            try:
                state = json.loads(stdout)
                if isinstance(state.get("result"), dict):
                    state = state["result"]
                current_symbol = state.get("symbol") or ""
                current_interval = (state.get("interval")
                                     or state.get("timeframe")
                                     or state.get("resolution")
                                     or "")
                want_sym = (symbol or "").split(":")[-1].upper()
                got_sym = str(current_symbol).split(":")[-1].upper()
                if want_sym and got_sym and want_sym != got_sym:
                    log.warning("  ⚠ symbol mismatch: want=%s got=%s — retrying",
                                 want_sym, got_sym)
                    _run_tv_cli(["symbol", symbol], cli_path)
                    time.sleep(5.0)
                want_int = _norm_interval(interval)
                got_int = _norm_interval(str(current_interval))
                if want_int and got_int and want_int != got_int:
                    log.warning("  ⚠ interval mismatch: want=%s got=%s — retrying",
                                 want_int, got_int)
                    _run_tv_cli(["timeframe", interval], cli_path)
                    time.sleep(4.0)
            except (json.JSONDecodeError, Exception) as exc:
                log.debug("  state parse failed: %s", exc)

    # AUTO-FIT: explicitly set visible range so chart shows recent bars
    # (per-TF range — see _RANGE_DAYS_PER_TF). Done AFTER verification so
    # we don't accidentally autofit an old chart.
    _autofit_chart(interval, cli_path, log)
    time.sleep(0.8)
    return True


# -----------------------------------------------------------------------------
# Request handlers
# -----------------------------------------------------------------------------
def _handle_screenshot(req: dict, cli_path: str, screenshot_dir: str,
                        log: logging.Logger) -> bool:
    """Take a screenshot and copy to req['out']."""
    symbol = req.get("symbol", "")
    interval = req.get("interval", "")
    region = req.get("region", "chart")
    out = _translate_path(req.get("out", ""))
    if not out:
        log.error("  ! screenshot request missing 'out'")
        return False
    if not _set_chart_state(symbol, interval, cli_path, log):
        return False

    # Mark time BEFORE the call so we can find the new file by mtime
    pre_call = time.time()
    filename = f"bridge_{int(pre_call)}"
    args = ["screenshot", "--region", region, "--filename", filename]
    rc, stdout, stderr = _run_tv_cli(args, cli_path, timeout=SCREENSHOT_TIMEOUT)
    if rc != 0:
        log.error("  ! screenshot failed: %s", (stderr or stdout)[:160])
        return False

    # Strategy 1: parse TV CLI JSON output for the actual saved path
    src = None
    try:
        result = json.loads(stdout)
        # CLI typically returns {"success": true, "path": "..."} or
        # nests under a "result" key — handle both
        if isinstance(result, dict):
            cand = result.get("path") or result.get("filepath") or \
                   (result.get("result") or {}).get("path")
            if cand and os.path.isfile(cand):
                src = Path(cand)
    except json.JSONDecodeError:
        pass

    # Strategy 2: file with our prefix in screenshot_dir
    if src is None:
        candidates = list(Path(screenshot_dir).glob(f"{filename}.*"))
        if candidates:
            src = candidates[0]

    # Strategy 3: any file created in screenshot_dir AFTER the call (within 5s)
    if src is None and os.path.isdir(screenshot_dir):
        for f in Path(screenshot_dir).iterdir():
            if f.is_file() and f.stat().st_mtime >= pre_call - 1:
                src = f
                break

    if src is None or not src.exists():
        log.error("  ! screenshot file not found (tried JSON, prefix, mtime)")
        log.error("    stdout snippet: %s", stdout[:200])
        return False

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    shutil.copy2(src, out)
    log.info("  ✓ screenshot saved to %s (from %s)", out, src.name)
    return True


def _handle_pine_data(req: dict, cli_path: str, log: logging.Logger,
                       kind: str) -> bool:
    """kind ∈ {labels, lines, boxes, tables}. Extract Pine data → JSON file."""
    symbol = req.get("symbol", "")
    interval = req.get("interval", "")
    study_filter = req.get("study_filter", "")
    out = _translate_path(req.get("out", ""))
    if not out:
        log.error("  ! pine_%s request missing 'out'", kind)
        return False
    if not _set_chart_state(symbol, interval, cli_path, log):
        return False

    args = ["data", kind]
    if study_filter:
        args += ["--study-filter", study_filter]
    rc, stdout, stderr = _run_tv_cli(args, cli_path)
    if rc != 0:
        log.error("  ! pine_%s failed: %s", kind, (stderr or stdout)[:160])
        return False
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(stdout)
    log.info("  ✓ pine_%s saved to %s (%d bytes)", kind, out, len(stdout))
    return True


def _handle_ohlcv(req: dict, cli_path: str, log: logging.Logger) -> bool:
    symbol = req.get("symbol", "")
    interval = req.get("interval", "")
    out = _translate_path(req.get("out", ""))
    count = req.get("count", 500)
    if not out:
        log.error("  ! ohlcv request missing 'out'")
        return False
    if not _set_chart_state(symbol, interval, cli_path, log):
        return False
    args = ["ohlcv", "--count", str(count)]
    rc, stdout, stderr = _run_tv_cli(args, cli_path)
    if rc != 0:
        log.error("  ! ohlcv failed: %s", (stderr or stdout)[:160])
        return False
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(stdout)
    log.info("  ✓ ohlcv saved to %s (%d bytes)", out, len(stdout))
    return True


def _handle_state(req: dict, cli_path: str, log: logging.Logger) -> bool:
    out = _translate_path(req.get("out", ""))
    if not out:
        log.error("  ! state request missing 'out'")
        return False
    rc, stdout, stderr = _run_tv_cli(["state"], cli_path)
    if rc != 0:
        log.error("  ! state failed: %s", (stderr or stdout)[:160])
        return False
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(stdout)
    log.info("  ✓ state saved to %s", out)
    return True


HANDLERS = {
    "screenshot":  _handle_screenshot,
    "pine_labels": lambda r, c, l: _handle_pine_data(r, c, l, "labels"),
    "pine_lines":  lambda r, c, l: _handle_pine_data(r, c, l, "lines"),
    "pine_boxes":  lambda r, c, l: _handle_pine_data(r, c, l, "boxes"),
    "pine_tables": lambda r, c, l: _handle_pine_data(r, c, l, "tables"),
    "ohlcv":       _handle_ohlcv,
    "state":       _handle_state,
}


# -----------------------------------------------------------------------------
# Queue processor
# -----------------------------------------------------------------------------
def _process_request(req_path: Path, cli_path: str, screenshot_dir: str,
                      log: logging.Logger) -> None:
    try:
        with open(req_path, "r", encoding="utf-8") as f:
            req = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.error("  ! cannot parse %s: %s", req_path.name, exc)
        _mark_done(req_path, status="error", reason=str(exc))
        return

    req_type = req.get("type")
    log.info("→ %s: type=%s symbol=%s interval=%s",
              req_path.name, req_type, req.get("symbol"), req.get("interval"))

    handler = HANDLERS.get(req_type)
    if handler is None:
        log.error("  ! unknown request type %r", req_type)
        _mark_done(req_path, status="error", reason=f"unknown type {req_type}")
        return

    try:
        if req_type == "screenshot":
            ok = handler(req, cli_path, screenshot_dir, log)
        else:
            ok = handler(req, cli_path, log)
    except Exception as exc:
        log.exception("  ! handler crashed: %s", exc)
        _mark_done(req_path, status="error", reason=str(exc))
        return

    _mark_done(req_path, status="ok" if ok else "error")


def _mark_done(req_path: Path, status: str, reason: str = "") -> None:
    """Move .req → .done.{status} so reader knows it finished."""
    done_path = req_path.with_suffix(f".done.{status}")
    try:
        # Write the status meta into the done file
        if status == "error":
            with open(done_path, "w", encoding="utf-8") as f:
                json.dump({"status": status, "reason": reason,
                           "completed_at": datetime.now(timezone.utc).isoformat()},
                          f, ensure_ascii=False)
        else:
            done_path.touch()
        req_path.unlink()
    except OSError:
        pass


def run_loop(queue_dir: str, cli_path: str, screenshot_dir: str,
              poll_interval: float, log: logging.Logger) -> None:
    log.info("tv_bridge starting")
    log.info("  queue_dir:      %s", queue_dir)
    log.info("  tv_mcp_cli:     %s", cli_path)
    log.info("  screenshot_dir: %s", screenshot_dir)
    log.info("  poll_interval:  %.1fs", poll_interval)
    os.makedirs(queue_dir, exist_ok=True)

    # Sanity check
    if not os.path.isfile(cli_path):
        log.error("TV MCP CLI not found at %s — set --cli-path", cli_path)
        sys.exit(2)
    log.info("Polling for .req files…")

    while True:
        try:
            reqs = sorted(Path(queue_dir).glob("*.req"))
            for req_path in reqs:
                _process_request(req_path, cli_path, screenshot_dir, log)
        except Exception as exc:
            log.exception("Loop error: %s", exc)
        time.sleep(poll_interval)


def run_once(queue_dir: str, cli_path: str, screenshot_dir: str,
              log: logging.Logger) -> int:
    """Process the queue once and exit. Useful for testing or cron."""
    os.makedirs(queue_dir, exist_ok=True)
    reqs = sorted(Path(queue_dir).glob("*.req"))
    log.info("Processing %d pending request(s)…", len(reqs))
    for req_path in reqs:
        _process_request(req_path, cli_path, screenshot_dir, log)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue-dir", default=DEFAULT_QUEUE_DIR)
    p.add_argument("--cli-path", default=DEFAULT_TV_MCP_CLI,
                    help="path to tradingview-mcp's cli/index.js")
    p.add_argument("--screenshot-dir", default=DEFAULT_TV_SCREENSHOT_DIR,
                    help="where the TV MCP writes screenshots before we copy them")
    p.add_argument("--interval-sec", type=float, default=DEFAULT_POLL_INTERVAL)
    p.add_argument("--once", action="store_true",
                    help="process queue once and exit")
    args = p.parse_args()

    log = _setup_logging()
    if args.once:
        return run_once(args.queue_dir, args.cli_path, args.screenshot_dir, log)
    try:
        run_loop(args.queue_dir, args.cli_path, args.screenshot_dir,
                  args.interval_sec, log)
    except KeyboardInterrupt:
        log.info("Stopped by user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
