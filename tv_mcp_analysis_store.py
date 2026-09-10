"""Storage and retrieval for TV MCP "second opinion" analyses.

When the user gets a high-quality Elliott Wave analysis from the TV MCP
agent (which has access to LuxAlgo Pine labels, full chart context, etc.),
we want to:
  1. Persist it for later comparison with our own holistic Claude answer
  2. Display it side-by-side with holistic on the Elliott Waves page
  3. Use it as a reference benchmark for evaluation

Two modes:
  * **Manual paste** — user copies analysis text from MCP into a textarea,
    we save it as the latest analysis for the ticker.
  * **Auto-request** (future) — submit `{type:"mcp_analysis"}` to
    tv_bridge_queue; the host bridge daemon dispatches to MCP and the
    result lands back as a JSON file. Code stub present, requires the
    MCP server.js to support `analyze` command.

Storage: ``tv_mcp_analysis/by_ticker.json`` (bind-mounted via predictions
volume — same idea as last_holistic_by_ticker.json).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone

# We piggy-back on the existing predictions/ bind mount to avoid adding
# another volume — keeps the docker-compose volume list short.
STORE_DIR = "predictions"
STORE_PATH = os.path.join(STORE_DIR, "tv_mcp_analyses.json")

# For auto-request path — uses the same queue as other tv_bridge requests
QUEUE_DIR = "tv_bridge_queue"


def _ensure_dir() -> None:
    os.makedirs(STORE_DIR, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_all() -> dict:
    if not os.path.isfile(STORE_PATH):
        return {}
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_all(data: dict) -> None:
    _ensure_dir()
    tmp = STORE_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, STORE_PATH)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Manual paste — user copies MCP output and saves it for the ticker
# --------------------------------------------------------------------------
def save_manual_analysis(ticker: str, analysis_text: str,
                          author: str = "tv_mcp_manual") -> dict:
    """Persist a manually-pasted MCP analysis for this ticker."""
    ticker = ticker.strip().upper()
    if not ticker or not analysis_text:
        return {}
    data = _load_all()
    entry = {
        "saved_at":      _now_iso(),
        "ticker":        ticker,
        "author":        author,
        "analysis_text": analysis_text,
        "source":        "manual_paste",
    }
    data[ticker] = entry
    _write_all(data)
    return entry


def load_last_analysis(ticker: str) -> dict | None:
    """Return the most recent MCP analysis for this ticker, or None."""
    ticker = ticker.strip().upper()
    if not ticker:
        return None
    data = _load_all()
    entry = data.get(ticker)
    return entry if isinstance(entry, dict) else None


def list_tickers() -> list[str]:
    """All tickers that have a stored MCP analysis."""
    return sorted(_load_all().keys())


def age_minutes(entry: dict) -> float | None:
    """Minutes since this entry was saved."""
    saved_at = entry.get("saved_at")
    if not saved_at:
        return None
    try:
        t = datetime.fromisoformat(saved_at)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() / 60.0
    except Exception:
        return None


# --------------------------------------------------------------------------
# Auto-request via tv_bridge (stub — requires MCP `analyze` command)
# --------------------------------------------------------------------------
def is_bridge_available() -> bool:
    return os.path.isdir(QUEUE_DIR)


def submit_analysis_request(ticker: str, tf: str = "1D") -> str:
    """Drop a `{type:"mcp_analysis"}` request into tv_bridge_queue/.

    Requires the host bridge daemon (tv_bridge.py) to support the
    `mcp_analysis` request type, which in turn requires
    `tradingview-mcp/src/server.js` to expose an `analyze` command.

    Without that, the request will sit unprocessed and time out. The
    manual-paste flow above is the working alternative until MCP supports
    automatic analyses.
    """
    os.makedirs(QUEUE_DIR, exist_ok=True)
    rid = uuid.uuid4().hex[:12]
    req = {
        "type":     "mcp_analysis",
        "symbol":   ticker,
        "interval": tf,
        "out":      os.path.abspath(
            os.path.join("predictions", f"mcp_analysis_{rid}.json")),
    }
    path = os.path.join(QUEUE_DIR, f"{rid}.req")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(req, f, ensure_ascii=False)
    os.replace(tmp, path)
    return rid


def get_analysis_result(rid: str, timeout: float = 60.0,
                         poll: float = 1.0) -> dict | None:
    """Poll for the result of a submitted analysis request. Returns None
    on timeout. If MCP doesn't support `analyze` — will always return None."""
    deadline = time.time() + timeout
    out_path = os.path.join("predictions", f"mcp_analysis_{rid}.json")
    while time.time() < deadline:
        if os.path.isfile(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        # Also check for .done.error marker from the bridge
        err_path = os.path.join(QUEUE_DIR, f"{rid}.done.error")
        if os.path.isfile(err_path):
            return None
        time.sleep(poll)
    return None
