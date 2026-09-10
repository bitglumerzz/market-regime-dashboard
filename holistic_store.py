"""Persistent storage for the last holistic Claude opinion per ticker.

Streamlit's ``st.session_state`` is per-session and resets on hard reload.
This module persists the most recent holistic opinion for each ticker to a
JSON file on disk so:

  1. Returning to the Elliott Waves page after navigating away → previous
     opinion is auto-restored without a new Claude call ($0.05+ saved).
  2. Browser F5 / app restart → previous opinions survive.
  3. The retry-capture button can rebuild annotations from the persisted
     opinion even after a session restart.

Schema (one JSON file with all tickers, keyed by ticker):

    {
      "BTC-USD": {
        "saved_at": "2026-05-15T11:08:00+00:00",
        "ticker": "BTC-USD",
        "opinion": { ... full AIHolisticOpinion as dict ... },
        "pid": "abc123..."
      },
      "ETH-USD": { ... }
    }

Read/write are atomic via tmpfile + rename to avoid half-written state.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

STORE_DIR = "predictions"
STORE_PATH = os.path.join(STORE_DIR, "last_holistic_by_ticker.json")


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


def save_last_holistic(ticker: str, opinion: dict[str, Any],
                        pid: str | None = None) -> None:
    """Persist the most recent holistic opinion for this ticker.

    Overwrites any previous entry for the same ticker. Other tickers
    are left untouched.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return
    data = _load_all()
    data[ticker] = {
        "saved_at": _now_iso(),
        "ticker":   ticker,
        "opinion":  opinion,
        "pid":      pid or "",
    }
    _write_all(data)


def load_last_holistic(ticker: str) -> dict | None:
    """Return the most recent persisted holistic for this ticker, or None.

    The returned dict has keys: saved_at, ticker, opinion, pid.
    The ``opinion`` sub-dict mirrors the structure of ``last_holistic_full``
    in session_state (full AIHolisticOpinion fields + annotated_screenshots).
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return None
    data = _load_all()
    entry = data.get(ticker)
    if not isinstance(entry, dict):
        return None
    return entry


def age_minutes(entry: dict) -> float | None:
    """How many minutes ago this entry was saved."""
    saved_at = entry.get("saved_at")
    if not saved_at:
        return None
    try:
        from datetime import datetime, timezone
        t = datetime.fromisoformat(saved_at)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - t
        return delta.total_seconds() / 60.0
    except Exception:
        return None


def list_tickers() -> list[str]:
    """All tickers with persisted holistic opinions."""
    return sorted(_load_all().keys())
