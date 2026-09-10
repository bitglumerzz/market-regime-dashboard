"""Track-record storage and verification for holistic Claude opinions.

Phase 43 — keeps a per-ticker rolling history of holistic predictions
(last N versions) and verifies each one against subsequent OHLC data:
target hit, stop hit, invalidation hit, or still pending.

Storage: ``predictions/holistic_history_by_ticker.json`` (atomic writes).
Shape::

    {
      "HYPE-USD": [
        {
          "version": 4,
          "saved_at": "2026-05-18T03:15:00Z",
          "bias": "long", "action": "wait",
          "entry": 44.20, "stop_loss": 42.50, "target": 49.70,
          "summary": "Wave-4 bottom at $44.20...",
          "critical_level": "$42.50 — invalidation",
          "golden_entry_zone": "GOLDEN ENTRY: $44.20..."
        },
        ...
      ]
    }

Verification happens *on demand* — when the next holistic is generated,
we ask the OHLC fetcher for prices since each saved entry and decide
hit/stop/invalid/pending.

Used by:
  * pages/4_Elliott_Waves.py — appends new entries, loads + verifies
    history, passes verified track record to ask_claude_holistic.
  * wave_ai.build_holistic_prompt — formats history as a TRACK RECORD
    prompt section.
  * holistic_report.py — renders the track-record table in MCP-style.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable

_HISTORY_PATH = "predictions/holistic_history_by_ticker.json"
MAX_HISTORY_PER_TICKER = 10


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def _read_history() -> dict:
    if not os.path.isfile(_HISTORY_PATH):
        return {}
    try:
        with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_history(data: dict) -> None:
    _ensure_dir(_HISTORY_PATH)
    fd, tmp = tempfile.mkstemp(
        prefix="holistic_history_", suffix=".json",
        dir=os.path.dirname(_HISTORY_PATH) or ".",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _HISTORY_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_holistic_entry(ticker: str, opinion: dict) -> int:
    """Append a new holistic version to ticker's history. Returns version#."""
    if not isinstance(opinion, dict) or not ticker:
        return 0
    data = _read_history()
    arr = data.setdefault(ticker, [])
    next_version = (arr[-1]["version"] + 1) if arr else 1
    entry = {
        "version":   next_version,
        "saved_at":  datetime.now(timezone.utc).isoformat(),
        "bias":      opinion.get("bias", ""),
        "action":    opinion.get("action", ""),
        "setup_quality": opinion.get("setup_quality", ""),
        "entry":     opinion.get("entry"),
        "stop_loss": opinion.get("stop_loss"),
        "target":    opinion.get("target"),
        "risk_reward": opinion.get("risk_reward"),
        "summary":   (opinion.get("summary") or "")[:400],
        "critical_level":     (opinion.get("critical_level") or "")[:200],
        "golden_entry_zone":  (opinion.get("golden_entry_zone") or "")[:200],
        "verdict":   "pending",  # filled later by verify_history
        "verdict_note": "",
        "verified_at": None,
        "extreme_price": None,
    }
    arr.append(entry)
    # Cap at MAX_HISTORY_PER_TICKER, drop oldest
    if len(arr) > MAX_HISTORY_PER_TICKER:
        data[ticker] = arr[-MAX_HISTORY_PER_TICKER:]
    _write_history(data)
    return next_version


def load_history(ticker: str, max_n: int | None = None) -> list[dict]:
    """Return last N versions of holistic for ticker (newest last)."""
    if not ticker:
        return []
    data = _read_history()
    arr = data.get(ticker, []) or []
    if max_n is not None and max_n > 0:
        arr = arr[-max_n:]
    return list(arr)


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def verify_entry(entry: dict, ohlc_fetcher: Callable[[datetime], Any]) -> dict:
    """Verify a single entry against OHLC since its saved_at.

    ``ohlc_fetcher(since_dt)`` should return a pandas-like object with
    'high' and 'low' columns indexed by datetime, OR None if no data.

    Returns the entry dict with mutated 'verdict', 'verdict_note',
    'verified_at', 'extreme_price' fields.
    """
    if entry.get("verdict") and entry["verdict"] != "pending":
        return entry  # already verified

    bias = (entry.get("bias") or "").lower()
    target = entry.get("target")
    stop = entry.get("stop_loss")
    entry_price = entry.get("entry")

    saved_at = _parse_iso(entry.get("saved_at", ""))
    if saved_at is None:
        return entry

    try:
        ohlc = ohlc_fetcher(saved_at)
    except Exception as e:
        entry["verdict_note"] = f"OHLC fetch error: {e}"
        return entry

    if ohlc is None or len(ohlc) == 0:
        return entry  # still pending — no data yet

    try:
        highs = ohlc["high"] if "high" in ohlc else ohlc.get("High")
        lows = ohlc["low"] if "low" in ohlc else ohlc.get("Low")
        period_high = float(highs.max())
        period_low = float(lows.min())
    except (KeyError, AttributeError, ValueError):
        entry["verdict_note"] = "Cannot parse OHLC"
        return entry

    # Decide verdict based on direction
    is_long = bias == "long"
    is_short = bias == "short"

    target_hit = False
    stop_hit = False

    if is_long and isinstance(target, (int, float)):
        target_hit = period_high >= target
    elif is_short and isinstance(target, (int, float)):
        target_hit = period_low <= target

    if is_long and isinstance(stop, (int, float)):
        stop_hit = period_low <= stop
    elif is_short and isinstance(stop, (int, float)):
        stop_hit = period_high >= stop

    extreme = period_high if is_long else period_low
    entry["extreme_price"] = round(extreme, 4)
    entry["verified_at"] = datetime.now(timezone.utc).isoformat()

    if target_hit and not stop_hit:
        entry["verdict"] = "hit"
        entry["verdict_note"] = (
            f"Target {target} hit ({'max' if is_long else 'min'} = {extreme:.4f})"
        )
    elif stop_hit and not target_hit:
        entry["verdict"] = "stopped"
        entry["verdict_note"] = (
            f"Stop {stop} hit ({'min' if is_long else 'max'} = {extreme:.4f})"
        )
    elif target_hit and stop_hit:
        # Both hit — without bar-level path info we can't tell which first.
        # Mark as ambiguous; user can look at OHLC manually.
        entry["verdict"] = "ambiguous"
        entry["verdict_note"] = (
            f"Both target {target} and stop {stop} touched"
        )
    else:
        entry["verdict"] = "pending"
        entry["verdict_note"] = (
            f"Neither target {target} nor stop {stop} hit yet "
            f"(current extreme = {extreme:.4f})"
        )

    return entry


def verify_history(
    ticker: str,
    ohlc_fetcher: Callable[[datetime], Any],
    max_n: int | None = 5,
) -> list[dict]:
    """Verify the last N holistic entries for ticker against OHLC.

    Mutates the stored history with new verdicts, returns the list.
    Skips entries already verified (verdict != 'pending').
    """
    data = _read_history()
    arr = data.get(ticker) or []
    if not arr:
        return []
    # Verify the last N (newest)
    target_slice = arr[-max_n:] if max_n else arr
    for entry in target_slice:
        if entry.get("verdict") == "pending":
            verify_entry(entry, ohlc_fetcher)
    _write_history(data)
    return target_slice


def format_for_prompt(history: list[dict], max_n: int = 5) -> str:
    """Format track-record as a prompt section for Claude.

    Returns multi-line markdown-ish string to inject into the prompt.
    """
    if not history:
        return ""
    sub = history[-max_n:]
    lines = [
        "\n\n## TRACK RECORD (предыдущие версии holistic vs реальность)",
        "Используй для калибровки confidence: если предыдущие predictions",
        "сбывались — повышай confidence; если регулярно ✗ — понижай.",
        "",
    ]
    for e in sub:
        ver = e.get("version", "?")
        bias = e.get("bias", "?")
        action = e.get("action", "?")
        entry = e.get("entry")
        target = e.get("target")
        stop = e.get("stop_loss")
        verdict = e.get("verdict", "pending")
        note = e.get("verdict_note", "")
        emoji = {"hit": "✓", "stopped": "✗", "ambiguous": "⚠",
                  "pending": "⏳"}.get(verdict, "?")
        lines.append(
            f"- **v{ver}** ({bias}/{action}): entry={entry}, "
            f"stop={stop}, target={target} → {emoji} {verdict}"
        )
        if note:
            lines.append(f"  • {note}")
        if e.get("summary"):
            lines.append(f"  • summary: {(e['summary'] or '')[:200]}")
    return "\n".join(lines)
