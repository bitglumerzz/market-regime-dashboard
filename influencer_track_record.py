"""Influencer prediction track-record.

For every directional take from Twitter or TradingView, we:
  1. Log it immediately to `influencer_track_record/predictions.jsonl`
     with the price snapshot at that moment
  2. Later (when sufficient bars elapsed), verify whether price moved in
     the predicted direction
  3. Compute per-author hit-rate, avg R, average lead time

This is symmetrical to `prediction_log` for Claude predictions, but for
the human influencers we track.

Sources of directional takes:
  - Twitter post: VADER+crypto sentiment ≥0.3 (bullish) or ≤-0.3 (bearish)
  - TradingView idea: explicit direction field (long / short)

Verification window:
  - Default 7 days for Twitter takes (no explicit timeframe)
  - Per-timeframe for TV ideas: 1D idea = 14 day window, 4H = 7d, 1H = 2d
  - WIN if price closes in the predicted direction by ≥3% in the window
  - LOSS if price moves ≥3% in the OPPOSITE direction first
  - EXPIRED if neither within window
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Callable, Any

import pandas as pd

TRACK_RECORD_DIR = "influencer_track_record"
PREDICTIONS_PATH = os.path.join(TRACK_RECORD_DIR, "predictions.jsonl")

# Win/loss thresholds (as fraction of starting price)
DEFAULT_TARGET_PCT = 0.03    # ≥3% in predicted direction = win
DEFAULT_STOP_PCT = 0.03      # ≥3% in opposite direction first = loss

# Per-timeframe verification window in days
TF_WINDOW_DAYS = {
    "1D": 14, "1d": 14,
    "4H": 7,  "4h": 7,
    "1H": 2,  "1h": 2,
    "15m": 1,
    "5m":  0.5,
    "":    7,    # default for tweets without TF
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hash_id(*parts: Any) -> str:
    blob = "|".join(str(p) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _ensure_dir() -> None:
    os.makedirs(TRACK_RECORD_DIR, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Logging API
# ---------------------------------------------------------------------------
def log_twitter_take(
    handle: str,
    ticker: str,
    sentiment: float,           # VADER+crypto score
    text: str,
    posted_at: str,
    url: str,
    tier: str,
    current_price: float,
) -> dict | None:
    """Log a directional take from Twitter. Returns the record or None if
    sentiment is too weak to count as directional."""
    if abs(sentiment) < 0.3:
        return None
    direction = "long" if sentiment > 0 else "short"
    minute_key = (posted_at or _now_iso())[:16]
    pid = _hash_id("twitter", handle, ticker, direction, minute_key)
    existing = _find_by_id(pid)
    if existing is not None:
        return existing

    record = {
        "id": pid,
        "source": "twitter",
        "logged_at":   _now_iso(),
        "posted_at":   posted_at,
        "handle":      handle,
        "tier":        tier,
        "ticker":      ticker,
        "direction":   direction,
        "strength":    round(abs(sentiment), 3),
        "current_price": float(current_price),
        "timeframe":   "",    # tweets don't have explicit TF
        "title":       text[:280],
        "url":         url,
        # Verification slots (filled later)
        "outcome":     None,
        "exit_price":  None,
        "exit_time":   None,
        "max_favorable_excursion": None,
        "max_adverse_excursion":   None,
        "verified_at": None,
    }
    _append(record)
    return record


def log_tv_idea(
    handle: str,
    ticker: str,
    direction: str,             # 'long' | 'short' | 'education'
    title: str,
    timeframe: str,
    posted_at: str,
    url: str,
    tier: str,
    followers: int,
    current_price: float,
) -> dict | None:
    """Log a directional take from TradingView. Returns the record."""
    direction = (direction or "").lower()
    if direction not in ("long", "short"):
        return None       # education-only ideas don't count as predictions

    minute_key = (posted_at or _now_iso())[:16]
    pid = _hash_id("tv", handle, ticker, direction, minute_key)
    existing = _find_by_id(pid)
    if existing is not None:
        return existing

    record = {
        "id":          pid,
        "source":      "tradingview",
        "logged_at":   _now_iso(),
        "posted_at":   posted_at,
        "handle":      handle,
        "tier":        tier,
        "followers":   int(followers),
        "ticker":      ticker,
        "direction":   direction,
        "strength":    1.0,        # TV: explicit direction, full strength
        "current_price": float(current_price),
        "timeframe":   timeframe,
        "title":       title[:280],
        "url":         url,
        "outcome":     None,
        "exit_price":  None,
        "exit_time":   None,
        "max_favorable_excursion": None,
        "max_adverse_excursion":   None,
        "verified_at": None,
    }
    _append(record)
    return record


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_predictions(
    ohlc_fetcher: Callable[[str, str, pd.Timestamp], pd.DataFrame],
    only_open: bool = True,
    target_pct: float = DEFAULT_TARGET_PCT,
    stop_pct: float = DEFAULT_STOP_PCT,
) -> dict:
    """Walk every open prediction forward; mark win/loss/expired."""
    preds = load_all()
    stats = {"checked": 0, "won": 0, "lost": 0,
             "still_open": 0, "expired": 0}
    changed = False

    for p in preds:
        if only_open and p.get("outcome") is not None:
            continue
        stats["checked"] += 1

        direction = p.get("direction")
        entry = p.get("current_price")
        if direction not in ("long", "short") or not entry:
            continue

        tf = p.get("timeframe") or ""
        window_days = TF_WINDOW_DAYS.get(tf, 7)

        try:
            df = ohlc_fetcher(p["ticker"], "1d",
                              pd.Timestamp(p["posted_at"] or p["logged_at"]))
        except Exception:
            stats["still_open"] += 1
            continue
        if df is None or df.empty:
            stats["still_open"] += 1
            continue
        df = df.iloc[:int(window_days)]

        outcome = _resolve_outcome(df, direction, float(entry),
                                     target_pct, stop_pct)
        if outcome["status"] == "open":
            stats["still_open"] += 1
            continue
        p["outcome"]    = outcome["status"]
        p["exit_price"] = outcome["exit_price"]
        p["exit_time"]  = outcome["exit_time"]
        p["max_favorable_excursion"] = outcome["mfe"]
        p["max_adverse_excursion"]   = outcome["mae"]
        p["verified_at"] = _now_iso()

        if outcome["status"] == "win":  stats["won"] += 1
        elif outcome["status"] == "loss": stats["lost"] += 1
        elif outcome["status"] == "expired": stats["expired"] += 1
        changed = True

    if changed:
        _save_all(preds)
    return stats


def _resolve_outcome(df: pd.DataFrame, direction: str,
                      entry: float, target_pct: float, stop_pct: float) -> dict:
    """For each bar after entry, see if MOVE in predicted direction ≥target
    or in opposite direction ≥stop. Whichever first."""
    if "high" not in df.columns or "low" not in df.columns:
        df = df.copy()
        df["high"] = df["close"]
        df["low"]  = df["close"]
    target_price = (entry * (1 + target_pct) if direction == "long"
                     else entry * (1 - target_pct))
    stop_price   = (entry * (1 - stop_pct) if direction == "long"
                     else entry * (1 + stop_pct))
    mfe = 0.0
    mae = 0.0
    is_long = direction == "long"
    for ts, row in df.iterrows():
        hi = float(row["high"])
        lo = float(row["low"])
        excursion_up   = (hi - entry) / entry
        excursion_dn   = (entry - lo) / entry
        if is_long:
            mfe = max(mfe, excursion_up)
            mae = max(mae, excursion_dn)
        else:
            mfe = max(mfe, excursion_dn)
            mae = max(mae, excursion_up)
        target_hit = (hi >= target_price) if is_long else (lo <= target_price)
        stop_hit   = (lo <= stop_price)   if is_long else (hi >= stop_price)
        if stop_hit and target_hit:
            return {"status": "loss", "exit_price": stop_price,
                    "exit_time": str(ts), "mfe": round(mfe, 4), "mae": round(mae, 4)}
        if stop_hit:
            return {"status": "loss", "exit_price": float(stop_price),
                    "exit_time": str(ts), "mfe": round(mfe, 4), "mae": round(mae, 4)}
        if target_hit:
            return {"status": "win", "exit_price": float(target_price),
                    "exit_time": str(ts), "mfe": round(mfe, 4), "mae": round(mae, 4)}
    if len(df) == 0:
        return {"status": "open", "exit_price": None, "exit_time": None,
                "mfe": 0.0, "mae": 0.0}
    return {"status": "expired", "exit_price": float(df.iloc[-1]["close"]),
            "exit_time": str(df.index[-1]),
            "mfe": round(mfe, 4), "mae": round(mae, 4)}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def author_stats() -> list[dict]:
    """Per-author leaderboard: hit_rate, n_predictions, avg_mfe."""
    preds = load_all()
    by_h: dict[str, dict] = {}
    for p in preds:
        h = p.get("handle", "?")
        d = by_h.setdefault(h, {
            "handle": h, "tier": p.get("tier", "?"), "source": p.get("source", "?"),
            "n_total": 0, "n_verified": 0, "won": 0, "lost": 0, "expired": 0,
            "sum_mfe": 0.0, "sum_mae": 0.0,
        })
        d["n_total"] += 1
        if p.get("outcome") in ("win", "loss", "expired"):
            d["n_verified"] += 1
            if p["outcome"] == "win":  d["won"] += 1
            elif p["outcome"] == "lost": d["lost"] += 1
            elif p["outcome"] == "expired": d["expired"] += 1
            d["sum_mfe"] += float(p.get("max_favorable_excursion") or 0)
            d["sum_mae"] += float(p.get("max_adverse_excursion") or 0)
    out = []
    for h, d in by_h.items():
        decisive = d["won"] + d["lost"]
        d["hit_rate"] = (d["won"] / decisive) if decisive else None
        d["avg_mfe"]  = (d["sum_mfe"] / d["n_verified"]) if d["n_verified"] else 0.0
        d["avg_mae"]  = (d["sum_mae"] / d["n_verified"]) if d["n_verified"] else 0.0
        out.append(d)
    return sorted(out, key=lambda x: (
        -1 if x["hit_rate"] is None else -x["hit_rate"],
        -x["n_verified"],
    ))


# ---------------------------------------------------------------------------
# Storage internals
# ---------------------------------------------------------------------------
def _append(rec: dict) -> None:
    _ensure_dir()
    with open(PREDICTIONS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _save_all(records: list[dict]) -> None:
    _ensure_dir()
    tmp = PREDICTIONS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, PREDICTIONS_PATH)


def load_all() -> list[dict]:
    if not os.path.isfile(PREDICTIONS_PATH):
        return []
    out: list[dict] = []
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _find_by_id(pid: str) -> dict | None:
    for p in load_all():
        if p.get("id") == pid:
            return p
    return None
