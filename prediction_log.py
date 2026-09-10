"""Persistent log of Claude predictions with later outcome verification.

Each call to ask_claude / ask_claude_holistic can be persisted here. Later, a
verifier walks forward in price history and decides whether the predicted
target was reached, the stop was hit, or the prediction is still open.

Storage: JSONL file at PREDICTIONS_PATH. Append-only on log; full rewrite on
verification updates (small file — no performance concern at our scale).

Idempotency: a prediction has a deterministic `id` derived from
(ticker, timeframe, action, entry, prediction_minute). Re-logging the same
prediction within the same minute is a no-op.

Outcome detection (v1): uses close-only prices. A long is a win if any future
close >= target before any close <= stop_loss. We document this approximation
in the UI — intraday wicks can flip the order of events.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Iterable, Callable, Any

import pandas as pd

# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
PREDICTIONS_DIR = "predictions"
PREDICTIONS_PATH = os.path.join(PREDICTIONS_DIR, "claude_predictions.jsonl")


def _ensure_dir() -> None:
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_id(*parts: Any) -> str:
    blob = "|".join(str(p) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def log_per_tf(
    ticker: str,
    timeframe: str,
    opinion: Any,                       # AIWaveOpinion
    current_price: float,
    prediction_time: pd.Timestamp | datetime | None = None,
    forecast_pattern: str | None = None,
    forecast_direction: str | None = None,
) -> dict:
    """Persist a per-TF Claude opinion. Returns the stored dict.

    No-ops (returns the existing record) if a prediction with the same
    deterministic id was logged within the same minute.
    """
    pt = pd.Timestamp(prediction_time) if prediction_time is not None else pd.Timestamp(_now_iso())
    if pt.tzinfo is None:
        pt = pt.tz_localize("UTC")
    minute_key = pt.strftime("%Y-%m-%dT%H:%M")

    pid = _hash_id("pertf", ticker, timeframe,
                   getattr(opinion, "action", "wait"),
                   _safe_float(getattr(opinion, "entry", None)),
                   minute_key)

    existing = _find_by_id(pid)
    if existing is not None:
        return existing

    record = {
        "id": pid,
        "type": "per_tf",
        "logged_at": _now_iso(),
        "ticker": ticker,
        "timeframe": timeframe,
        "current_price": float(current_price),
        "action": getattr(opinion, "action", "wait"),
        "entry": _safe_float(getattr(opinion, "entry", None)),
        "stop_loss": _safe_float(getattr(opinion, "stop_loss", None)),
        "target": _safe_float(getattr(opinion, "target", None)),
        # Per-TF-only fields
        "agrees_with_top": bool(getattr(opinion, "agrees_with_top", False)),
        "confidence": getattr(opinion, "confidence", "low"),
        "preferred_pattern": getattr(opinion, "preferred_pattern", ""),
        "reasoning": getattr(opinion, "reasoning", ""),
        "agrees_with_forecast": getattr(opinion, "agrees_with_forecast", None),
        "forecast_critique": getattr(opinion, "forecast_critique", ""),
        # Forecast context (for later analysis: does Claude agree with our engine more often when right?)
        "forecast_pattern": forecast_pattern,
        "forecast_direction": forecast_direction,
        # Verification slots
        "outcome": None,                 # 'win' | 'loss' | 'open' | 'expired' | 'no_trade'
        "exit_price": None,
        "exit_time": None,
        "bars_to_resolve": None,
        "r_realized": None,
        "verified_at": None,
    }
    _append(record)
    return record


def log_holistic(
    ticker: str,
    opinion: Any,                       # AIHolisticOpinion
    current_price: float,
    prediction_time: pd.Timestamp | datetime | None = None,
) -> dict:
    """Persist a holistic (all-TF) Claude opinion. Returns the stored dict."""
    pt = pd.Timestamp(prediction_time) if prediction_time is not None else pd.Timestamp(_now_iso())
    if pt.tzinfo is None:
        pt = pt.tz_localize("UTC")
    minute_key = pt.strftime("%Y-%m-%dT%H:%M")

    pid = _hash_id("holistic", ticker,
                   getattr(opinion, "action", "wait"),
                   _safe_float(getattr(opinion, "entry", None)),
                   minute_key)

    existing = _find_by_id(pid)
    if existing is not None:
        return existing

    record = {
        "id": pid,
        "type": "holistic",
        "logged_at": _now_iso(),
        "ticker": ticker,
        "timeframe": "all",
        "current_price": float(current_price),
        "action": getattr(opinion, "action", "wait"),
        "entry": _safe_float(getattr(opinion, "entry", None)),
        "stop_loss": _safe_float(getattr(opinion, "stop_loss", None)),
        "target": _safe_float(getattr(opinion, "target", None)),
        # Holistic-only fields
        "bias": getattr(opinion, "bias", "neutral"),
        "setup_quality": getattr(opinion, "setup_quality", "low"),
        "summary": getattr(opinion, "summary", ""),
        "risk_reward": _safe_float(getattr(opinion, "risk_reward", None)),
        "invalidation_explained": getattr(opinion, "invalidation_explained", ""),
        "per_tf_notes": getattr(opinion, "per_tf_notes", {}) or {},
        # Verification slots
        "outcome": None,
        "exit_price": None,
        "exit_time": None,
        "bars_to_resolve": None,
        "r_realized": None,
        "verified_at": None,
    }
    _append(record)

    # Phase 7.1 + 16 — best-effort TV chart screenshot for visual verification
    # AND for PIL annotation overlay by tv_annotate.
    # Captures all 4 TFs so chart_annotations from Claude can target any of
    # them. Each TF is best-effort: failures don't block the rest.
    try:
        from tv_screenshot import capture
        for tf in ("1d", "4h", "15m", "5m"):
            try:
                capture(f"{pid}_{tf}", ticker, tf)
            except Exception:
                continue
    except Exception:
        pass

    return record


def load_all() -> list[dict]:
    """Read every prediction from disk. Empty list if file doesn't exist."""
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


def verify_predictions(
    ohlc_fetcher: Callable[[str, str, pd.Timestamp], pd.DataFrame],
    max_bars_lookforward: dict[str, int] | None = None,
    only_open: bool = True,
) -> dict:
    """Walk every open prediction forward and decide its outcome.

    Args:
        ohlc_fetcher: callable(ticker, timeframe, start_ts) -> DataFrame
            with columns ['high', 'low', 'close'] indexed by timestamp.
            Should return bars at-or-after start_ts.
        max_bars_lookforward: per-TF maximum bars to wait before marking 'expired'.
            Defaults to {1d: 60, 4h: 80, 15m: 96, 5m: 144, all: 60}.
        only_open: skip already-verified predictions.

    Returns: {checked, won, lost, still_open, expired, no_trade}
    """
    if max_bars_lookforward is None:
        max_bars_lookforward = {
            "1d": 60, "4h": 80, "15m": 96, "5m": 144, "all": 60,
        }

    preds = load_all()
    stats = {"checked": 0, "won": 0, "lost": 0,
             "still_open": 0, "expired": 0, "no_trade": 0}
    changed = False

    for p in preds:
        if only_open and p.get("outcome") is not None:
            continue
        stats["checked"] += 1

        action = (p.get("action") or "").lower()
        entry = p.get("entry")
        stop = p.get("stop_loss")
        target = p.get("target")
        ticker = p.get("ticker")
        tf = p.get("timeframe") or "1d"

        if action not in ("long", "short") or entry is None or stop is None or target is None:
            p["outcome"] = "no_trade"
            p["verified_at"] = _now_iso()
            stats["no_trade"] += 1
            changed = True
            continue

        # For holistic predictions ("all"), verify on the 1d series — that's
        # the timeframe the unified plan plays out on. Could be made configurable.
        verify_tf = "1d" if tf == "all" else tf

        try:
            df = ohlc_fetcher(ticker, verify_tf, pd.Timestamp(p["logged_at"]))
        except Exception:
            stats["still_open"] += 1
            continue

        if df is None or len(df) == 0:
            stats["still_open"] += 1
            continue

        max_bars = max_bars_lookforward.get(tf, 60)
        df = df.iloc[:max_bars]

        outcome = _resolve_outcome(df, action, entry, stop, target)
        if outcome["status"] == "open":
            stats["still_open"] += 1
            continue

        p["outcome"] = outcome["status"]
        p["exit_price"] = outcome["exit_price"]
        p["exit_time"] = outcome["exit_time"]
        p["bars_to_resolve"] = outcome["bars_to_resolve"]
        p["r_realized"] = outcome["r_realized"]
        p["verified_at"] = _now_iso()

        if outcome["status"] == "win":
            stats["won"] += 1
        elif outcome["status"] == "loss":
            stats["lost"] += 1
        elif outcome["status"] == "expired":
            stats["expired"] += 1
        changed = True

    if changed:
        _save_all(preds)
    return stats


def compute_stats(predictions: list[dict]) -> dict:
    """Roll-up statistics across (verified) predictions."""
    out: dict = {
        "total": len(predictions),
        "verified": 0,
        "won": 0,
        "lost": 0,
        "expired": 0,
        "open": 0,
        "no_trade": 0,
        "hit_rate": None,
        "avg_r_realized": None,
        "by_tf": {},
        "by_bias": {},
        "by_setup_quality": {},
        "by_confidence": {},
        "by_agrees_with_top": {},
    }
    r_values: list[float] = []
    by_tf: dict[str, dict] = {}
    by_bias: dict[str, dict] = {}
    by_setup: dict[str, dict] = {}
    by_conf: dict[str, dict] = {}
    by_agree: dict[str, dict] = {}

    for p in predictions:
        outcome = p.get("outcome")
        if outcome is None:
            out["open"] += 1
            continue
        if outcome == "no_trade":
            out["no_trade"] += 1
            continue
        out["verified"] += 1
        if outcome == "win":
            out["won"] += 1
        elif outcome == "loss":
            out["lost"] += 1
        elif outcome == "expired":
            out["expired"] += 1
        elif outcome == "open":
            out["open"] += 1
            continue

        r = p.get("r_realized")
        if isinstance(r, (int, float)):
            r_values.append(float(r))

        # Breakdowns
        _bump(by_tf, p.get("timeframe", "?"), outcome)
        if p.get("type") == "holistic":
            _bump(by_bias, p.get("bias", "?"), outcome)
            _bump(by_setup, p.get("setup_quality", "?"), outcome)
        if p.get("type") == "per_tf":
            _bump(by_conf, p.get("confidence", "?"), outcome)
            _bump(by_agree, str(p.get("agrees_with_top", "?")), outcome)

    decisive = out["won"] + out["lost"]
    if decisive > 0:
        out["hit_rate"] = out["won"] / decisive
    if r_values:
        out["avg_r_realized"] = sum(r_values) / len(r_values)

    out["by_tf"] = _finalize(by_tf)
    out["by_bias"] = _finalize(by_bias)
    out["by_setup_quality"] = _finalize(by_setup)
    out["by_confidence"] = _finalize(by_conf)
    out["by_agrees_with_top"] = _finalize(by_agree)
    return out


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------
def _append(record: dict) -> None:
    _ensure_dir()
    with open(PREDICTIONS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_all(records: list[dict]) -> None:
    _ensure_dir()
    tmp = PREDICTIONS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, PREDICTIONS_PATH)


def _find_by_id(pid: str) -> dict | None:
    for p in load_all():
        if p.get("id") == pid:
            return p
    return None


def _resolve_outcome(
    df: pd.DataFrame, action: str,
    entry: float, stop: float, target: float,
) -> dict:
    """Walk bars and decide win/loss/expired.

    A long wins when high >= target before low <= stop.
    A short wins when low <= target before high >= stop.

    On the same bar both could trigger — we conservatively call it a loss
    (worst-case assumption — drawdown happens first intra-bar).
    """
    if "high" not in df.columns or "low" not in df.columns:
        # Close-only fallback — approximate
        df = df.copy()
        df["high"] = df["close"]
        df["low"] = df["close"]

    is_long = (action == "long")
    risk = abs(entry - stop) if entry != stop else 1e-9

    for ts, row in df.iterrows():
        hi = float(row["high"])
        lo = float(row["low"])
        stop_hit = (lo <= stop) if is_long else (hi >= stop)
        target_hit = (hi >= target) if is_long else (lo <= target)

        if stop_hit and target_hit:
            # Worst-case: stop first
            exit_price = stop
            r = -1.0
            return {"status": "loss", "exit_price": exit_price,
                    "exit_time": str(ts), "bars_to_resolve": 0, "r_realized": r}
        if stop_hit:
            r = -1.0
            return {"status": "loss", "exit_price": float(stop),
                    "exit_time": str(ts),
                    "bars_to_resolve": int(df.index.get_loc(ts)) + 1,
                    "r_realized": r}
        if target_hit:
            reward = abs(target - entry)
            r = reward / risk
            return {"status": "win", "exit_price": float(target),
                    "exit_time": str(ts),
                    "bars_to_resolve": int(df.index.get_loc(ts)) + 1,
                    "r_realized": r}

    # Neither hit within window
    if len(df) == 0:
        return {"status": "open", "exit_price": None, "exit_time": None,
                "bars_to_resolve": None, "r_realized": None}
    last_close = float(df.iloc[-1]["close"])
    # MFE-based partial credit, optional — keep simple: just expired.
    return {"status": "expired", "exit_price": last_close,
            "exit_time": str(df.index[-1]),
            "bars_to_resolve": len(df),
            "r_realized": (last_close - entry) / risk if is_long
                          else (entry - last_close) / risk}


_OUTCOME_TO_BUCKET = {
    "win": "won", "loss": "lost",
    "expired": "expired", "no_trade": "no_trade",
}


def _bump(d: dict, key: str, outcome: str) -> None:
    bucket = d.setdefault(key, {"won": 0, "lost": 0, "expired": 0, "no_trade": 0})
    field = _OUTCOME_TO_BUCKET.get(outcome)
    if field is not None:
        bucket[field] += 1


def _finalize(d: dict) -> dict:
    """Add hit_rate to each breakdown bucket."""
    out = {}
    for k, v in d.items():
        decisive = v.get("won", 0) + v.get("lost", 0)
        v = dict(v)
        v["hit_rate"] = (v["won"] / decisive) if decisive else None
        out[k] = v
    return out
