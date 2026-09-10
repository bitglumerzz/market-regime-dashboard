"""Persistent journal of wave analyses + manual trade actions + detector transitions.

Three streams, each in its own JSONL file under ``wave_journal/``:

  1. ``snapshots.jsonl`` — full snapshot of one ``analyze_ticker`` run.
     Per-TF: top pattern, score, current_position, swing endpoints,
     forecast (direction/targets/invalidation), threshold metadata.
     Plus the top-down synthesis verdict.

  2. ``actions.jsonl`` — manual user actions linked to a snapshot:
     "took the setup", "skipped", "closed at stop", "closed at target".
     Each action carries direction/entry/stop/target/notes so we can
     verify outcomes the same way ``prediction_log`` does for Claude.

  3. ``transitions.jsonl`` — detector classification changes per (ticker,TF).
     Whenever a TF's top pattern changes between snapshots we record the
     ``before -> after`` pair. Useful to see how often the detector
     "flips" — a thrashing detector is a bad-quality detector.

Storage choices mirror ``prediction_log``:
  * append-only writes, full rewrite for verification updates
  * deterministic ``id`` so repeated logs in the same minute no-op
  * bind-mounted directory (see docker-compose.yml) so the journal
    survives ``docker compose down -v``.

This module deliberately depends only on the dataclasses defined by
``elliott``, ``forecast``, ``zigzag`` and on ``multi_tf.TFAnalysis``. It
does not import Streamlit, so it can be exercised from smoke tests and
sidecars.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import pandas as pd


# --------------------------------------------------------------------------
# Storage paths (bind mount: ./wave_journal:/app/wave_journal)
# --------------------------------------------------------------------------
WAVE_JOURNAL_DIR = "wave_journal"
SNAPSHOTS_PATH = os.path.join(WAVE_JOURNAL_DIR, "snapshots.jsonl")
ACTIONS_PATH = os.path.join(WAVE_JOURNAL_DIR, "actions.jsonl")
TRANSITIONS_PATH = os.path.join(WAVE_JOURNAL_DIR, "transitions.jsonl")


ACTION_TYPES = ("take_setup", "skip", "stop_hit", "take_profit",
                "partial_close", "note")


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _ensure_dir() -> None:
    os.makedirs(WAVE_JOURNAL_DIR, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_id(*parts: Any) -> str:
    blob = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
        return x if x == x else None  # filter NaN
    except (TypeError, ValueError):
        return None


def _ts_str(ts: Any) -> str | None:
    if ts is None:
        return None
    try:
        t = pd.Timestamp(ts)
    except Exception:
        return str(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.isoformat()


def _append(path: str, record: dict) -> None:
    _ensure_dir()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _save_all(path: str, records: list[dict]) -> None:
    _ensure_dir()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Serialization of multi_tf results
# --------------------------------------------------------------------------
def _serialize_swing(s: Any) -> dict:
    return {
        "index": _ts_str(getattr(s, "index", None)),
        "price": _safe_float(getattr(s, "price", None)),
        "kind":  getattr(s, "kind", None),
        "bar_index": int(getattr(s, "bar_index", 0)),
        "provisional": bool(getattr(s, "provisional", False)),
    }


def _serialize_forecast(fc: Any) -> dict | None:
    if fc is None:
        return None
    targets = []
    for t in getattr(fc, "targets", []) or []:
        targets.append({
            "label":       getattr(t, "label", ""),
            "price":       _safe_float(getattr(t, "price", None)),
            "fib_label":   getattr(t, "fib_label", ""),
            "probability": getattr(t, "probability", "medium"),
            "empirical_p": _safe_float(getattr(t, "empirical_p", None)),
        })
    return {
        "next_pattern":         getattr(fc, "next_pattern", "unknown"),
        "direction":            getattr(fc, "direction", "unclear"),
        "confidence":           getattr(fc, "confidence", "low"),
        "rationale":            getattr(fc, "rationale", ""),
        "targets":              targets,
        "invalidation_level":   _safe_float(getattr(fc, "invalidation_level", None)),
        "invalidation_reason":  getattr(fc, "invalidation_reason", ""),
        "expected_duration_bars": list(getattr(fc, "expected_duration_bars", None) or ()),
        "is_actionable":        bool(getattr(fc, "is_actionable", False)),
    }


def _serialize_candidate(c: Any) -> dict:
    """Compact serialization of a WaveCandidate — keep only what matters for evaluation."""
    return {
        "pattern":          getattr(c, "pattern", "undefined"),
        "score":            _safe_float(getattr(c, "score", 0.0)),
        "current_position": getattr(c, "current_position", ""),
        "labels":           list(getattr(c, "labels", []) or []),
        "is_extended":      bool(getattr(c, "is_extended", False)),
        "extension_ratio":  _safe_float(getattr(c, "extension_ratio", None)),
        "is_truncated":     bool(getattr(c, "is_truncated", False)),
        "is_diagonal":      bool(getattr(c, "is_diagonal", False)),
        "diagonal_kind":    getattr(c, "diagonal_kind", ""),
        "triangle_kind":    getattr(c, "triangle_kind", ""),
        "rule_violations":  list(getattr(c, "rule_violations", []) or []),
        "swings_used":      [_serialize_swing(s)
                              for s in getattr(c, "swings_used", []) or []],
        "next_targets":     {k: _safe_float(v)
                              for k, v in (getattr(c, "next_targets", {}) or {}).items()},
    }


def _serialize_tf(tf: Any) -> dict:
    """One ``TFAnalysis`` → a compact dict (no full price series)."""
    top = getattr(tf, "top", None)
    candidates = getattr(tf, "candidates", []) or []
    prices = getattr(tf, "prices", None)
    last_price = None
    last_time = None
    if prices is not None and len(prices):
        try:
            last_price = float(prices.iloc[-1])
            last_time = _ts_str(prices.index[-1])
        except Exception:
            pass
    return {
        "name":                getattr(tf, "name", "?"),
        "interval":            getattr(tf, "interval", "?"),
        "ok":                  bool(getattr(tf, "ok", True)),
        "error":               getattr(tf, "error", ""),
        "source":              getattr(tf, "source", ""),
        "threshold_source":    getattr(tf, "threshold_source", ""),
        "major_threshold_pct": _safe_float(getattr(tf, "major_threshold_pct", None)),
        "minor_threshold_pct": _safe_float(getattr(tf, "minor_threshold_pct", None)),
        "last_price":          last_price,
        "last_time":           last_time,
        "n_major_swings":      len(getattr(tf, "major_swings", []) or []),
        "n_minor_swings":      len(getattr(tf, "minor_swings", []) or []),
        "top":                 _serialize_candidate(top) if top is not None else None,
        "candidates":          [_serialize_candidate(c) for c in candidates[:3]],
        "forecast":            _serialize_forecast(getattr(tf, "forecast", None)),
    }


# --------------------------------------------------------------------------
# Public API — snapshots
# --------------------------------------------------------------------------
def _snapshot_fingerprint(per_tf: list[dict]) -> str:
    """Short hash of (TF name, pattern, current_position) tuples — used both
    for ID idempotency within the same minute AND to detect transitions
    between two snapshots of the same ticker."""
    parts: list[str] = []
    for tf in per_tf:
        top = tf.get("top") or {}
        parts.append("|".join([
            tf.get("name", ""),
            top.get("pattern", ""),
            top.get("current_position", ""),
            str(top.get("is_extended", False)),
            str(top.get("is_truncated", False)),
        ]))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def log_snapshot(
    ticker: str,
    tf_results: Iterable[Any],
    synthesis: dict | None = None,
    *,
    snapshot_time: pd.Timestamp | datetime | None = None,
    source: str = "elliott_page",
) -> dict:
    """Persist one multi-TF analysis. Returns the stored snapshot dict.

    Idempotent within the same minute when the per-TF fingerprint is
    unchanged: ``log_snapshot`` called twice for the same ticker in a
    one-minute window will return the first record without writing again.
    """
    per_tf = [_serialize_tf(r) for r in tf_results]
    fingerprint = _snapshot_fingerprint(per_tf)

    st = pd.Timestamp(snapshot_time) if snapshot_time is not None else pd.Timestamp(_now_iso())
    if st.tzinfo is None:
        st = st.tz_localize("UTC")
    minute_key = st.strftime("%Y-%m-%dT%H:%M")

    sid = _hash_id("snap", ticker.upper(), fingerprint, minute_key)

    existing = find_snapshot(sid)
    if existing is not None:
        return existing

    record: dict[str, Any] = {
        "id":             sid,
        "type":           "snapshot",
        "logged_at":      _now_iso(),
        "snapshot_time":  _ts_str(st),
        "ticker":         ticker.upper(),
        "source":         source,
        "fingerprint":    fingerprint,
        "per_tf":         per_tf,
        "synthesis":      synthesis or {},
        # Verification slots (filled by ``verify_snapshots``):
        #   per_tf_outcomes: {tf_name: {target_hit, invalidated, expired, ...}}
        "per_tf_outcomes": {},
        "verified_at":    None,
    }
    _append(SNAPSHOTS_PATH, record)
    return record


def load_snapshots() -> list[dict]:
    """All snapshots, oldest first."""
    return _load(SNAPSHOTS_PATH)


def find_snapshot(sid: str) -> dict | None:
    for s in load_snapshots():
        if s.get("id") == sid:
            return s
    return None


def latest_snapshot_for(ticker: str,
                        before: dict | None = None) -> dict | None:
    """Latest snapshot for ``ticker``.

    If ``before`` is given, returns the most recent snapshot that is NOT
    ``before`` itself (matched by ``id``). We can't simply filter by
    ``logged_at`` < before.logged_at, because two ``log_snapshot`` calls in
    the same second produce identical timestamps — we'd lose the prior.
    """
    ticker = ticker.upper()
    snaps = [s for s in load_snapshots() if s.get("ticker") == ticker]
    if before is not None:
        snaps = [s for s in snaps if s.get("id") != before.get("id")]
    if not snaps:
        return None
    return max(snaps, key=lambda s: s.get("logged_at", ""))


# --------------------------------------------------------------------------
# Public API — transitions
# --------------------------------------------------------------------------
def detect_and_log_transitions(new_snapshot: dict) -> list[dict]:
    """Compare ``new_snapshot`` against the previous snapshot for the same
    ticker. For every TF whose top pattern changed, append a transition.

    Returns the list of newly written transition records (possibly empty).
    """
    prev = latest_snapshot_for(new_snapshot["ticker"], before=new_snapshot)
    if prev is None:
        return []
    if prev.get("fingerprint") == new_snapshot.get("fingerprint"):
        return []

    prev_by_tf = {tf["name"]: tf for tf in prev.get("per_tf", [])}
    new_by_tf = {tf["name"]: tf for tf in new_snapshot.get("per_tf", [])}
    written: list[dict] = []

    for tf_name, new_tf in new_by_tf.items():
        old_tf = prev_by_tf.get(tf_name)
        if old_tf is None:
            continue
        old_top = (old_tf.get("top") or {})
        new_top = (new_tf.get("top") or {})
        old_pat = old_top.get("pattern")
        new_pat = new_top.get("pattern")
        old_pos = old_top.get("current_position")
        new_pos = new_top.get("current_position")
        if old_pat == new_pat and old_pos == new_pos:
            continue

        rid = _hash_id("trans", new_snapshot["ticker"], tf_name,
                       prev["id"], new_snapshot["id"])
        rec = {
            "id":             rid,
            "type":           "transition",
            "logged_at":      _now_iso(),
            "ticker":         new_snapshot["ticker"],
            "timeframe":      tf_name,
            "from_snapshot":  prev["id"],
            "to_snapshot":    new_snapshot["id"],
            "from_logged_at": prev.get("logged_at"),
            "to_logged_at":   new_snapshot.get("logged_at"),
            "from_pattern":   old_pat,
            "to_pattern":     new_pat,
            "from_position":  old_pos,
            "to_position":    new_pos,
            "from_score":     old_top.get("score"),
            "to_score":       new_top.get("score"),
            "price_at_to":    new_tf.get("last_price"),
        }
        _append(TRANSITIONS_PATH, rec)
        written.append(rec)
    return written


def load_transitions() -> list[dict]:
    return _load(TRANSITIONS_PATH)


# --------------------------------------------------------------------------
# Public API — manual actions
# --------------------------------------------------------------------------
def log_action(
    snapshot_id: str | None,
    ticker: str,
    action_type: str,
    *,
    direction: str | None = None,        # "long" / "short" / None
    entry: float | None = None,
    stop_loss: float | None = None,
    target: float | None = None,
    price_at_action: float | None = None,
    timeframe: str | None = None,
    notes: str = "",
    extra: dict | None = None,
) -> dict:
    """Persist a manual trader action. Returns the stored dict."""
    action_type = action_type if action_type in ACTION_TYPES else "note"
    now = _now_iso()
    aid = _hash_id("action", ticker.upper(), action_type, snapshot_id or "",
                   direction or "", now)
    record = {
        "id":              aid,
        "type":            "action",
        "logged_at":       now,
        "ticker":          ticker.upper(),
        "snapshot_id":     snapshot_id,
        "action_type":     action_type,
        "direction":       direction,
        "timeframe":       timeframe,
        "entry":           _safe_float(entry),
        "stop_loss":       _safe_float(stop_loss),
        "target":          _safe_float(target),
        "price_at_action": _safe_float(price_at_action),
        "notes":           notes or "",
        "extra":           extra or {},
        # Outcome slots (filled by ``verify_actions``):
        "outcome":         None,         # 'win'|'loss'|'open'|'expired'|'no_trade'
        "exit_price":      None,
        "exit_time":       None,
        "bars_to_resolve": None,
        "r_realized":      None,
        "verified_at":     None,
    }
    _append(ACTIONS_PATH, record)
    return record


def load_actions() -> list[dict]:
    return _load(ACTIONS_PATH)


# --------------------------------------------------------------------------
# Verification — snapshots (detector hit-rate)
# --------------------------------------------------------------------------
def _resolve_forecast_outcome(
    df: pd.DataFrame, direction: str, entry: float,
    invalidation: float | None, targets: list[dict],
) -> dict:
    """For a forecast launched at ``entry``, walk forward through ``df`` and
    decide whether the first target was hit, whether invalidation triggered,
    or whether the lookforward window expired.

    Bar-conservative: if both invalidation and first target trigger on the
    same bar we call it ``loss`` (worst-case intra-bar drawdown).
    """
    if not targets:
        return {"status": "no_targets", "exit_price": None, "exit_time": None,
                "bars_to_resolve": None, "r_realized": None}
    is_up = direction == "up"
    first_target = _safe_float(targets[0].get("price"))
    if first_target is None:
        return {"status": "no_targets", "exit_price": None, "exit_time": None,
                "bars_to_resolve": None, "r_realized": None}

    if "high" not in df.columns or "low" not in df.columns:
        df = df.copy()
        df["high"] = df["close"]
        df["low"] = df["close"]

    inv = _safe_float(invalidation)
    # Risk for R-multiple is distance to invalidation; fall back to 1% of entry.
    risk = abs(entry - inv) if (inv is not None and inv != entry) else max(abs(entry) * 0.01, 1e-9)

    for ts, row in df.iterrows():
        hi = float(row["high"])
        lo = float(row["low"])
        target_hit = (hi >= first_target) if is_up else (lo <= first_target)
        invalidated = False
        if inv is not None:
            invalidated = (lo <= inv) if is_up else (hi >= inv)

        if target_hit and invalidated:
            # Conservative — call it a loss.
            return {"status": "loss", "exit_price": float(inv),
                    "exit_time": str(ts),
                    "bars_to_resolve": int(df.index.get_loc(ts)) + 1,
                    "r_realized": -1.0}
        if invalidated:
            return {"status": "loss", "exit_price": float(inv),
                    "exit_time": str(ts),
                    "bars_to_resolve": int(df.index.get_loc(ts)) + 1,
                    "r_realized": -1.0}
        if target_hit:
            reward = abs(first_target - entry)
            return {"status": "win", "exit_price": float(first_target),
                    "exit_time": str(ts),
                    "bars_to_resolve": int(df.index.get_loc(ts)) + 1,
                    "r_realized": reward / risk}

    if len(df) == 0:
        return {"status": "open", "exit_price": None, "exit_time": None,
                "bars_to_resolve": None, "r_realized": None}
    last_close = float(df.iloc[-1]["close"])
    r = ((last_close - entry) / risk) if is_up else ((entry - last_close) / risk)
    return {"status": "expired", "exit_price": last_close,
            "exit_time": str(df.index[-1]),
            "bars_to_resolve": len(df), "r_realized": r}


def verify_snapshots(
    ohlc_fetcher: Callable[[str, str, pd.Timestamp], pd.DataFrame],
    max_bars_lookforward: dict[str, int] | None = None,
    only_open: bool = True,
) -> dict:
    """For every snapshot with actionable forecasts, walk forward and decide
    per-TF outcomes. The outcome is stored in ``snapshot["per_tf_outcomes"]``.

    ``ohlc_fetcher(ticker, tf, start_ts)`` should return a DataFrame indexed
    by timestamp with at least a ``close`` column (``high``/``low`` if
    available — recommended).
    """
    if max_bars_lookforward is None:
        max_bars_lookforward = {"1d": 60, "4h": 80, "15m": 96, "5m": 144}

    snaps = load_snapshots()
    stats = {"checked": 0, "tf_won": 0, "tf_lost": 0,
             "tf_open": 0, "tf_expired": 0, "tf_skipped": 0}
    changed = False

    for snap in snaps:
        if only_open and snap.get("verified_at") and snap.get("per_tf_outcomes"):
            # Already fully verified — check whether any TF is still open
            outcomes = snap.get("per_tf_outcomes") or {}
            still_open = any(o.get("status") == "open" for o in outcomes.values())
            if not still_open:
                continue

        snap_per_tf_outcomes: dict[str, dict] = dict(snap.get("per_tf_outcomes") or {})
        snap_changed = False

        for tf in snap.get("per_tf", []):
            tf_name = tf.get("name")
            existing = snap_per_tf_outcomes.get(tf_name)
            if existing and existing.get("status") not in (None, "open"):
                continue

            fc = tf.get("forecast")
            if (fc is None or not fc.get("is_actionable")
                    or fc.get("direction") in (None, "unclear")):
                snap_per_tf_outcomes[tf_name] = {"status": "skipped",
                                                  "reason": "not actionable"}
                stats["tf_skipped"] += 1
                snap_changed = True
                continue

            entry = tf.get("last_price")
            if entry is None:
                snap_per_tf_outcomes[tf_name] = {"status": "skipped",
                                                  "reason": "no entry price"}
                stats["tf_skipped"] += 1
                snap_changed = True
                continue

            try:
                df = ohlc_fetcher(tf.get("name"), tf.get("name"),
                                   pd.Timestamp(snap.get("logged_at"))) \
                    if False else ohlc_fetcher(snap["ticker"], tf_name,
                                                pd.Timestamp(snap.get("logged_at")))
            except Exception as exc:
                snap_per_tf_outcomes[tf_name] = {"status": "open",
                                                  "reason": f"fetch error: {exc}"}
                stats["tf_open"] += 1
                continue
            if df is None or len(df) == 0:
                snap_per_tf_outcomes[tf_name] = {"status": "open",
                                                  "reason": "no bars"}
                stats["tf_open"] += 1
                continue

            max_bars = max_bars_lookforward.get(tf_name, 60)
            df = df.iloc[:max_bars]
            res = _resolve_forecast_outcome(
                df, fc.get("direction", "unclear"), float(entry),
                fc.get("invalidation_level"), fc.get("targets") or [],
            )
            res["pattern"] = (tf.get("top") or {}).get("pattern")
            res["forecast_pattern"] = fc.get("next_pattern")
            res["forecast_direction"] = fc.get("direction")
            snap_per_tf_outcomes[tf_name] = res
            snap_changed = True
            if res["status"] == "win":
                stats["tf_won"] += 1
            elif res["status"] == "loss":
                stats["tf_lost"] += 1
            elif res["status"] == "expired":
                stats["tf_expired"] += 1
            elif res["status"] == "open":
                stats["tf_open"] += 1
            elif res["status"] == "skipped":
                stats["tf_skipped"] += 1

        if snap_changed:
            snap["per_tf_outcomes"] = snap_per_tf_outcomes
            snap["verified_at"] = _now_iso()
            stats["checked"] += 1
            changed = True

    if changed:
        _save_all(SNAPSHOTS_PATH, snaps)
    return stats


# --------------------------------------------------------------------------
# Verification — manual actions
# --------------------------------------------------------------------------
def verify_actions(
    ohlc_fetcher: Callable[[str, str, pd.Timestamp], pd.DataFrame],
    max_bars_lookforward: dict[str, int] | None = None,
    only_open: bool = True,
) -> dict:
    """Walk every open ``take_setup`` action forward and decide its outcome
    using the same logic as ``prediction_log._resolve_outcome``."""
    if max_bars_lookforward is None:
        max_bars_lookforward = {"1d": 60, "4h": 80, "15m": 96, "5m": 144}

    actions = load_actions()
    stats = {"checked": 0, "won": 0, "lost": 0, "open": 0,
             "expired": 0, "no_trade": 0}
    changed = False

    for a in actions:
        if only_open and a.get("outcome") is not None:
            continue
        stats["checked"] += 1

        if a.get("action_type") != "take_setup":
            a["outcome"] = "no_trade"
            a["verified_at"] = _now_iso()
            stats["no_trade"] += 1
            changed = True
            continue

        direction = a.get("direction")
        entry = a.get("entry")
        stop = a.get("stop_loss")
        target = a.get("target")
        if direction not in ("long", "short") or entry is None \
                or stop is None or target is None:
            a["outcome"] = "no_trade"
            a["verified_at"] = _now_iso()
            stats["no_trade"] += 1
            changed = True
            continue

        tf = a.get("timeframe") or "1d"
        try:
            df = ohlc_fetcher(a["ticker"], tf,
                              pd.Timestamp(a.get("logged_at")))
        except Exception:
            stats["open"] += 1
            continue
        if df is None or len(df) == 0:
            stats["open"] += 1
            continue

        max_bars = max_bars_lookforward.get(tf, 60)
        df = df.iloc[:max_bars]

        is_long = direction == "long"
        risk = abs(entry - stop) if entry != stop else 1e-9

        outcome = {"status": "open"}
        if "high" not in df.columns or "low" not in df.columns:
            df = df.copy()
            df["high"] = df["close"]
            df["low"] = df["close"]
        for ts, row in df.iterrows():
            hi = float(row["high"])
            lo = float(row["low"])
            stop_hit = (lo <= stop) if is_long else (hi >= stop)
            target_hit = (hi >= target) if is_long else (lo <= target)
            if stop_hit and target_hit:
                outcome = {"status": "loss", "exit_price": float(stop),
                           "exit_time": str(ts),
                           "bars_to_resolve": int(df.index.get_loc(ts)) + 1,
                           "r_realized": -1.0}
                break
            if stop_hit:
                outcome = {"status": "loss", "exit_price": float(stop),
                           "exit_time": str(ts),
                           "bars_to_resolve": int(df.index.get_loc(ts)) + 1,
                           "r_realized": -1.0}
                break
            if target_hit:
                outcome = {"status": "win", "exit_price": float(target),
                           "exit_time": str(ts),
                           "bars_to_resolve": int(df.index.get_loc(ts)) + 1,
                           "r_realized": abs(target - entry) / risk}
                break

        if outcome["status"] == "open":
            if len(df) > 0:
                last_close = float(df.iloc[-1]["close"])
                outcome = {"status": "expired",
                           "exit_price": last_close,
                           "exit_time": str(df.index[-1]),
                           "bars_to_resolve": len(df),
                           "r_realized": ((last_close - entry) / risk) if is_long
                                          else ((entry - last_close) / risk)}
            else:
                stats["open"] += 1
                continue

        a["outcome"] = outcome["status"]
        a["exit_price"] = outcome.get("exit_price")
        a["exit_time"] = outcome.get("exit_time")
        a["bars_to_resolve"] = outcome.get("bars_to_resolve")
        a["r_realized"] = outcome.get("r_realized")
        a["verified_at"] = _now_iso()
        if outcome["status"] == "win":
            stats["won"] += 1
        elif outcome["status"] == "loss":
            stats["lost"] += 1
        elif outcome["status"] == "expired":
            stats["expired"] += 1
        changed = True

    if changed:
        _save_all(ACTIONS_PATH, actions)
    return stats


# --------------------------------------------------------------------------
# Aggregate stats
# --------------------------------------------------------------------------
def compute_detector_stats(snapshots: list[dict] | None = None) -> dict:
    """Hit-rate of the wave detector, sliced by pattern and timeframe."""
    snaps = snapshots if snapshots is not None else load_snapshots()
    by_pattern: dict[str, dict] = {}
    by_tf: dict[str, dict] = {}
    by_direction: dict[str, dict] = {}
    by_pattern_tf: dict[tuple[str, str], dict] = {}

    def _bump(d: dict, key, status: str) -> None:
        bucket = d.setdefault(key, {"won": 0, "lost": 0, "expired": 0,
                                     "open": 0, "skipped": 0})
        if status in bucket:
            bucket[status] += 1

    total_tf_outcomes = 0
    for snap in snaps:
        outcomes = snap.get("per_tf_outcomes") or {}
        for tf_name, res in outcomes.items():
            status = res.get("status", "open")
            if status == "no_targets":
                status = "skipped"
            total_tf_outcomes += 1
            pat = res.get("pattern") or "?"
            dirn = res.get("forecast_direction") or "?"
            _bump(by_pattern, pat, status)
            _bump(by_tf, tf_name, status)
            _bump(by_direction, dirn, status)
            _bump(by_pattern_tf, (pat, tf_name), status)

    def _add_rates(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            decisive = v["won"] + v["lost"]
            v = dict(v)
            v["hit_rate"] = (v["won"] / decisive) if decisive else None
            out[k] = v
        return out

    return {
        "total_snapshots":   len(snaps),
        "total_tf_outcomes": total_tf_outcomes,
        "by_pattern":        _add_rates(by_pattern),
        "by_timeframe":      _add_rates(by_tf),
        "by_direction":      _add_rates(by_direction),
        "by_pattern_tf":     {f"{p}@{tf}": v
                               for (p, tf), v in _add_rates(by_pattern_tf).items()},
    }


def compute_action_stats(actions: list[dict] | None = None) -> dict:
    """Winrate / average R / breakdown by ticker & timeframe."""
    acts = actions if actions is not None else load_actions()
    out: dict[str, Any] = {
        "total":           len(acts),
        "take_setup":      0,
        "skip":            0,
        "stop_hit":        0,
        "take_profit":     0,
        "partial_close":   0,
        "note":            0,
        "verified_setups": 0,
        "won":             0,
        "lost":            0,
        "open":            0,
        "expired":         0,
        "hit_rate":        None,
        "avg_r_realized":  None,
        "by_ticker":       {},
        "by_timeframe":    {},
    }
    r_values: list[float] = []
    by_tk: dict[str, dict] = {}
    by_tf: dict[str, dict] = {}

    def _bump(d: dict, key, outcome: str) -> None:
        bucket = d.setdefault(key, {"won": 0, "lost": 0, "expired": 0})
        if outcome in bucket:
            bucket[outcome] += 1

    for a in acts:
        atype = a.get("action_type", "note")
        if atype in out:
            out[atype] += 1
        if atype != "take_setup":
            continue
        outcome = a.get("outcome")
        if outcome is None:
            out["open"] += 1
            continue
        if outcome == "no_trade":
            continue
        out["verified_setups"] += 1
        if outcome == "win":
            out["won"] += 1
        elif outcome == "loss":
            out["lost"] += 1
        elif outcome == "expired":
            out["expired"] += 1
        r = a.get("r_realized")
        if isinstance(r, (int, float)):
            r_values.append(float(r))
        _bump(by_tk, a.get("ticker", "?"), outcome)
        _bump(by_tf, a.get("timeframe") or "?", outcome)

    decisive = out["won"] + out["lost"]
    if decisive:
        out["hit_rate"] = out["won"] / decisive
    if r_values:
        out["avg_r_realized"] = sum(r_values) / len(r_values)
    out["by_ticker"] = by_tk
    out["by_timeframe"] = by_tf
    return out


def compute_stability_stats(transitions: list[dict] | None = None,
                             window_days: int = 14) -> dict:
    """How often the detector flips its top pattern, per (ticker, TF).

    A noisy detector that re-classifies every few bars is a *bad* detector.
    """
    trs = transitions if transitions is not None else load_transitions()
    cutoff = pd.Timestamp(_now_iso()) - pd.Timedelta(days=window_days)

    counts: dict[tuple[str, str], int] = {}
    pair_counts: dict[tuple[str, str, str], int] = {}
    for t in trs:
        try:
            ts = pd.Timestamp(t.get("logged_at"))
        except Exception:
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if ts < cutoff:
            continue
        key = (t.get("ticker", "?"), t.get("timeframe", "?"))
        counts[key] = counts.get(key, 0) + 1
        pair_key = (key[0], key[1],
                    f"{t.get('from_pattern','?')} → {t.get('to_pattern','?')}")
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

    return {
        "window_days":     window_days,
        "by_ticker_tf":    {f"{tk}@{tf}": c for (tk, tf), c in counts.items()},
        "by_pattern_pair": {f"{tk}@{tf} : {pat}": c
                              for (tk, tf, pat), c in pair_counts.items()},
        "total_transitions_in_window": sum(counts.values()),
    }
