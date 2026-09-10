"""Incremental scrape state for Twitter.

Tracks the **timestamp of the last successful scrape** + a rolling list of
**seen tweet IDs** so each new scrape only pays Apify for fresh content.

Two layers of dedup:

  1. **Server-side filter** — we pass `since=last_scrape_time` to the Apify
     actor. The actor itself skips older tweets, so we pay only for content
     created after the previous run. This is the big cost saver.

  2. **Client-side dedup** — after fetching, we filter against `seen_ids`.
     If the actor returns a tweet we already processed (e.g. because we
     re-ran within seconds, or the actor's `since` is slightly fuzzy),
     it gets dropped instead of being saved a second time.

State file: `twitter_sentiment/scrape_state.json`. Bind-mounted via the
twitter_sentiment volume so it survives container restarts.

`seen_ids` is bounded to ~5000 entries (oldest dropped) so the file doesn't
grow indefinitely. Three weeks of two-a-day scrapes × 100 accounts × ~3 new
tweets each ≈ 4200 IDs, so 5000 is a reasonable cap.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

STATE_DIR = "twitter_sentiment"
STATE_PATH = os.path.join(STATE_DIR, "scrape_state.json")

# Cap to prevent unbounded file growth. ~5000 IDs ≈ 200KB JSON.
MAX_SEEN_IDS = 5000

# How far back to scan if there's no prior state (fresh container, first
# ever scrape). 24 hours is plenty for influencer sentiment.
DEFAULT_INITIAL_LOOKBACK_HOURS = 24


def _ensure_dir() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load() -> dict:
    if not os.path.isfile(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    _ensure_dir()
    tmp = STATE_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, STATE_PATH)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def get_since(default_lookback_hours: int = DEFAULT_INITIAL_LOOKBACK_HOURS
              ) -> datetime:
    """Return the timestamp from which to ask Apify for new tweets.

    On first run there's no prior state → return now - default_lookback_hours.
    On subsequent runs → return the saved last_scrape_time.

    Subtracts a 5-minute safety margin so we don't miss tweets posted right
    at the scrape boundary (actor's `since` is sometimes slightly lossy).
    """
    data = _load()
    saved = data.get("last_scrape_time")
    if saved:
        try:
            t = datetime.fromisoformat(saved)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            # 5-min safety overlap to catch boundary tweets
            return t - timedelta(minutes=5)
        except ValueError:
            pass
    return _now() - timedelta(hours=default_lookback_hours)


def get_seen_ids() -> set[str]:
    """Set of tweet IDs we've already processed. Used to dedup."""
    data = _load()
    ids = data.get("seen_ids", [])
    return set(str(i) for i in ids if i)


def mark_scrape_complete(seen_now: list[str]) -> dict:
    """Persist that a scrape completed successfully.

    Updates last_scrape_time to now, adds any new tweet IDs to seen_ids,
    trims seen_ids to MAX_SEEN_IDS (FIFO — newest kept). Returns the
    resulting state dict for diagnostics.
    """
    data = _load()
    data["last_scrape_time"] = _now().isoformat(timespec="seconds")

    existing = data.get("seen_ids", [])
    # Append new ones in order, dedup via dict.fromkeys (preserves insertion order)
    merged = list(dict.fromkeys(list(existing) + [str(i) for i in seen_now if i]))
    if len(merged) > MAX_SEEN_IDS:
        merged = merged[-MAX_SEEN_IDS:]  # keep newest
    data["seen_ids"] = merged
    data["total_scrapes"] = int(data.get("total_scrapes", 0)) + 1

    _save(data)
    return data


def reset() -> None:
    """Wipe state (e.g. when changing the influencer registry materially)."""
    if os.path.isfile(STATE_PATH):
        try:
            os.remove(STATE_PATH)
        except OSError:
            pass


def stats() -> dict:
    """Diagnostics for the UI: when last scraped, how many IDs cached, etc."""
    data = _load()
    last_at = data.get("last_scrape_time")
    age_minutes = None
    if last_at:
        try:
            t = datetime.fromisoformat(last_at)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            age_minutes = (_now() - t).total_seconds() / 60.0
        except ValueError:
            pass
    return {
        "last_scrape_time":  last_at,
        "age_minutes":       age_minutes,
        "seen_ids_count":    len(data.get("seen_ids", [])),
        "total_scrapes":     data.get("total_scrapes", 0),
    }
