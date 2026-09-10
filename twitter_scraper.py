"""Twitter scraper via Apify.

Pulls recent tweets from accounts listed in `influencer_registry`, filters
by ticker keywords, and saves a snapshot to `twitter_sentiment/snapshots/`.

Apify actor used: `apidojo/twitter-scraper` (configurable via env var).
The Apify token is read from `APIFY_API_TOKEN` in `.env`.

CLI usage:
    python twitter_scraper.py --hours 24             # live run, last 24h
    python twitter_scraper.py --mock                 # use a mock response for tests
    python twitter_scraper.py --max-tweets-per-acct 5

If APIFY_API_TOKEN is missing, the script logs a warning and falls back to
empty snapshot — keeping the rest of the system unblocked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import requests

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent))
from influencer_registry import all_accounts, Account, tickers_for, tier_of, tier_weight


SNAPSHOTS_DIR = "twitter_sentiment/snapshots"
# Correct actor name (verified 2026-05-13): apidojo/tweet-scraper.
# Slashes in URLs become tildes per Apify convention.
APIFY_ACTOR_ID = os.getenv("APIFY_TWITTER_ACTOR", "apidojo~tweet-scraper")
APIFY_API_BASE = "https://api.apify.com/v2"
DEFAULT_MAX_TWEETS_PER_ACCOUNT = 10
APIFY_RUN_TIMEOUT_S = 180


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------
@dataclass
class ScrapedTweet:
    id: str
    author_handle: str
    author_name: str
    tier: str
    weight: float
    text: str
    created_at: str           # ISO 8601
    url: str
    tickers: list[str]        # e.g. ["BTC-USD"]
    metrics: dict             # likes, retweets, etc.
    source: str = "apify"


# ---------------------------------------------------------------------------
# Apify caller
# ---------------------------------------------------------------------------
def _apify_token() -> str | None:
    return os.getenv("APIFY_API_TOKEN") or os.getenv("APIFY_TOKEN")


def _run_apify_actor(handles: list[str],
                      max_tweets: int,
                      since: datetime) -> list[dict]:
    """Synchronously run the Apify actor and return raw tweet dicts.

    Uses the `start` parameter (apidojo/tweet-scraper's date filter) to ask
    only for tweets newer than ``since``. This is the primary cost saver —
    after the first scrape, subsequent scrapes only pay for new content.
    """
    token = _apify_token()
    if not token:
        raise RuntimeError(
            "APIFY_API_TOKEN not set — add it to .env "
            "or pass --mock to test without live API."
        )

    # apidojo/tweet-scraper input schema (verified May 2026):
    #   twitterHandles: list of accounts (without @)
    #   maxItems: total cap across all handles
    #   sort: "Latest"
    #   tweetLanguage: optional ("en", "ru", etc.)
    #   start: YYYY-MM-DD or ISO datetime — filter tweets newer than this
    #   end:   optional cutoff in the future (we don't set it)
    payload = {
        "twitterHandles": handles,
        "maxItems": max_tweets * len(handles),
        "sort": "Latest",
        "tweetLanguage": "en",
        # Phase 25 — incremental scraping. Format that apidojo accepts:
        # ISO 8601 with timezone. The actor server-side filters out older
        # tweets so we don't pay compute units for content we already have.
        "start": since.strftime("%Y-%m-%dT%H:%M:%SZ") if since.tzinfo is None
                  else since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    url = f"{APIFY_API_BASE}/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"
    params = {"token": token, "timeout": APIFY_RUN_TIMEOUT_S}
    response = requests.post(url, params=params, json=payload,
                              timeout=APIFY_RUN_TIMEOUT_S + 10)
    response.raise_for_status()
    items = response.json()
    if not isinstance(items, list):
        raise RuntimeError(f"Unexpected Apify response shape: {type(items)}")
    return items


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def _normalize_raw(raw: dict) -> ScrapedTweet | None:
    """Convert apidojo/twitter-scraper item → ScrapedTweet.

    Field names verified May 2026. Other actors will need a separate adapter.
    """
    handle = (raw.get("author", {}).get("userName")
              or raw.get("user", {}).get("screen_name")
              or "")
    handle = handle.lstrip("@")
    if not handle:
        return None

    text = raw.get("text") or raw.get("full_text") or ""
    if not text.strip():
        return None

    name = (raw.get("author", {}).get("name")
            or raw.get("user", {}).get("name") or handle)

    tier = tier_of(handle) or "unknown"
    weight = tier_weight(handle) or 0.0
    if tier == "unknown":
        # Not in our registry → skip (we may pick up retweets, replies, etc.)
        return None

    tickers = tickers_for(text)
    if not tickers:
        return None    # only keep crypto-relevant tweets

    tweet_id = str(raw.get("id") or raw.get("id_str") or "")
    created_at = (raw.get("createdAt")
                  or raw.get("created_at")
                  or datetime.now(timezone.utc).isoformat())
    url = raw.get("url") or f"https://twitter.com/{handle}/status/{tweet_id}"

    metrics = {
        "likes":    raw.get("likeCount")    or raw.get("favorite_count")    or 0,
        "retweets": raw.get("retweetCount") or raw.get("retweet_count")     or 0,
        "replies":  raw.get("replyCount")   or raw.get("reply_count")       or 0,
        "views":    raw.get("viewCount")    or raw.get("view_count")        or 0,
    }

    return ScrapedTweet(
        id=tweet_id, author_handle=handle, author_name=name,
        tier=tier, weight=weight,
        text=text, created_at=str(created_at),
        url=url, tickers=tickers, metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Mock for tests
# ---------------------------------------------------------------------------
_MOCK_TWEETS: list[dict] = [
    {
        "id": "1001",
        "author": {"userName": "saylor", "name": "Michael Saylor"},
        "text": "Bitcoin at 100k is just the beginning. $BTC accumulation continues.",
        "createdAt": "2026-05-13T07:00:00Z",
        "url": "https://twitter.com/saylor/status/1001",
        "likeCount": 12000, "retweetCount": 3400, "replyCount": 890,
    },
    {
        "id": "1002",
        "author": {"userName": "CryptoHayes", "name": "Arthur Hayes"},
        "text": "ETH looks overstretched on 1d. Expecting a 20% pullback before the next leg up.",
        "createdAt": "2026-05-13T05:30:00Z",
        "url": "https://twitter.com/CryptoHayes/status/1002",
        "likeCount": 8500, "retweetCount": 1200, "replyCount": 410,
    },
    {
        "id": "1003",
        "author": {"userName": "Bluntz_Capital", "name": "Bluntz"},
        "text": "$HYPE wave 5 looking complete. Risk:reward favors patience here.",
        "createdAt": "2026-05-13T03:00:00Z",
        "url": "https://twitter.com/Bluntz_Capital/status/1003",
        "likeCount": 2300, "retweetCount": 320, "replyCount": 110,
    },
    {
        "id": "1004",
        "author": {"userName": "WatcherGuru", "name": "Watcher.Guru"},
        "text": "JUST IN: Bitcoin spot ETF saw $1.2B inflows yesterday.",
        "createdAt": "2026-05-13T02:00:00Z",
        "url": "https://twitter.com/WatcherGuru/status/1004",
        "likeCount": 4500, "retweetCount": 1800, "replyCount": 220,
    },
    {
        "id": "1005",
        "author": {"userName": "Pentosh1", "name": "Pentoshi"},
        "text": "BTC is going to dump hard. Bear flag forming on 4h.",
        "createdAt": "2026-05-13T01:00:00Z",
        "url": "https://twitter.com/Pentosh1/status/1005",
        "likeCount": 6200, "retweetCount": 880, "replyCount": 340,
    },
]


# ---------------------------------------------------------------------------
# Snapshot save / load
# ---------------------------------------------------------------------------
def save_snapshot(tweets: list[ScrapedTweet],
                   ts: datetime | None = None) -> str:
    """Append tweets to a timestamped JSONL snapshot. Returns the file path."""
    if ts is None:
        ts = datetime.now(timezone.utc)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    fname = ts.strftime("%Y-%m-%d_%H-%M") + ".jsonl"
    path = os.path.join(SNAPSHOTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        for t in tweets:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
    return path


def load_snapshot(path: str) -> list[ScrapedTweet]:
    if not os.path.isfile(path):
        return []
    out: list[ScrapedTweet] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append(ScrapedTweet(**d))
            except (json.JSONDecodeError, TypeError):
                continue
    return out


def list_snapshots(within_hours: int = 48) -> list[str]:
    """Return paths of snapshot files newer than `within_hours`."""
    if not os.path.isdir(SNAPSHOTS_DIR):
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=within_hours)
    out: list[str] = []
    for fname in sorted(os.listdir(SNAPSHOTS_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(SNAPSHOTS_DIR, fname)
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        if mtime >= cutoff:
            out.append(path)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def scrape(hours_back: int = 24,
            max_tweets_per_account: int = DEFAULT_MAX_TWEETS_PER_ACCOUNT,
            mock: bool = False,
            incremental: bool = True,
            ) -> list[ScrapedTweet]:
    """Fetch fresh tweets from the registry and return normalized list.

    Args:
        hours_back: fallback lookback if state has no prior scrape recorded.
        max_tweets_per_account: cap per handle per scrape.
        mock: use built-in mock tweets (no Apify call) — for tests.
        incremental: if True (default), use twitter_scrape_state to fetch
            ONLY tweets newer than the last scrape. Pass False to force a
            full re-scrape (e.g. backfill or manual debugging).

    Cost optimization (Phase 25): incremental scraping cuts Apify compute
    cost by 5-10× because subsequent runs only pay for new content. The
    `since` parameter is server-side; client-side dedup via seen_ids
    catches any duplicates the actor's filter misses.
    """
    accounts = all_accounts()
    handles = [a.handle for a in accounts]

    # Decide `since` window: incremental mode uses last_scrape_time, full
    # mode uses now - hours_back (legacy behavior).
    if incremental and not mock:
        from twitter_scrape_state import get_since, get_seen_ids
        since = get_since(default_lookback_hours=hours_back)
        seen_before = get_seen_ids()
    else:
        since = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        seen_before = set()

    if mock:
        raw_items = list(_MOCK_TWEETS)
    else:
        try:
            raw_items = _run_apify_actor(handles, max_tweets_per_account, since)
        except Exception as exc:
            print(f"  ! Apify scrape failed: {type(exc).__name__}: {exc}")
            print("    Returning empty result (system continues without sentiment data)")
            return []

    tweets: list[ScrapedTweet] = []
    skipped_dup = 0
    for raw in raw_items:
        t = _normalize_raw(raw)
        if t is None:
            continue
        # Client-side dedup — catches anything the server-side `since`
        # filter missed (boundary tweets, actor lag, etc).
        if incremental and t.id and t.id in seen_before:
            skipped_dup += 1
            continue
        tweets.append(t)

    if incremental and not mock:
        from twitter_scrape_state import mark_scrape_complete
        try:
            new_ids = [t.id for t in tweets if t.id]
            mark_scrape_complete(new_ids)
        except Exception as e:
            print(f"  ! Failed to update scrape state: "
                  f"{type(e).__name__}: {e}")
        if skipped_dup:
            print(f"  • dedup: skipped {skipped_dup} already-seen tweet(s)")
        if not tweets:
            print(f"  • no new tweets since {since.isoformat()} — "
                  f"cheapest scrape possible")

    return tweets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24,
                        help="fallback lookback hours if no prior scrape state")
    parser.add_argument("--max-tweets-per-acct", type=int, default=10)
    parser.add_argument("--mock", action="store_true",
                        help="use built-in mock data (no Apify call)")
    parser.add_argument("--save", action="store_true", default=True,
                        help="save result to snapshot file")
    parser.add_argument("--full-rescrape", action="store_true",
                        help="ignore incremental state, fetch ALL tweets in "
                             "the --hours window (more expensive — use for "
                             "backfill or after major registry changes)")
    parser.add_argument("--reset-state", action="store_true",
                        help="wipe scrape_state.json before running")
    args = parser.parse_args()

    if args.reset_state:
        from twitter_scrape_state import reset, stats
        reset()
        print(f"State wiped. Pre-reset stats: {stats()}")

    if not args.mock and not args.full_rescrape:
        from twitter_scrape_state import stats
        s = stats()
        if s["last_scrape_time"]:
            print(f"Incremental: last scrape was "
                  f"{s['age_minutes']:.1f} min ago, "
                  f"{s['seen_ids_count']} IDs cached "
                  f"(total scrapes: {s['total_scrapes']})")
        else:
            print(f"First scrape — falling back to {args.hours}h lookback")

    print(f"Scraping Twitter "
          f"({'MOCK' if args.mock else 'LIVE'}, "
          f"{'FULL' if args.full_rescrape else 'incremental'})…")
    tweets = scrape(hours_back=args.hours,
                     max_tweets_per_account=args.max_tweets_per_acct,
                     mock=args.mock,
                     incremental=not args.full_rescrape)
    print(f"Got {len(tweets)} ticker-relevant tweets")
    for t in tweets[:5]:
        print(f"  [{t.tier} ×{t.weight:.0f}] @{t.author_handle}: "
              f"{t.text[:80]}... (tickers={t.tickers})")

    if args.save and tweets:
        path = save_snapshot(tweets)
        print(f"Saved snapshot to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
