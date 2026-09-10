"""On-demand Apify scraping utility.

Use this for ad-hoc scraping that doesn't fit the sidecar refresh schedule:
  - Backfill: re-pull data for a specific account or symbol on demand
  - Investigation: check what an influencer posted around a key event
  - Verification: cross-check what live sidecar collected
  - Experimentation: try new accounts before adding to the curated registry

Usage examples:
    # Twitter — specific accounts, last 24h
    python apify_oneshot.py twitter --handles saylor cz_binance --hours 24

    # TV ideas — specific symbol
    python apify_oneshot.py tv-ideas --symbols BTCUSD --max 50

    # Twitter — direct search query
    python apify_oneshot.py twitter --search "bitcoin halving" --hours 72

    # Save to specific snapshot file
    python apify_oneshot.py twitter --handles VitalikButerin --hours 168 \\
        --out twitter_sentiment/snapshots/vitalik_week.jsonl

Reuses APIFY_API_TOKEN from .env. All output normalized to the same JSONL
format the sidecars use — drop into snapshots/ and the system picks it up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))


APIFY_API_BASE = "https://api.apify.com/v2"


def _token() -> str:
    tok = os.getenv("APIFY_API_TOKEN") or os.getenv("APIFY_TOKEN")
    if not tok:
        print("ERROR: APIFY_API_TOKEN not set in .env", file=sys.stderr)
        sys.exit(2)
    return tok


def _run_actor(actor_id: str, payload: dict, timeout_s: int = 240) -> list[dict]:
    url = f"{APIFY_API_BASE}/acts/{actor_id}/run-sync-get-dataset-items"
    params = {"token": _token(), "timeout": timeout_s}
    r = requests.post(url, params=params, json=payload, timeout=timeout_s + 10)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected response shape: {type(data)}")
    return data


# ---------------------------------------------------------------------------
# Twitter
# ---------------------------------------------------------------------------
def cmd_twitter(args) -> int:
    from twitter_scraper import _normalize_raw, save_snapshot

    actor_id = "apidojo~tweet-scraper"
    payload: dict = {
        "maxItems": (args.max or 20) * max(1, len(args.handles or [])),
        "sort": "Latest",
        "tweetLanguage": args.lang or "en",
    }
    if args.handles:
        payload["twitterHandles"] = args.handles
    if args.search:
        payload["searchTerms"] = [args.search]

    print(f"Apify call: {actor_id}")
    print(f"  payload: {payload}")
    raw = _run_actor(actor_id, payload)
    print(f"  raw items: {len(raw)}")

    tweets = []
    for r in raw:
        t = _normalize_raw(r)
        if t is not None:
            tweets.append(t)
    print(f"  normalized tweets: {len(tweets)}")
    for t in tweets[:5]:
        print(f"    [{t.tier} ×{t.weight:.0f}] @{t.author_handle}: "
              f"{t.text[:80]}...")
    if not tweets:
        print("  (no ticker-relevant tweets after filtering)")
        return 0

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for t in tweets:
                f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
        print(f"  saved → {args.out}")
    else:
        path = save_snapshot(tweets)
        print(f"  saved → {path}")
    return 0


# ---------------------------------------------------------------------------
# TV Ideas
# ---------------------------------------------------------------------------
def cmd_tv_ideas(args) -> int:
    from tv_ideas_scraper import _normalize, save_snapshot

    actor_id = "scrapemint~tradingview-ideas-scraper"
    payload = {
        "symbols": args.symbols or [],
        "tags": args.tags or [],
        "direction": args.direction or "all",
        "maxIdeasPerSource": args.max or 30,
        "includeAuthorIntel": True,
        "concurrency": 5,
        "proxyConfiguration": {"useApifyProxy": True,
                               "apifyProxyGroups": ["RESIDENTIAL"]},
    }
    print(f"Apify call: {actor_id}")
    print(f"  payload: {payload}")
    raw = _run_actor(actor_id, payload)
    print(f"  raw items: {len(raw)}")

    ideas = []
    for r in raw:
        i = _normalize(r)
        if i is not None:
            ideas.append(i)
    print(f"  normalized ideas: {len(ideas)}")
    for i in ideas[:5]:
        print(f"    [{i.tier} ×{i.weight:.0f}] @{i.author_handle} "
              f"({i.author_followers}f): {i.direction.upper()} "
              f"{i.ticker} — {i.title[:60]}")

    if not ideas:
        return 0

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for i in ideas:
                f.write(json.dumps(asdict(i), ensure_ascii=False) + "\n")
        print(f"  saved → {args.out}")
    else:
        path = save_snapshot(ideas)
        print(f"  saved → {path}")
    return 0


# ---------------------------------------------------------------------------
# Generic — run any actor by ID with raw input JSON
# ---------------------------------------------------------------------------
def cmd_raw(args) -> int:
    """Run any Apify actor by ID with raw JSON input — escape hatch."""
    if not args.input:
        print("ERROR: --input required", file=sys.stderr)
        return 1
    try:
        payload = json.loads(args.input)
    except json.JSONDecodeError as exc:
        print(f"ERROR: bad JSON: {exc}", file=sys.stderr)
        return 1

    print(f"Apify call: {args.actor_id}")
    print(f"  payload: {payload}")
    raw = _run_actor(args.actor_id, payload)
    print(f"  raw items: {len(raw)}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in raw:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  saved → {args.out}")
    else:
        # Just dump first 3 items for inspection
        for r in raw[:3]:
            print(json.dumps(r, indent=2, ensure_ascii=False)[:500])
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # twitter
    p_tw = sub.add_parser("twitter", help="on-demand Twitter scrape")
    p_tw.add_argument("--handles", nargs="+", help="accounts (without @)")
    p_tw.add_argument("--search",  help="search query (alternative to --handles)")
    p_tw.add_argument("--hours",   type=int, default=24, help="lookback hours")
    p_tw.add_argument("--lang",    default="en")
    p_tw.add_argument("--max",     type=int, default=20, help="per-account max")
    p_tw.add_argument("--out",     help="output JSONL path")
    p_tw.set_defaults(func=cmd_twitter)

    # tv-ideas
    p_tv = sub.add_parser("tv-ideas", help="on-demand TradingView ideas")
    p_tv.add_argument("--symbols", nargs="+", help="e.g. BTCUSD ETHUSD")
    p_tv.add_argument("--tags",    nargs="+", help="e.g. wave-analysis fibonacci")
    p_tv.add_argument("--direction", choices=["all","long","short","education"],
                        default="all")
    p_tv.add_argument("--max",     type=int, default=30, help="per-source max")
    p_tv.add_argument("--out",     help="output JSONL path")
    p_tv.set_defaults(func=cmd_tv_ideas)

    # raw — any actor
    p_raw = sub.add_parser("raw", help="run any Apify actor by ID")
    p_raw.add_argument("--actor-id", required=True,
                        help="e.g. apidojo~tweet-scraper")
    p_raw.add_argument("--input",    required=True, help="JSON input")
    p_raw.add_argument("--out",      help="output JSONL path")
    p_raw.set_defaults(func=cmd_raw)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
