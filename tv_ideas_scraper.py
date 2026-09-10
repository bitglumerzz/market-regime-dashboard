"""TradingView Ideas scraper via Apify.

Pulls public trade ideas for BTC/ETH/HYPE filtered by wave-analysis,
supply-and-demand, fibonacci tags. Every idea carries:
  - author handle, followers count, total ideas published, TV reputation
  - direction (long/short/education)
  - timeframe (1D, 4H, 1H, etc.)
  - likes, views, comments
  - title, optional chart snapshot URL

The author's TIER is derived automatically from followers count:
  - tier_1 (×3 weight): >50k followers — top-rated TV publishers
  - tier_2 (×2 weight): 5k-50k followers — established analysts
  - tier_3 (×1 weight): <5k followers — general crowd

This is a CRITICAL upgrade vs Twitter sentiment because TV Ideas have
EXPLICIT direction labels — no NLP parsing needed.

Apify actor: scrapemint/tradingview-ideas-scraper (FREE).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal

import requests

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APIFY_ACTOR_ID = os.getenv("APIFY_TV_IDEAS_ACTOR",
                            "scrapemint~tradingview-ideas-scraper")
APIFY_API_BASE = "https://api.apify.com/v2"
APIFY_RUN_TIMEOUT_S = 300       # TV ideas scraping can be slow

SNAPSHOTS_DIR = "tv_ideas/snapshots"

# Yahoo-ticker → TradingView symbol mapping
TICKER_TO_TV: dict[str, str] = {
    "BTC-USD":  "BTCUSD",
    "ETH-USD":  "ETHUSD",
    "HYPE-USD": "HYPEUSD",
}

# Filters: prefer wave-analysis + fibonacci + supply-demand for our use case.
# Adjust based on what you want. Empty tags = pull all directional ideas.
DEFAULT_TAGS = ["wave-analysis", "fibonacci", "supply-and-demand"]

# Tier thresholds — automatic ranking by follower count
TIER_THRESHOLDS = [
    ("tier_1", 50_000, 3.0),
    ("tier_2",  5_000, 2.0),
    ("tier_3",      0, 1.0),
]


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------
@dataclass
class TVIdea:
    id: str
    ticker: str                     # our canonical "BTC-USD"
    symbol: str                     # TV's "BTCUSD"
    author_handle: str
    author_followers: int
    author_idea_count: int
    author_reputation: float
    tier: str                       # tier_1 / 2 / 3
    weight: float
    title: str
    direction: str                  # 'long' | 'short' | 'education'
    timeframe: str                  # '1D' / '4H' / '1H' etc.
    likes: int
    views: int
    comments: int
    tags: list[str]
    published_at: str               # ISO 8601
    url: str
    image_url: str = ""
    source: str = "tradingview"


# ---------------------------------------------------------------------------
# Apify caller
# ---------------------------------------------------------------------------
def _apify_token() -> str | None:
    return os.getenv("APIFY_API_TOKEN") or os.getenv("APIFY_TOKEN")


def _run_actor(symbols: list[str], tags: list[str],
                max_per_source: int = 30) -> list[dict]:
    token = _apify_token()
    if not token:
        raise RuntimeError(
            "APIFY_API_TOKEN not set — add to .env or use --mock"
        )
    payload = {
        "symbols": symbols,
        "tags": tags or [],
        "direction": "all",
        "maxIdeasPerSource": max_per_source,
        "includeAuthorIntel": True,
        "concurrency": 5,
        "proxyConfiguration": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"],
        },
    }
    url = f"{APIFY_API_BASE}/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"
    params = {"token": token, "timeout": APIFY_RUN_TIMEOUT_S}
    r = requests.post(url, params=params, json=payload,
                       timeout=APIFY_RUN_TIMEOUT_S + 10)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected response shape: {type(data)}")
    return data


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def _tier_for(followers: int) -> tuple[str, float]:
    for tier, threshold, weight in TIER_THRESHOLDS:
        if followers >= threshold:
            return tier, weight
    return "tier_3", 1.0


def _normalize(raw: dict) -> TVIdea | None:
    """Convert one Apify item to TVIdea. Returns None if data unusable."""
    symbol = (raw.get("symbol") or "").upper()
    if not symbol:
        return None
    # Reverse-lookup our canonical ticker
    ticker = next((t for t, sym in TICKER_TO_TV.items()
                     if sym.upper() == symbol), symbol)

    author = raw.get("author") or {}
    handle = (author.get("username") or author.get("handle") or "").lstrip("@")
    if not handle:
        return None
    followers   = int(author.get("followers") or 0)
    idea_count  = int(author.get("ideasPublished") or author.get("ideaCount") or 0)
    reputation  = float(author.get("reputation") or 0)
    tier, weight = _tier_for(followers)

    direction = (raw.get("direction") or "education").lower()
    if direction not in ("long", "short", "education"):
        direction = "education"

    return TVIdea(
        id=str(raw.get("id") or ""),
        ticker=ticker,
        symbol=symbol,
        author_handle=handle,
        author_followers=followers,
        author_idea_count=idea_count,
        author_reputation=reputation,
        tier=tier, weight=weight,
        title=str(raw.get("title") or ""),
        direction=direction,
        timeframe=str(raw.get("timeframe") or raw.get("interval") or ""),
        likes=int(raw.get("likes") or raw.get("likeCount") or 0),
        views=int(raw.get("views") or raw.get("viewCount") or 0),
        comments=int(raw.get("comments") or raw.get("commentCount") or 0),
        tags=list(raw.get("tags") or []),
        published_at=str(raw.get("publishedAt") or raw.get("date")
                          or datetime.now(timezone.utc).isoformat()),
        url=str(raw.get("url") or ""),
        image_url=str(raw.get("imageUrl") or raw.get("chartImage") or ""),
    )


# ---------------------------------------------------------------------------
# Mock for tests
# ---------------------------------------------------------------------------
_MOCK_IDEAS: list[dict] = [
    {
        "id": "tv001", "symbol": "BTCUSD",
        "author": {"username": "MasterCharts", "followers": 78000,
                   "ideasPublished": 412, "reputation": 4.8},
        "title": "BTC: Ending diagonal complete, expect crash to 67k",
        "direction": "short", "timeframe": "1D",
        "likes": 1240, "views": 89000, "comments": 87,
        "tags": ["wave-analysis", "fibonacci", "elliott"],
        "publishedAt": "2026-05-13T05:30:00Z",
        "url": "https://tradingview.com/idea/tv001",
    },
    {
        "id": "tv002", "symbol": "BTCUSD",
        "author": {"username": "WaveBro", "followers": 12000,
                   "ideasPublished": 156, "reputation": 4.3},
        "title": "BTC bullish — wave 3 of 5 just starting",
        "direction": "long", "timeframe": "4H",
        "likes": 480, "views": 23000, "comments": 41,
        "tags": ["wave-analysis"],
        "publishedAt": "2026-05-13T07:00:00Z",
        "url": "https://tradingview.com/idea/tv002",
    },
    {
        "id": "tv003", "symbol": "ETHUSD",
        "author": {"username": "RandomDude", "followers": 340,
                   "ideasPublished": 8, "reputation": 3.1},
        "title": "ETH long 2300 → 2500",
        "direction": "long", "timeframe": "1H",
        "likes": 12, "views": 540, "comments": 3,
        "tags": ["fibonacci"],
        "publishedAt": "2026-05-13T08:00:00Z",
        "url": "https://tradingview.com/idea/tv003",
    },
    {
        "id": "tv004", "symbol": "ETHUSD",
        "author": {"username": "BluntzReplica", "followers": 22000,
                   "ideasPublished": 245, "reputation": 4.5},
        "title": "ETH expanding triangle into thrust down",
        "direction": "short", "timeframe": "1D",
        "likes": 720, "views": 45000, "comments": 52,
        "tags": ["wave-analysis", "elliott"],
        "publishedAt": "2026-05-13T06:00:00Z",
        "url": "https://tradingview.com/idea/tv004",
    },
    {
        "id": "tv005", "symbol": "HYPEUSD",
        "author": {"username": "HypeHunter", "followers": 8500,
                   "ideasPublished": 67, "reputation": 4.0},
        "title": "HYPE breaking 45 — next leg to 60",
        "direction": "long", "timeframe": "4H",
        "likes": 220, "views": 8400, "comments": 18,
        "tags": ["fibonacci", "supply-and-demand"],
        "publishedAt": "2026-05-13T08:30:00Z",
        "url": "https://tradingview.com/idea/tv005",
    },
]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def save_snapshot(ideas: list[TVIdea],
                   ts: datetime | None = None) -> str:
    if ts is None:
        ts = datetime.now(timezone.utc)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    fname = ts.strftime("%Y-%m-%d_%H-%M") + ".jsonl"
    path = os.path.join(SNAPSHOTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        for i in ideas:
            f.write(json.dumps(asdict(i), ensure_ascii=False) + "\n")
    return path


def load_snapshot(path: str) -> list[TVIdea]:
    if not os.path.isfile(path):
        return []
    out: list[TVIdea] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(TVIdea(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return out


def list_snapshots(within_hours: int = 48) -> list[str]:
    if not os.path.isdir(SNAPSHOTS_DIR):
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=within_hours)
    out = []
    for fname in sorted(os.listdir(SNAPSHOTS_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(SNAPSHOTS_DIR, fname)
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        if mtime >= cutoff:
            out.append(path)
    return out


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def scrape(symbols: list[str] | None = None,
            tags: list[str] | None = None,
            max_per_source: int = 30,
            mock: bool = False) -> list[TVIdea]:
    """Fetch fresh TV ideas. Returns normalized TVIdea list."""
    if symbols is None:
        symbols = list(TICKER_TO_TV.values())
    if tags is None:
        tags = DEFAULT_TAGS

    if mock:
        raw = list(_MOCK_IDEAS)
    else:
        try:
            raw = _run_actor(symbols, tags, max_per_source)
        except Exception as exc:
            print(f"  ! TV ideas scrape failed: {type(exc).__name__}: {exc}")
            return []

    ideas: list[TVIdea] = []
    for item in raw:
        i = _normalize(item)
        if i is not None:
            ideas.append(i)
    return ideas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-per-source", type=int, default=30)
    args = parser.parse_args()

    print(f"Scraping TV Ideas ({'MOCK' if args.mock else 'LIVE'})...")
    ideas = scrape(max_per_source=args.max_per_source, mock=args.mock)
    print(f"Got {len(ideas)} ideas")
    for i in ideas[:10]:
        print(f"  [{i.tier} ×{i.weight:.0f}] @{i.author_handle} ({i.author_followers}f): "
              f"{i.direction.upper()} {i.ticker} {i.timeframe} — {i.title[:60]}")
    if ideas:
        path = save_snapshot(ideas)
        print(f"Saved snapshot to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
