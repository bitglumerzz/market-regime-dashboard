"""Sentiment scoring + aggregation for scraped tweets.

Pipeline:
  1. score_with_vader(text) — fast per-tweet sentiment in [-1, +1]
  2. score_tweets(tweets) — apply VADER to each tweet, add `sentiment` field
  3. aggregate_by_ticker_tier(scored) — bucket scores by (ticker, tier),
     compute weighted means, identify divergence
  4. save_sentiment_index(aggregated) → twitter_sentiment/sentiment_index.json

We deliberately keep the sentiment computation FAST and offline (VADER) for
the bulk of posts. An optional Claude pass can rescore the top-N posts per
ticker for nuanced reading — that's behind a flag to avoid runaway API cost.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from twitter_scraper import ScrapedTweet, list_snapshots, load_snapshot


SENTIMENT_INDEX_PATH = "twitter_sentiment/sentiment_index.json"
# Append-only time series of all past aggregations, for backtesting "did
# tier-1 sentiment predict price moves?" — one record per refresh cycle.
SENTIMENT_HISTORY_PATH = "twitter_sentiment/sentiment_history.jsonl"

# Lazy singleton — VADER warms up on first use
_vader: SentimentIntensityAnalyzer | None = None


# Crypto-specific lexicon overrides. VADER scores on [-4, +4] per word.
# Tuned for typical crypto Twitter parlance.
_CRYPTO_LEXICON = {
    # Bullish terms
    "bullish":     2.5,  "bull":         2.0,
    "moon":        3.0,  "mooning":      3.0,  "moonshot":   2.5,
    "pump":        2.0,  "pumping":      2.0,  "pumped":     1.5,
    "rally":       2.0,  "rallying":     2.0,
    "breakout":    2.0,  "breakouts":    2.0,
    "accumulate":  2.0,  "accumulation": 2.0,  "accumulating": 2.0,
    "rip":         1.5,  "ripping":      2.0,
    "send":        1.5,  "sending":      1.5,
    "ath":         1.5,  "athhh":        2.0,
    "long":        1.5,  "longing":      2.0,
    "buy":         1.5,  "buying":       1.5,  "bought":     1.5,
    "hodl":        1.5,  "holding":      1.0,
    "incoming":    1.0,
    "support":     1.0,  "bounce":       1.5,  "reversal":   1.0,
    "inflow":      1.5,  "inflows":      1.5,
    "etf":         1.0,  "institutional": 1.5,

    # Bearish terms
    "bearish":     -2.5, "bear":         -2.0,
    "dump":        -2.5, "dumping":      -2.5, "dumped":     -2.0,
    "crash":       -3.0, "crashing":     -3.0, "crashed":    -2.5,
    "rekt":        -2.5,
    "liquidation": -2.0, "liquidations": -2.0, "liquidated": -2.5,
    "short":       -1.5, "shorting":     -2.0,
    "sell":        -1.5, "selling":      -1.5, "sold":       -1.5,
    "drop":        -1.5, "dropping":     -1.5, "dropped":    -1.5,
    "pullback":    -1.2, "pullbacks":    -1.2,
    "correction":  -1.0, "corrections":  -1.0,
    "rejection":   -1.5, "rejected":     -1.5,
    "outflow":     -1.5, "outflows":     -1.5,
    "fear":        -2.0, "panic":        -2.5, "capitulation": -2.5,
    "exit":        -1.0, "exits":        -1.0,
    "downtrend":   -2.0, "down":         -1.0,
    "weak":        -1.5, "weakness":     -1.5,
    "overstretched": -1.5, "overheated": -1.5, "overbought": -1.5,
    "exhaustion":  -1.5, "exhausted":    -1.5,

    # Neutral but useful structural terms
    "consolidation": 0.5,
    "range":         0.0,
    "neutral":       0.0,
}


def _get_vader() -> SentimentIntensityAnalyzer:
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
        # Inject crypto-specific scores so VADER reads CT properly
        _vader.lexicon.update(_CRYPTO_LEXICON)
    return _vader


@dataclass
class ScoredTweet:
    tweet: ScrapedTweet
    sentiment: float                         # VADER compound score [-1, +1]
    label: str = "neutral"                   # bullish / bearish / neutral


@dataclass
class TickerSentiment:
    """Aggregated sentiment for one ticker."""
    ticker: str
    n_tweets_total: int
    by_tier: dict[str, dict] = field(default_factory=dict)
    weighted_score: float = 0.0
    weighted_label: str = "neutral"
    divergence: str = ""                     # describes tier-1 vs crowd gap


def score_with_vader(text: str) -> float:
    """Return compound sentiment [-1, +1] using VADER."""
    if not text:
        return 0.0
    return float(_get_vader().polarity_scores(text)["compound"])


def _label_for(score: float) -> str:
    if score >= 0.15:
        return "bullish"
    if score <= -0.15:
        return "bearish"
    return "neutral"


def score_tweets(tweets: list[ScrapedTweet]) -> list[ScoredTweet]:
    """Score every tweet with VADER."""
    out: list[ScoredTweet] = []
    for t in tweets:
        s = score_with_vader(t.text)
        out.append(ScoredTweet(tweet=t, sentiment=s, label=_label_for(s)))
    return out


def aggregate_by_ticker_tier(
    scored: list[ScoredTweet],
    decay_hours: float = 24.0,
) -> dict[str, TickerSentiment]:
    """Bucket scored tweets by (ticker, tier) and compute weighted aggregates.

    Older posts get less weight via time-decay (exponential, half-life=12h).
    Final per-ticker score is a weighted mean of all tiers using tier weight.
    """
    now = datetime.now(timezone.utc)
    half_life_s = decay_hours * 3600.0 / 2.0

    by_ticker_tier: dict[str, dict[str, list[tuple[float, float, ScoredTweet]]]] = (
        defaultdict(lambda: defaultdict(list))
    )

    for st in scored:
        try:
            tweet_ts = pd.Timestamp(st.tweet.created_at)
            if tweet_ts.tzinfo is None:
                tweet_ts = tweet_ts.tz_localize("UTC")
            age_s = max(0.0, (now - tweet_ts.to_pydatetime()).total_seconds())
        except Exception:
            age_s = 0.0
        decay = 0.5 ** (age_s / half_life_s) if half_life_s > 0 else 1.0
        for ticker in st.tweet.tickers:
            by_ticker_tier[ticker][st.tweet.tier].append(
                (st.sentiment, decay, st)
            )

    out: dict[str, TickerSentiment] = {}
    for ticker, tier_data in by_ticker_tier.items():
        total_count = sum(len(items) for items in tier_data.values())
        tier_summary: dict[str, dict] = {}
        weighted_sum = 0.0
        weight_sum = 0.0
        for tier_label, items in tier_data.items():
            if not items:
                continue
            tier_weight = {"tier_1": 3.0, "tier_2": 2.0, "tier_3": 1.0}.get(tier_label, 1.0)
            # Per-tier weighted mean using time-decay only
            num = sum(s * d for s, d, _ in items)
            den = sum(d for _, d, _ in items)
            tier_mean = (num / den) if den > 0 else 0.0
            tier_summary[tier_label] = {
                "n_tweets": len(items),
                "mean_sentiment": round(tier_mean, 4),
                "label": _label_for(tier_mean),
                "top_posts": _select_top_posts(items, k=3),
            }
            # Overall: weight by tier weight × sum-of-decays
            weighted_sum += tier_mean * tier_weight * den
            weight_sum += tier_weight * den
        total_score = (weighted_sum / weight_sum) if weight_sum > 0 else 0.0

        # Divergence: tier1 vs tier3 difference
        divergence = ""
        t1 = tier_summary.get("tier_1", {}).get("mean_sentiment")
        t3 = tier_summary.get("tier_3", {}).get("mean_sentiment")
        if t1 is not None and t3 is not None and abs(t1 - t3) > 0.3:
            if t1 > t3:
                divergence = (
                    f"📈 SMART MONEY BULLISH vs crowd: tier-1 {t1:+.2f}, "
                    f"crowd {t3:+.2f} — потенциальный CONTRARIAN long"
                )
            else:
                divergence = (
                    f"📉 SMART MONEY BEARISH vs crowd: tier-1 {t1:+.2f}, "
                    f"crowd {t3:+.2f} — потенциальный CONTRARIAN short"
                )

        out[ticker] = TickerSentiment(
            ticker=ticker,
            n_tweets_total=total_count,
            by_tier=tier_summary,
            weighted_score=round(total_score, 4),
            weighted_label=_label_for(total_score),
            divergence=divergence,
        )
    return out


def _select_top_posts(items: list[tuple], k: int) -> list[dict]:
    """Return the top-k posts by absolute sentiment × decay (most impactful)."""
    ranked = sorted(items, key=lambda x: -abs(x[0] * x[1]))[:k]
    out = []
    for sent, decay, st in ranked:
        out.append({
            "handle":    st.tweet.author_handle,
            "name":      st.tweet.author_name,
            "text":      (st.tweet.text[:200] + "...")
                         if len(st.tweet.text) > 200 else st.tweet.text,
            "sentiment": round(sent, 3),
            "label":     st.label,
            "url":       st.tweet.url,
            "created_at": st.tweet.created_at,
        })
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_sentiment_index(by_ticker: dict[str, TickerSentiment],
                          path: str = SENTIMENT_INDEX_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "n_tickers": len(by_ticker),
        "tickers": {
            ticker: {
                "n_tweets_total": ts.n_tweets_total,
                "weighted_score": ts.weighted_score,
                "weighted_label": ts.weighted_label,
                "by_tier": ts.by_tier,
                "divergence": ts.divergence,
            }
            for ticker, ts in by_ticker.items()
        },
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_sentiment_index(path: str = SENTIMENT_INDEX_PATH) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# End-to-end refresh
# ---------------------------------------------------------------------------
def refresh_from_snapshots(within_hours: int = 24) -> dict[str, TickerSentiment]:
    """Read all recent snapshots, score, aggregate, save index. Returns aggregates."""
    tweets: list[ScrapedTweet] = []
    for path in list_snapshots(within_hours=within_hours):
        tweets.extend(load_snapshot(path))
    # Dedupe by tweet id
    seen: set[str] = set()
    unique = [t for t in tweets if not (t.id in seen or seen.add(t.id))]
    scored = score_tweets(unique)

    # Phase 6.4 — optional Claude rescore for tier-1/2 (off by default).
    # Set CLAUDE_RESCORE_TIER12=1 in .env to enable (~$0.05/day extra).
    if os.getenv("CLAUDE_RESCORE_TIER12") == "1":
        try:
            from claude_rescore import rescore_with_claude, apply_rescores_to_scored
            rescores = rescore_with_claude(scored)
            scored = apply_rescores_to_scored(scored, rescores)
        except Exception:
            pass

    aggregated = aggregate_by_ticker_tier(scored, decay_hours=within_hours)
    save_sentiment_index(aggregated)
    append_sentiment_history(aggregated)

    # Phase 6.3 — log every strongly-directional take to track-record system.
    # We use the current price = last cached price from cache_5m if available.
    try:
        from influencer_track_record import log_twitter_take
        from cache_5m import load_cached_5m
        # Snapshot prices per ticker (best effort — fall through if no cache)
        for st in scored:
            if abs(st.sentiment) < 0.3:
                continue
            for ticker in st.tweet.tickers:
                cached = load_cached_5m(ticker, max_age_minutes=60 * 24)
                price = float(cached.close.iloc[-1]) if cached else 0.0
                log_twitter_take(
                    handle=st.tweet.author_handle,
                    ticker=ticker,
                    sentiment=st.sentiment,
                    text=st.tweet.text,
                    posted_at=st.tweet.created_at,
                    url=st.tweet.url,
                    tier=st.tweet.tier,
                    current_price=price,
                )
    except Exception:
        # Track-record is best-effort; do not block the refresh.
        pass

    return aggregated


def append_sentiment_history(by_ticker: dict[str, TickerSentiment],
                              path: str = SENTIMENT_HISTORY_PATH) -> None:
    """Append a time-series record per ticker for later backtesting."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        for ticker, ts in by_ticker.items():
            record = {
                "logged_at": now,
                "ticker": ticker,
                "weighted_score": ts.weighted_score,
                "weighted_label": ts.weighted_label,
                "n_tweets_total": ts.n_tweets_total,
                "by_tier": {
                    tier: {
                        "n_tweets": d["n_tweets"],
                        "mean_sentiment": d["mean_sentiment"],
                        "label": d["label"],
                    }
                    for tier, d in ts.by_tier.items()
                },
                "divergence": ts.divergence,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_sentiment_history(path: str = SENTIMENT_HISTORY_PATH) -> list[dict]:
    """Read every past sentiment-aggregation record."""
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
