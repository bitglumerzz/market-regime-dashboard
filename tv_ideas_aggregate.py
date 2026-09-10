"""Aggregate TradingView ideas into per-ticker per-tier bias signals.

Unlike Twitter sentiment (which is fuzzy NLP), TV ideas have EXPLICIT
directional labels (long/short). So aggregation is just weighted vote-counting.

Pipeline:
  1. load_recent_ideas(within_hours) — read recent snapshots
  2. aggregate_by_ticker_tier(ideas) — vote per ticker per tier
  3. save_tv_index(aggregated) → tv_ideas/tv_ideas_index.json
  4. append_tv_history(aggregated) → tv_ideas/tv_ideas_history.jsonl

The bias score per (ticker, tier) is:
  weighted_bias = (long_votes - short_votes) / total_votes ∈ [-1, +1]
Education ideas don't count.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tv_ideas_scraper import (
    TVIdea, list_snapshots, load_snapshot, SNAPSHOTS_DIR,
)

TV_INDEX_PATH = "tv_ideas/tv_ideas_index.json"
TV_HISTORY_PATH = "tv_ideas/tv_ideas_history.jsonl"


@dataclass
class TickerTVAggregate:
    ticker: str
    n_ideas_total: int
    n_long: int
    n_short: int
    n_education: int
    weighted_bias: float          # [-1, +1]; +1 = unanimous long
    label: str                    # 'bullish' | 'bearish' | 'neutral'
    by_tier: dict[str, dict] = field(default_factory=dict)
    top_ideas: list[dict] = field(default_factory=list)
    divergence: str = ""


def _label_for(score: float) -> str:
    if score >= 0.2:  return "bullish"
    if score <= -0.2: return "bearish"
    return "neutral"


def aggregate_by_ticker_tier(ideas: list[TVIdea]) -> dict[str, TickerTVAggregate]:
    """Vote-count ideas grouped by (ticker, tier). Weights from author tier."""
    by_t: dict[str, list[TVIdea]] = defaultdict(list)
    for i in ideas:
        by_t[i.ticker].append(i)

    out: dict[str, TickerTVAggregate] = {}
    for ticker, items in by_t.items():
        n_long  = sum(1 for i in items if i.direction == "long")
        n_short = sum(1 for i in items if i.direction == "short")
        n_edu   = sum(1 for i in items if i.direction == "education")
        directional = n_long + n_short

        # Tier breakdown with weights
        by_tier: dict[str, dict] = {}
        weighted_num = 0.0
        weighted_den = 0.0
        for tier_label in ("tier_1", "tier_2", "tier_3"):
            tier_items = [i for i in items if i.tier == tier_label]
            tier_long = sum(1 for i in tier_items if i.direction == "long")
            tier_short = sum(1 for i in tier_items if i.direction == "short")
            tier_dir = tier_long + tier_short
            if tier_dir == 0:
                continue
            tier_bias = (tier_long - tier_short) / tier_dir
            # Weight: tier weight × (engagement × log-followers boost)
            tier_weight_total = sum(
                i.weight * max(1.0, (i.likes + i.views / 100) / 100)
                for i in tier_items if i.direction != "education"
            )
            by_tier[tier_label] = {
                "n_ideas": len(tier_items),
                "n_long":  tier_long,
                "n_short": tier_short,
                "bias":    round(tier_bias, 3),
                "label":   _label_for(tier_bias),
                "top_ideas": _top_ideas(tier_items, k=3),
            }
            weighted_num += tier_bias * tier_weight_total
            weighted_den += tier_weight_total

        weighted_bias = (weighted_num / weighted_den) if weighted_den > 0 else 0.0
        label = _label_for(weighted_bias)

        # Divergence: tier-1 disagrees with tier-3
        divergence = ""
        t1 = by_tier.get("tier_1", {}).get("bias")
        t3 = by_tier.get("tier_3", {}).get("bias")
        if t1 is not None and t3 is not None and abs(t1 - t3) > 0.4:
            if t1 > t3:
                divergence = (
                    f"📈 TV smart money BULLISH vs crowd: tier-1 {t1:+.2f} "
                    f"vs crowd {t3:+.2f}"
                )
            else:
                divergence = (
                    f"📉 TV smart money BEARISH vs crowd: tier-1 {t1:+.2f} "
                    f"vs crowd {t3:+.2f}"
                )

        # Global top-5 ideas across all tiers (by likes × weight)
        global_top = sorted(
            [i for i in items if i.direction != "education"],
            key=lambda i: -(i.likes * i.weight),
        )[:5]
        top_serialized = [_serialize_idea(i) for i in global_top]

        out[ticker] = TickerTVAggregate(
            ticker=ticker,
            n_ideas_total=len(items),
            n_long=n_long, n_short=n_short, n_education=n_edu,
            weighted_bias=round(weighted_bias, 3),
            label=label,
            by_tier=by_tier,
            top_ideas=top_serialized,
            divergence=divergence,
        )
    return out


def _top_ideas(items: list[TVIdea], k: int) -> list[dict]:
    ranked = sorted(items, key=lambda i: -(i.likes + i.views // 100))[:k]
    return [_serialize_idea(i) for i in ranked]


def _serialize_idea(i: TVIdea) -> dict:
    return {
        "author":     i.author_handle,
        "followers":  i.author_followers,
        "tier":       i.tier,
        "direction":  i.direction,
        "title":      i.title[:200],
        "timeframe":  i.timeframe,
        "likes":      i.likes,
        "views":      i.views,
        "tags":       i.tags,
        "url":        i.url,
        "published":  i.published_at[:16],
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_tv_index(by_ticker: dict[str, TickerTVAggregate],
                   path: str = TV_INDEX_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_tickers": len(by_ticker),
        "tickers": {
            t: {
                "n_ideas_total": ts.n_ideas_total,
                "n_long": ts.n_long, "n_short": ts.n_short,
                "n_education": ts.n_education,
                "weighted_bias": ts.weighted_bias, "label": ts.label,
                "by_tier": ts.by_tier,
                "top_ideas": ts.top_ideas,
                "divergence": ts.divergence,
            }
            for t, ts in by_ticker.items()
        },
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_tv_index(path: str = TV_INDEX_PATH) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_tv_history(by_ticker: dict[str, TickerTVAggregate],
                       path: str = TV_HISTORY_PATH) -> None:
    """Append a time-series record per ticker for backtest."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        for t, ts in by_ticker.items():
            record = {
                "logged_at": now, "ticker": t,
                "weighted_bias": ts.weighted_bias,
                "label": ts.label,
                "n_long":  ts.n_long, "n_short": ts.n_short,
                "n_ideas_total": ts.n_ideas_total,
                "tier_biases": {
                    tier: d.get("bias")
                    for tier, d in ts.by_tier.items()
                },
                "divergence": ts.divergence,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_tv_history(path: str = TV_HISTORY_PATH) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Refresh entry point — sidecar will call this
# ---------------------------------------------------------------------------
def refresh_from_snapshots(within_hours: int = 24) -> dict[str, TickerTVAggregate]:
    ideas: list[TVIdea] = []
    for path in list_snapshots(within_hours=within_hours):
        ideas.extend(load_snapshot(path))
    # Dedupe by idea id
    seen: set[str] = set()
    unique = [i for i in ideas if not (i.id in seen or seen.add(i.id))]
    aggregated = aggregate_by_ticker_tier(unique)
    save_tv_index(aggregated)
    append_tv_history(aggregated)

    # Phase 6.3 — log every directional TV idea to track-record system.
    try:
        from influencer_track_record import log_tv_idea
        from cache_5m import load_cached_5m
        for idea in unique:
            if idea.direction not in ("long", "short"):
                continue
            cached = load_cached_5m(idea.ticker, max_age_minutes=60 * 24)
            price = float(cached.close.iloc[-1]) if cached else 0.0
            log_tv_idea(
                handle=idea.author_handle, ticker=idea.ticker,
                direction=idea.direction, title=idea.title,
                timeframe=idea.timeframe, posted_at=idea.published_at,
                url=idea.url, tier=idea.tier,
                followers=idea.author_followers,
                current_price=price,
            )
    except Exception:
        pass

    return aggregated
