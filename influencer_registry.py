"""Reader and helper API for the influencer registry.

Loads `twitter_sentiment/influencers.json` and exposes:
  - all_accounts() — flat list of handles
  - tier_weight(handle) — weight multiplier (3 / 2 / 1)
  - tier_of(handle) — "tier_1" / "tier_2" / "tier_3" / None
  - tickers_for(text) — match tweet body against ticker keywords
  - track_record(handle) — placeholder for future prediction-accuracy tracking
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REGISTRY_PATH = "twitter_sentiment/influencers.json"

_TIER_KEYS = ("tier_1_market_movers", "tier_2_quality_analysts",
               "tier_3_crowd_sentiment")
_TIER_LABEL = {
    "tier_1_market_movers": "tier_1",
    "tier_2_quality_analysts": "tier_2",
    "tier_3_crowd_sentiment": "tier_3",
}


@dataclass
class Account:
    handle: str            # without @
    name: str
    focus: str
    tier: str              # "tier_1" / "tier_2" / "tier_3"
    weight: float


def _load() -> dict:
    if not os.path.isfile(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def all_accounts() -> list[Account]:
    """Return flat list of all accounts across all tiers."""
    reg = _load()
    meta = reg.get("_meta", {})
    out: list[Account] = []
    weights = {
        "tier_1_market_movers":   float(meta.get("tier_1_weight", 3.0)),
        "tier_2_quality_analysts":float(meta.get("tier_2_weight", 2.0)),
        "tier_3_crowd_sentiment": float(meta.get("tier_3_weight", 1.0)),
    }
    for tier_key in _TIER_KEYS:
        tier_data = reg.get(tier_key, {})
        for a in tier_data.get("accounts", []):
            out.append(Account(
                handle=a["handle"].lstrip("@"),
                name=a.get("name", ""),
                focus=a.get("focus", ""),
                tier=_TIER_LABEL[tier_key],
                weight=weights[tier_key],
            ))
    # Deduplicate by handle (last entry wins)
    by_handle: dict[str, Account] = {}
    for a in out:
        by_handle[a.handle.lower()] = a
    return list(by_handle.values())


def tier_of(handle: str) -> str | None:
    for a in all_accounts():
        if a.handle.lower() == handle.lower().lstrip("@"):
            return a.tier
    return None


def tier_weight(handle: str) -> float:
    for a in all_accounts():
        if a.handle.lower() == handle.lower().lstrip("@"):
            return a.weight
    return 0.0


def tickers_for(text: str) -> list[str]:
    """Return all tickers mentioned in `text` (case-insensitive)."""
    if not text:
        return []
    reg = _load()
    tickers_kw = reg.get("tickers_keywords", {})
    text_lower = text.lower()
    out: list[str] = []
    for ticker, keywords in tickers_kw.items():
        if ticker.startswith("_"):
            continue
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in text_lower:
                out.append(ticker)
                break
    return out


def track_record(handle: str) -> dict:
    """Placeholder for future prediction-tracking. Returns hit-rate etc.

    Long-term plan: every time an influencer posts a directional take, log it
    and verify later (same as Claude predictions). For now returns None values.
    """
    return {
        "handle": handle,
        "n_predictions": 0,
        "hit_rate": None,
        "avg_lead_time_days": None,
        "note": "track-record system not yet wired",
    }


def summary() -> dict:
    """Quick stats for the UI."""
    accts = all_accounts()
    by_tier: dict[str, int] = {}
    for a in accts:
        by_tier[a.tier] = by_tier.get(a.tier, 0) + 1
    return {
        "total": len(accts),
        "by_tier": by_tier,
        "registry_path": REGISTRY_PATH,
    }
