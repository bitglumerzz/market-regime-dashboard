"""Universal audit log for every Claude API call.

Every call to ask_claude or ask_claude_holistic appends one record to
`claude_calls/calls.jsonl`. The log captures:
  - timestamp, ticker, type (per_tf | holistic)
  - the FULL prompt sent and FULL response received
  - token usage and rough cost estimate
  - parsed AIWaveOpinion / AIHolisticOpinion (or error info)
  - call duration

Purpose:
  - Audit trail: see exactly what context produced any opinion
  - Debug: compare prompts when output is surprising
  - Backtest: with prompt + response + price history, we can later
    measure "did high setup_quality really lead to wins?"
  - Cost tracking: aggregate token usage / month per type / ticker

Storage: JSONL at CALLS_PATH. Bind-mounted to ./claude_calls in docker.
Reads never block: returns empty list if file missing.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

CALLS_DIR = "claude_calls"
CALLS_PATH = os.path.join(CALLS_DIR, "calls.jsonl")

# Anthropic pricing (May 2026) — Claude Opus 4.x. Update if model/pricing changes.
# Per million tokens.
PRICE_INPUT_PER_M_USD = 15.0
PRICE_OUTPUT_PER_M_USD = 75.0


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def log_call(
    *,
    ticker: str,
    call_type: str,            # "per_tf" | "holistic"
    timeframe: str = "",
    model: str = "",
    prompt: str = "",
    raw_response: str = "",
    parsed: dict | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    duration_s: float = 0.0,
    error: str = "",
    cached: bool = False,
) -> dict:
    """Append one call record. Never raises — logging must not break the UI."""
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": ticker,
        "type": call_type,
        "timeframe": timeframe,
        "model": model,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed": parsed or {},
        "tokens_in":  int(tokens_in),
        "tokens_out": int(tokens_out),
        "cost_usd":   _estimate_cost(tokens_in, tokens_out),
        "duration_s": float(duration_s),
        "error":      error,
        "cached":     bool(cached),
    }
    try:
        os.makedirs(CALLS_DIR, exist_ok=True)
        with open(CALLS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must NEVER block the actual Claude call.
        pass
    return record


def load_all(limit: int | None = None) -> list[dict]:
    """Read every call record. `limit` returns the most recent N."""
    if not os.path.isfile(CALLS_PATH):
        return []
    out: list[dict] = []
    with open(CALLS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None and limit > 0:
        return out[-limit:]
    return out


def cost_summary() -> dict:
    """Aggregate cost stats across all calls."""
    calls = load_all()
    total_in = sum(c.get("tokens_in", 0) for c in calls)
    total_out = sum(c.get("tokens_out", 0) for c in calls)
    total_cost = sum(c.get("cost_usd", 0.0) for c in calls)
    by_type: dict[str, dict] = {}
    by_ticker: dict[str, dict] = {}
    for c in calls:
        for key, holder in [("type", by_type), ("ticker", by_ticker)]:
            k = c.get(key, "?")
            b = holder.setdefault(k, {"calls": 0, "tokens_in": 0,
                                       "tokens_out": 0, "cost_usd": 0.0})
            b["calls"] += 1
            b["tokens_in"]  += c.get("tokens_in", 0)
            b["tokens_out"] += c.get("tokens_out", 0)
            b["cost_usd"]   += c.get("cost_usd", 0.0)
    return {
        "n_calls": len(calls),
        "tokens_in_total":  total_in,
        "tokens_out_total": total_out,
        "cost_usd_total":   round(total_cost, 4),
        "by_type":   by_type,
        "by_ticker": by_ticker,
    }


def _estimate_cost(tokens_in: int, tokens_out: int) -> float:
    return round(
        tokens_in / 1e6 * PRICE_INPUT_PER_M_USD
        + tokens_out / 1e6 * PRICE_OUTPUT_PER_M_USD,
        6,
    )
