"""Rescore tier-1/tier-2 posts using Claude for nuanced crypto sentiment.

VADER + crypto lexicon works fine for mass posts but mis-reads trader-speak:
  - "wave 5 looking complete" — VADER reads +0.25 bullish, but for traders
    it means SELL (the trend is ending)
  - "Risk:reward favors patience" — VADER reads neutral, but it's a HOLD signal
  - "Bear flag forming" — VADER reads as bearish (correct), but VADER scores
    are imprecise — Claude can give a calibrated [-1, +1] read

We only rescore the top ~20-30 highest-impact posts (tier-1/tier-2) since
each rescore costs ~$0.002. Total cost: ~$0.05/day.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# Maximum posts to rescore per call (cost cap)
DEFAULT_BATCH_SIZE = 30

CLAUDE_PROMPT = """Ты — аналитик crypto trading sentiment. Оцениваешь каждый
пост по шкале от -1.0 (сильно медвежий, прямой sell-signal) до +1.0
(сильно бычий, прямой buy-signal). 0 = нейтральный/непрямой.

ВАЖНО: оценивай ТРЕЙДЕРСКИЙ смысл, не общую тональность:
  - "wave 5 complete" / "topping pattern" / "exhaustion" → BEARISH (-0.5..-0.8)
  - "wave 3 starting" / "bullish breakout" / "accumulation zone" → BULLISH (+0.5..+0.8)
  - "Risk:reward favors patience" / "wait for confirmation" → 0 (neutral, hold)
  - "ETF inflows record" / "Saylor buying" / "FOMO incoming" → BULLISH (+0.4..+0.7)
  - "Bear flag" / "Distribution" / "Liquidation cascade" → BEARISH (-0.6..-0.8)

Ответ — СТРОГО JSON list, по одному score на пост в том же порядке:
[{{"id": "<post_id>", "score": -1..+1, "reasoning": "<1 предложение почему>"}}, ...]

Посты для оценки:
{posts_block}"""


def rescore_with_claude(
    scored_tweets: list,        # list[ScoredTweet]
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, dict]:
    """Rescore tier-1/tier-2 posts. Returns mapping tweet_id → {score, reasoning}.

    Posts NOT rescored stay with their VADER score.
    """
    if not scored_tweets:
        return {}

    # Pick top posts by tier weight × absolute current sentiment × engagement
    eligible = [
        st for st in scored_tweets
        if st.tweet.tier in ("tier_1", "tier_2")
    ]
    if not eligible:
        return {}

    # Sort by impact (top likes+retweets) and cap by batch_size
    eligible.sort(
        key=lambda st: -(st.tweet.metrics.get("likes", 0)
                          + st.tweet.metrics.get("retweets", 0) * 5),
    )
    batch = eligible[:batch_size]

    # Build the prompt block
    lines = []
    for st in batch:
        lines.append(
            f"[{st.tweet.id}] @{st.tweet.author_handle} "
            f"({st.tweet.tier}): {st.tweet.text[:280]}"
        )
    posts_block = "\n\n".join(lines)
    prompt = CLAUDE_PROMPT.format(posts_block=posts_block)

    # Call Claude (lazy import — keeps tests fast)
    try:
        from wave_ai import _get_api_key, DEFAULT_MODEL, DEFAULT_TIMEOUT_S
        from claude_call_log import log_call
        import anthropic
    except ImportError:
        return {}

    api_key = _get_api_key()
    if api_key is None:
        return {}

    client = anthropic.Anthropic(api_key=api_key, timeout=DEFAULT_TIMEOUT_S)
    try:
        msg = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        # Log + fail gracefully — rescoring is opportunistic
        try:
            log_call(ticker="multi", call_type="rescore",
                      model=DEFAULT_MODEL, prompt=prompt,
                      error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        return {}

    raw = ""
    for block in msg.content:
        if hasattr(block, "text"):
            raw += block.text
    raw = raw.strip()

    # Parse JSON array
    try:
        start = raw.find("[")
        end = raw.rfind("]")
        parsed = json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        parsed = []

    out: dict[str, dict] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        post_id = str(item.get("id", ""))
        score = item.get("score")
        if post_id and isinstance(score, (int, float)):
            out[post_id] = {
                "score": float(score),
                "reasoning": str(item.get("reasoning", "")),
            }

    # Audit log this rescore call
    try:
        tokens_in  = getattr(getattr(msg, "usage", None), "input_tokens", 0) or 0
        tokens_out = getattr(getattr(msg, "usage", None), "output_tokens", 0) or 0
        log_call(ticker="multi", call_type="rescore",
                  model=DEFAULT_MODEL, prompt=prompt, raw_response=raw,
                  parsed={"n_rescored": len(out), "post_ids": list(out.keys())},
                  tokens_in=tokens_in, tokens_out=tokens_out,
                  duration_s=0.0)
    except Exception:
        pass

    return out


def apply_rescores_to_scored(scored_tweets: list,
                              rescores: dict[str, dict]) -> list:
    """Mutate ScoredTweet.sentiment for posts present in rescores."""
    for st in scored_tweets:
        rs = rescores.get(st.tweet.id)
        if rs is None:
            continue
        new_score = float(rs["score"])
        st.sentiment = new_score
        st.label = (
            "bullish" if new_score >= 0.15
            else "bearish" if new_score <= -0.15
            else "neutral"
        )
    return scored_tweets
