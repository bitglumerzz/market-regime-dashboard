"""Optional Claude-API integration for a 'second opinion' on wave counts.

Caveats:
* Uses Anthropic's Python SDK (not MCP).
* The user supplies their own API key via `.env` (preferred — read by Docker
  Compose) or via `.streamlit/secrets.toml`. The check below tries both.
* Each query is a small Russian-language prompt; the response is structured
  JSON with reasoning in Russian + concrete trade levels.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from elliott import WaveCandidate
from forecast import WaveForecast
from zigzag import Swing

# Optional knowledge base — gracefully degrades if PyYAML or the folder
# isn't present (e.g. fresh install before the bind-mount is created).
try:
    from wave_knowledge import build_knowledge_prompt, slugs_for_candidates
    _KNOWLEDGE_AVAILABLE = True
except Exception:                              # pragma: no cover
    _KNOWLEDGE_AVAILABLE = False

    def build_knowledge_prompt(*args, **kwargs) -> str:  # type: ignore[no-redef]
        return ""

    def slugs_for_candidates(*args, **kwargs) -> set[str]:  # type: ignore[no-redef]
        return set()


def _system_prompt_with_knowledge(focus_candidates=None) -> str | None:
    """Compile the knowledge-base block as Claude's `system` parameter.

    Returns None when the base is empty / unavailable so the SDK call falls
    back to the default system prompt.
    """
    if not _KNOWLEDGE_AVAILABLE:
        return None
    try:
        focus = (slugs_for_candidates(focus_candidates)
                 if focus_candidates else None)
        block = build_knowledge_prompt(focus_patterns=focus or None,
                                        max_chars=22000)
        return block or None
    except Exception:
        return None


DEFAULT_MODEL: str = "claude-opus-4-7"
DEFAULT_TIMEOUT_S: float = 120.0      # generous; 15m series can be slow to think over
MAX_RETRIES: int = 3
RETRY_BACKOFF_S: tuple[float, ...] = (2.0, 6.0, 15.0)
# Truncate big swing lists — Claude doesn't need 300 swings to label the
# tail; the last 25 is plenty to see the active structure.
MAX_SWINGS_IN_PROMPT: int = 25


class ClaudeSpendLimitError(RuntimeError):
    """Raised when the user's Anthropic API spend limit is exhausted.

    Carries the regain-access date if Anthropic provided it, so the UI can
    display a friendly Russian message instead of a raw JSON error.
    """
    def __init__(self, message: str, regain_at: str | None = None):
        super().__init__(message)
        self.regain_at = regain_at


def _extract_json_from_response(raw: str) -> dict:
    """Robustly pull a JSON object out of Claude's response.

    Claude sometimes:
      * Wraps JSON in ```json ... ``` fences
      * Adds prose before or after the JSON
      * Returns valid JSON but with trailing commentary that confuses
        a naive first-{/last-} scan when there's another `{}` in the prose
      * Stuffs a multi-line JSON-looking blob inside the ``summary`` field
        when over-eager to be helpful

    This function tries strategies in order of decreasing reliability and
    returns the first one that parses to a dict. Empty dict on total failure.
    """
    if not raw:
        return {}
    text = raw.strip()

    # Strategy 1 — markdown fence ```json ... ``` (Claude's most common wrap)
    import re
    fence = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Strategy 2 — greedy first-{ to last-} (legacy behavior)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start: end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Strategy 3 — balanced-brace scanner starting at first {
    # Handles "JSON then prose then another { but unclosed" scenarios
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\" and in_str:
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start: i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break

    # Strategy 4 — fallback: try parsing the whole text
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Strategy 5 (Phase 33) — auto-repair truncated JSON. When Claude's
    # response hits max_tokens mid-JSON, the result is something like:
    #   { "bias": "long", ..., "scenarios": [ {"name": "A", ... }, {"name":
    # The naive parsers above all fail because closing }] are missing. Try
    # to:
    #   (a) close an open string literal,
    #   (b) trim trailing comma + partial last element,
    #   (c) append the right number of `]` and `}` based on brace/bracket
    #       balance tracked with string-aware scanning.
    # The repaired JSON gives us the FIELDS THAT WERE COMPLETE before
    # truncation — which is way better than empty dict + summary fallback.
    if start >= 0:
        body = text[start:]
        # Scan with string-aware brace/bracket counting; remember the last
        # safe truncation point (after a complete value, before any partial
        # next-key/value).
        depth_curly = 0
        depth_square = 0
        in_str = False
        esc = False
        last_safe = -1  # index of last position right after a complete value
        for i, c in enumerate(body):
            if esc:
                esc = False
                continue
            if c == "\\" and in_str:
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth_curly += 1
            elif c == "}":
                depth_curly -= 1
                if depth_curly >= 0 and depth_square == 0:
                    last_safe = i + 1
            elif c == "[":
                depth_square += 1
            elif c == "]":
                depth_square -= 1
                if depth_square >= 0 and depth_curly == 1:
                    last_safe = i + 1
            elif c == "," and not in_str and depth_curly <= 1 \
                    and depth_square <= 1:
                # Comma after a complete top-level value — safe truncation
                last_safe = i + 1

        # If string is still open at end, close it
        repaired = body
        if in_str:
            repaired = repaired + '"'

        # Build candidates to try, in order of "most preservation":
        candidates = []

        # (a) Close everything as-is
        close = "]" * max(0, depth_square) + "}" * max(0, depth_curly)
        candidates.append(repaired + close)

        # (b) Trim to last_safe point and close balanced
        if last_safe > 0:
            trimmed = body[:last_safe].rstrip().rstrip(",")
            # Recount depth for trimmed
            d_c = d_s = 0
            ins = False
            es = False
            for c in trimmed:
                if es:
                    es = False; continue
                if c == "\\" and ins:
                    es = True; continue
                if c == '"':
                    ins = not ins; continue
                if ins:
                    continue
                if c == "{": d_c += 1
                elif c == "}": d_c -= 1
                elif c == "[": d_s += 1
                elif c == "]": d_s -= 1
            cls = "]" * max(0, d_s) + "}" * max(0, d_c)
            candidates.append(trimmed + cls)

        # Try each candidate
        for cand in candidates:
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    # Phase 44 — flag that we recovered from truncation
                    # so caller can warn the user. Use a sentinel key
                    # starting with "_" so it won't collide with real fields.
                    obj["_was_truncated"] = True
                    return obj
            except json.JSONDecodeError:
                continue

    return {}


def _humanize_anthropic_error(exc: Exception) -> Exception:
    """Convert raw Anthropic errors into friendlier RuntimeErrors when possible.

    Returns either the original exception unchanged, or a new exception with a
    Russian-friendly message.
    """
    text = str(exc)
    # Spend / usage limit (the most common opaque-looking error users hit)
    if ("usage limit" in text.lower()
            or "spending limit" in text.lower()
            or "regain access" in text.lower()):
        # Try to extract the regain date (stop at the sentence-ending period,
        # quote, or JSON close-brace so we don't capture the API request id).
        regain_at = None
        import re
        m = re.search(r"regain access on ([\w\s\-:]+?)(?=[.'}])", text)
        if m:
            regain_at = m.group(1).strip()
        msg = (
            "Лимит расходов на Anthropic API исчерпан. "
            + (f"Доступ восстановится: {regain_at}. " if regain_at else "")
            + "Подними лимит в console.anthropic.com → Settings → Limits, "
              "либо подожди следующего цикла."
        )
        return ClaudeSpendLimitError(msg, regain_at=regain_at)
    # Rate limit (429-like)
    if "rate_limit" in text.lower() or "rate limit" in text.lower():
        return RuntimeError(
            "Превышен rate-limit Anthropic. Подожди ~1 минуту и попробуй "
            "снова. Если повторяется — снизь частоту запросов."
        )
    # Auth issues
    if "authentication" in text.lower() or "invalid api" in text.lower():
        return RuntimeError(
            "Anthropic не принял API-ключ. Проверь ANTHROPIC_API_KEY в .env."
        )
    return exc


@dataclass
class AIHolisticOpinion:
    """Top-down opinion across all timeframes at once.

    The single most useful artifact for live trading — instead of asking
    Claude separately about each TF and reconciling its 4 answers, we send
    everything in one prompt and get one coherent plan.
    """
    bias: str                  # 'long' | 'short' | 'neutral'
    setup_quality: str         # 'high' | 'medium' | 'low' | 'none'
    summary: str               # one-paragraph Russian recap
    action: str                # 'long' | 'short' | 'wait' | 'close'
    entry: float | None
    stop_loss: float | None
    target: float | None
    risk_reward: float | None
    invalidation_explained: str
    per_tf_notes: dict[str, str]   # {"1d": "...", "4h": "...", ...}
    raw_text: str
    # Phase 4.4 — Twitter sentiment integration
    sentiment_alignment: str = "n/a"   # 'aligned' / 'divergent' / 'n/a'
    # Phase 6.2 — TradingView ideas integration
    tv_alignment: str = "n/a"          # 'aligned' / 'divergent' / 'n/a'
    # Phase 13 — Chain-of-thought fields from the enriched prompt
    structure_position: str = ""       # where exactly we are inside the top-TF structure
    rules_check: str = ""              # which strict rules pass/fail
    alternation_check: str = ""        # alternation prediction
    fibonacci_check: str = ""          # which fib ratios hit/miss
    alternative_interpretation: str = ""  # alt labelling if any
    confidence_factors: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    # Phase 15 — structured chart annotations (pivot/fib_level/zone/arrow/
    # wave_label/note). Each entry has 'type', 'tf', and type-specific
    # coordinate fields. Rendered onto the corresponding TF chart by
    # pages/4_Elliott_Waves._overlay_claude_opinion.
    chart_annotations: list[dict] = field(default_factory=list)
    # Phase 24 — additional 5m scalp plan aligned with the 1d bias. Lets
    # the trader take a small intraday position even when the swing plan
    # is "wait" (because we're between the ideal swing entry points).
    # Has its own direction/entry/stop/targets/rationale.
    scalp_plan: dict = field(default_factory=dict)
    # Phase 26 — super-cycle context + monitoring matrix. Inspired by
    # the TV MCP BTC analysis format: explicit higher-degree wave count
    # ("where we are in the supercycle") and 3-5 monitoring levels with
    # if-then actions for when there's no immediate swing setup.
    higher_degree_context: dict = field(default_factory=dict)
    monitoring_levels: list[dict] = field(default_factory=list)
    # Phase 28 — entry alternatives (tight vs wider stop), target pyramid
    # with per-stop R/R, cross-correlation notes. Inspired by the TV MCP
    # HYPE truncated-W5 analysis format.
    entry_alternatives: list[dict] = field(default_factory=list)
    target_pyramid: list[dict] = field(default_factory=list)
    cross_correlation_notes: str = ""
    # Phase 29 — multi-scenario plan with probability weights, waiting
    # conditions checklist, honest invalidation tracking. Inspired by
    # the TV MCP BTC 5m followup analysis format.
    scenarios: list[dict] = field(default_factory=list)
    waiting_conditions: list[dict] = field(default_factory=list)
    invalidation_of_previous: str = ""
    # Phase 30 — cross-asset comparison table for correlated instruments
    # (BTC vs ETH, leader vs follower, etc.). Inspired by the TV MCP ETH
    # 5m followup analysis where W3-shortest violation on ETH revealed
    # corrective Y-wave (not impulsive), while BTC kept a valid 5-wave bear.
    # Each row should be a dict with keys: "parameter", "primary", "secondary"
    # (or similar self-describing keys), plus an optional "headers" list for
    # the column titles. Example:
    #   [{"parameter": "15m structure", "primary": "valid 5-wave bear",
    #     "secondary": "Y-wave corrective (W3 shortest)"}, ...]
    cross_asset_comparison: list[dict] = field(default_factory=list)
    leader_follower_note: str = ""
    # Phase 31 — concise, telegraph-style action plan with 4-6 if-then steps.
    # This is the trader's main actionable output, separate from the narrative
    # fields. Each step is a dict {"step": int, "action": str}.
    action_plan: list[dict] = field(default_factory=list)
    # Phase 34 — MCP-style full-report fields. Reference:
    # tv_mcp_btc_3tf_full_report_w2_golden_zone_setup. These bring the
    # response up to TV MCP visual quality with context header, per-TF
    # ASCII wave trace, Fib tables with Status, impulse decomposition,
    # wave projections, comparison with previous analysis, time-bound
    # monitor checklist, and a final "GOLDEN ENTRY" actionable summary.
    period_context: dict = field(default_factory=dict)
    tf_analysis: dict = field(default_factory=dict)
    verdict_per_tf: dict = field(default_factory=dict)
    confluence_factors: list = field(default_factory=list)
    comparison_with_previous: dict = field(default_factory=dict)
    # Phase 41 — versioned diff vs previous holistic. Inspired by MCP TV
    # v2→v3 updates: "buy zone сместилась с $43.20 (v2) на $43.80 (v3)".
    entry_zone_shift: str = ""
    # Phase 42 — three-tier entry variants (Aggressive / Confirmed / Breakout)
    # from MCP TV v4 format. Different aggressiveness, different R/R.
    # Example: aggressive=market entry $45.00 / confirmed=15m close > $45.20
    # / breakout=1h close > $46.30 retest.
    entry_variants: list = field(default_factory=list)
    # Phase 44 — flag set when tolerant parser had to repair truncated JSON.
    # UI displays a warning so the user knows to potentially re-run.
    was_truncated: bool = False
    # Phase 47 — three MCP v8 features:
    # 1) structural events timeline (Breakout/Retest/Bounce/Current с
    #    timestamps и ценами).
    # 2) 3-layer invalidation (soft / hard / macro) — каждый со своим
    #    уровнем и действием.
    # 3) time-bound gut probability ("next 24-48h: 65% continuation /
    #    25% sideways / 10% pullback").
    structural_events_timeline: list = field(default_factory=list)
    invalidation_layers: dict = field(default_factory=dict)
    gut_probability_next_window: dict = field(default_factory=dict)
    monitor_next_hours: dict = field(default_factory=dict)
    critical_level: str = ""
    golden_entry_zone: str = ""


@dataclass
class AIWaveOpinion:
    """Structured response from Claude."""
    agrees_with_top: bool
    confidence: str            # 'high' | 'medium' | 'low'
    preferred_pattern: str
    action: str                # 'long' | 'short' | 'wait' | 'close'
    reasoning: str
    entry: float | None
    stop_loss: float | None
    target: float | None
    # Forecast-related: did Claude agree with our rule-engine forecast?
    agrees_with_forecast: bool | None = None
    forecast_critique: str = ""
    raw_text: str = ""


def _get_api_key() -> str | None:
    """Pull Anthropic API key from env var or Streamlit secrets."""
    import os
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets["anthropic"]["api_key"]
    except Exception:
        return None


def is_available() -> bool:
    if _get_api_key() is None:
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def availability_message() -> str:
    if _get_api_key() is None:
        return (
            "Нет ключа API. Создай файл `.env` в корне проекта со строкой "
            "`ANTHROPIC_API_KEY=sk-ant-...` и перезапусти контейнер."
        )
    try:
        import anthropic  # noqa: F401
        return f"Подключено — используется модель {DEFAULT_MODEL}"
    except ImportError:
        return "Пакет anthropic не установлен. Добавь его в requirements.txt и пересобери."


def _format_swings(swings: list[Swing], n_last: int = MAX_SWINGS_IN_PROMPT) -> str:
    recent = swings[-n_last:]
    lines = []
    for i, s in enumerate(recent):
        suffix = " (provisional)" if s.provisional else ""
        lines.append(
            f"  {i}. {s.index.strftime('%Y-%m-%d %H:%M')}  "
            f"{s.kind:5s} @ {s.price:.4f}{suffix}"
        )
    return "\n".join(lines)


def _format_candidates(cands: list[WaveCandidate]) -> str:
    out = []
    for c in cands:
        out.append(
            f"  - {c.pattern.replace('_', ' ')} · score {c.score:.0f} "
            f"· {len(c.rule_violations)} violations "
            f"· {c.current_position}"
        )
    return "\n".join(out)


def _format_forecast(fc: WaveForecast | None) -> str:
    """Render a forecast as compact context for the LLM prompt."""
    if fc is None or not fc.is_actionable:
        return "  (детектор не даёт прогноз — структура неясная)"
    lines = [
        f"  Паттерн:       {fc.next_pattern}",
        f"  Направление:   {fc.direction}",
        f"  Уверенность:   {fc.confidence}",
    ]
    if fc.invalidation_level is not None:
        lines.append(f"  Отмена сценария: {fc.invalidation_level:.4f}")
    if fc.expected_duration_bars is not None:
        lo, hi = fc.expected_duration_bars
        lines.append(f"  Длительность:  {lo}-{hi} баров")
    if fc.targets:
        lines.append("  Ценовые цели:")
        for t in fc.targets:
            emp = (f" [эмпирическая P={t.empirical_p:.0%}]"
                    if t.empirical_p is not None else "")
            lines.append(f"    {t.fib_label:>8s}  @  {t.price:.4f}"
                         f"  ({t.label}){emp}")
    return "\n".join(lines)


def build_prompt(
    ticker: str, timeframe: str,
    swings: list[Swing], candidates: list[WaveCandidate],
    forecast: WaveForecast | None = None,
) -> str:
    """Build a Russian-language prompt asking for a structured opinion.

    If `forecast` is given, the rule-engine's prediction is included so
    Claude can validate, refine, or refute it.
    """
    last_price = swings[-1].price if swings else 0.0
    forecast_block = (
        f"\n\nПрогноз rule-engine на следующее движение:\n"
        f"{_format_forecast(forecast)}"
        if forecast is not None else ""
    )
    return f"""Ты — опытный аналитик волн Эллиотта. Делаешь обзор разметки от
rule-based детектора по тикеру **{ticker}** на таймфрейме **{timeframe}**.
Отвечай конкретно, кратко и скептически — на русском языке.

Текущая последняя цена: {last_price:.4f}

Последние swing-точки ZigZag (от старых к новым):
{_format_swings(swings)}

Ранжированные гипотезы детектора (топ = лучшая):
{_format_candidates(candidates)}{forecast_block}

Твоя задача:
  1. Согласен ли ты с топ-гипотезой? Если нет — какая твоя?
  2. Если есть прогноз — валидируй его. Реалистичны ли цели? Хороший ли уровень отмены?
  3. Дай рекомендацию что делать СЕЙЧАС (с конкретными числами).

Ответ — СТРОГО JSON с этими ключами (русский язык в строковых полях):
{{
  "agrees_with_top": true | false,
  "confidence": "high" | "medium" | "low",
  "preferred_pattern": "<краткая метка по-русски, напр. 'импульс вверх — волна 3 из 5' или 'A-B-C коррекция вниз завершена'>",
  "agrees_with_forecast": true | false | null,
  "forecast_critique": "<2-3 предложения о прогнозе, или null если прогноза нет>",
  "action": "long" | "short" | "wait" | "close",
  "reasoning": "<2-4 предложения на русском с обоснованием. Ссылайся на правила R1/R2/R3 или коэффициенты Фибоначчи если уместно>",
  "entry": <число | null — предлагаемая цена входа в той же валюте что цены выше; null если action=wait>,
  "stop_loss": <число | null — стоп-лосс; null для wait/close>,
  "target": <число | null — цель (тейк-профит); null для wait/close>
}}

Когда action=long/short — обязательно укажи entry, stop_loss и target числами.
Когда action=wait/close — ставь эти поля в null.

ТОЛЬКО JSON в ответе, без префиксов и комментариев."""


def _swings_signature(swings: list[Swing], forecast: WaveForecast | None) -> str:
    """Hash the inputs so identical requests are cached."""
    sig_parts = [
        f"{s.index.isoformat()}|{s.price:.6f}|{s.kind}"
        for s in swings[-MAX_SWINGS_IN_PROMPT:]
    ]
    if forecast is not None:
        sig_parts.append(f"FC:{forecast.next_pattern}:{forecast.direction}")
        for tg in forecast.targets:
            sig_parts.append(f"T:{tg.fib_label}:{tg.price:.6f}")
    h = hashlib.sha256("\n".join(sig_parts).encode("utf-8")).hexdigest()
    return h[:16]


# Cache lives in Streamlit session_state so it survives reruns but not server
# restarts. Keys: (ticker, timeframe, signature). Values: AIWaveOpinion.
def _cache_get(key: tuple) -> "AIWaveOpinion | None":
    cache = st.session_state.get("_claude_cache", {})
    return cache.get(key)


def _cache_put(key: tuple, value: "AIWaveOpinion") -> None:
    cache = st.session_state.setdefault("_claude_cache", {})
    cache[key] = value


def _num_or_none(v) -> float | None:
    """Coerce a JSON value to float, or None if missing / invalid / NaN."""
    if v is None:
        return None
    try:
        x = float(v)
        return x if x == x else None  # NaN guard
    except (TypeError, ValueError):
        return None


def _safe_call_log(**kwargs) -> None:
    """Log a Claude call without ever blocking the UI."""
    try:
        from claude_call_log import log_call
        log_call(**kwargs)
    except Exception:
        pass


def ask_claude(
    ticker: str, timeframe: str,
    swings: list[Swing], candidates: list[WaveCandidate],
    forecast: WaveForecast | None = None,
    model: str = DEFAULT_MODEL,
    use_cache: bool = True,
) -> AIWaveOpinion:
    api_key = _get_api_key()
    if api_key is None:
        raise RuntimeError(
            "API-ключ Anthropic не настроен. Добавь ANTHROPIC_API_KEY в .env."
        )
    import anthropic

    # Cache short-circuit. Log cache hits separately so cost accounting is accurate.
    cache_key = (ticker, timeframe, _swings_signature(swings, forecast))
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            _safe_call_log(
                ticker=ticker, call_type="per_tf", timeframe=timeframe,
                model=model, prompt="", raw_response=cached.raw_text,
                parsed={"agrees_with_top": cached.agrees_with_top,
                         "action": cached.action, "entry": cached.entry,
                         "stop_loss": cached.stop_loss, "target": cached.target},
                tokens_in=0, tokens_out=0, duration_s=0.0, cached=True,
            )
            return cached

    client = anthropic.Anthropic(api_key=api_key, timeout=DEFAULT_TIMEOUT_S)
    prompt = build_prompt(ticker, timeframe, swings, candidates, forecast=forecast)
    system_block = _system_prompt_with_knowledge(focus_candidates=candidates)

    call_started = time.monotonic()

    # Retry with backoff on transient timeouts.
    last_error: Exception | None = None
    msg = None
    for attempt in range(MAX_RETRIES):
        try:
            create_kwargs: dict = {
                "model": model,
                "max_tokens": 900,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_block:
                create_kwargs["system"] = system_block
            msg = client.messages.create(**create_kwargs)
            break
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_S[attempt])
            continue
        except anthropic.APIStatusError as exc:
            # Phase 45 — retry on transient server errors (529 overloaded,
            # 503 unavailable, 502 bad gateway, 504 gateway timeout, 500
            # internal). 4xx surfaces immediately to caller.
            status = getattr(exc, "status_code", None)
            if status in (500, 502, 503, 504, 529) and attempt < MAX_RETRIES - 1:
                last_error = exc
                wait = RETRY_BACKOFF_S[attempt] * 2
                time.sleep(wait)
                continue
            _safe_call_log(ticker=ticker, call_type="per_tf",
                            timeframe=timeframe, model=model, prompt=prompt,
                            error=f"{type(exc).__name__}: {exc}",
                            duration_s=time.monotonic() - call_started)
            raise _humanize_anthropic_error(exc) from exc
        except anthropic.BadRequestError as exc:
            _safe_call_log(ticker=ticker, call_type="per_tf",
                            timeframe=timeframe, model=model, prompt=prompt,
                            error=f"{type(exc).__name__}: {exc}",
                            duration_s=time.monotonic() - call_started)
            raise _humanize_anthropic_error(exc) from exc
    if msg is None:
        _safe_call_log(ticker=ticker, call_type="per_tf", timeframe=timeframe,
                        model=model, prompt=prompt,
                        error=f"timeout after {MAX_RETRIES} retries: {last_error}",
                        duration_s=time.monotonic() - call_started)
        raise RuntimeError(
            f"Claude недоступен после {MAX_RETRIES} попыток: "
            f"{type(last_error).__name__}: {last_error}"
        )

    raw = ""
    for block in msg.content:
        if hasattr(block, "text"):
            raw += block.text
    raw = raw.strip()

    # Capture token usage for cost accounting.
    tokens_in  = getattr(getattr(msg, "usage", None), "input_tokens", 0) or 0
    tokens_out = getattr(getattr(msg, "usage", None), "output_tokens", 0) or 0
    call_duration = time.monotonic() - call_started

    parsed: dict = _extract_json_from_response(raw)

    agrees_fc = parsed.get("agrees_with_forecast")
    if agrees_fc is not None and not isinstance(agrees_fc, bool):
        agrees_fc = None

    opinion = AIWaveOpinion(
        agrees_with_top=bool(parsed.get("agrees_with_top", False)),
        confidence=str(parsed.get("confidence", "low")),
        preferred_pattern=str(parsed.get("preferred_pattern", "")),
        action=str(parsed.get("action", "wait")).lower(),
        reasoning=str(parsed.get("reasoning", raw if not parsed else "")),
        entry=_num_or_none(parsed.get("entry")),
        stop_loss=_num_or_none(parsed.get("stop_loss")),
        target=_num_or_none(parsed.get("target")),
        agrees_with_forecast=agrees_fc,
        forecast_critique=str(parsed.get("forecast_critique") or ""),
        raw_text=raw,
    )
    if use_cache:
        _cache_put(cache_key, opinion)

    # Audit log: prompt + full response + token usage + cost
    _safe_call_log(
        ticker=ticker, call_type="per_tf", timeframe=timeframe,
        model=model, prompt=prompt, raw_response=raw,
        parsed=parsed, tokens_in=tokens_in, tokens_out=tokens_out,
        duration_s=call_duration, cached=False,
    )
    return opinion


# ---------------------------------------------------------------------------
# Holistic — all 4 TFs in one shot
# ---------------------------------------------------------------------------
def _format_tv_ideas(ticker: str) -> str:
    """Render the TradingView ideas block for the holistic prompt.

    Empty string if no index or no data for this ticker.
    """
    try:
        from tv_ideas_aggregate import load_tv_index
        idx = load_tv_index()
    except Exception:
        return ""
    if not idx or "tickers" not in idx:
        return ""
    t_data = idx["tickers"].get(ticker)
    if not t_data or t_data.get("n_ideas_total", 0) == 0:
        return ""

    lines = [
        f"## TradingView Ideas (обновлено: {idx.get('updated_at', '?')[:16]}):",
        f"Всего идей про {ticker} за 24ч: {t_data['n_ideas_total']} "
        f"({t_data['n_long']} LONG, {t_data['n_short']} SHORT)",
        f"Взвешенный bias: {t_data['weighted_bias']:+.2f} "
        f"→ **{t_data['label'].upper()}**",
    ]
    by_tier = t_data.get("by_tier", {})
    for tier_label in ("tier_1", "tier_2", "tier_3"):
        d = by_tier.get(tier_label)
        if not d:
            continue
        tier_human = {
            "tier_1": "Tier-1 топ-авторы (>50k followers)",
            "tier_2": "Tier-2 (5k-50k followers)",
            "tier_3": "Tier-3 crowd (<5k followers)",
        }[tier_label]
        lines.append(
            f"  - {tier_human}: {d['n_ideas']} идей "
            f"({d['n_long']}L/{d['n_short']}S), bias {d['bias']:+.2f} "
            f"({d['label']})"
        )
        for ti in d.get("top_ideas", [])[:2]:
            lines.append(
                f"      @{ti['author']} ({ti['followers']}f) "
                f"{ti['direction'].upper()} {ti['timeframe']}: "
                f"\"{ti['title'][:120]}\""
            )
    if t_data.get("divergence"):
        lines.append(f"  ⚡ {t_data['divergence']}")
    return "\n".join(lines)


def _format_twitter_sentiment(ticker: str) -> str:
    """Render the Twitter sentiment block for the holistic prompt.

    Returns an empty string if no sentiment index exists or no data for ticker.
    """
    try:
        from twitter_sentiment import load_sentiment_index
        idx = load_sentiment_index()
    except Exception:
        return ""
    if not idx or "tickers" not in idx:
        return ""
    t_data = idx["tickers"].get(ticker)
    if not t_data:
        return ""

    lines = [
        f"## Twitter sentiment (обновлено: {idx.get('updated_at', '?')[:16]}):",
        f"Всего постов про {ticker} за 24ч: {t_data.get('n_tweets_total', 0)}",
        f"Взвешенный sentiment: {t_data.get('weighted_score', 0):+.2f} "
        f"→ **{t_data.get('weighted_label', 'neutral').upper()}**",
    ]
    by_tier = t_data.get("by_tier", {})
    for tier_label in ("tier_1", "tier_2", "tier_3"):
        d = by_tier.get(tier_label)
        if not d:
            continue
        tier_human = {
            "tier_1": "Tier-1 market movers",
            "tier_2": "Tier-2 quality analysts",
            "tier_3": "Tier-3 crowd/mainstream",
        }[tier_label]
        lines.append(
            f"  - {tier_human}: {d['n_tweets']} постов, "
            f"среднее {d['mean_sentiment']:+.2f} ({d['label']})"
        )
        for tp in d.get("top_posts", [])[:2]:
            lines.append(
                f"      @{tp['handle']}: "
                f"\"{tp['text'][:120]}\" ({tp['sentiment']:+.2f})"
            )
    if t_data.get("divergence"):
        lines.append(f"  ⚡ {t_data['divergence']}")
    return "\n".join(lines)


def build_holistic_prompt(
    ticker: str,
    tf_blocks: list[dict],     # [{name, swings, candidates, forecast}]
    prev_holistic: dict | None = None,  # Phase 41 — previous opinion for versioned diff
    track_record: list[dict] | None = None,  # Phase 43 — verified history
) -> str:
    """Compose one prompt covering all timeframes for a unified trade plan."""
    last_price = 0.0
    for tfb in tf_blocks:
        if tfb["swings"]:
            last_price = tfb["swings"][-1].price
            break

    sections = []
    for tfb in tf_blocks:
        sec = (
            f"## TF: {tfb['name']}\n"
            f"Последние свинги:\n{_format_swings(tfb['swings'])}\n"
            f"Гипотезы детектора:\n{_format_candidates(tfb['candidates'])}\n"
            f"Прогноз rule-engine:\n{_format_forecast(tfb['forecast'])}"
        )
        sections.append(sec)
    sections_str = "\n\n".join(sections)

    # Phase 4.4 — Twitter sentiment context (only if data is available).
    twitter_block = _format_twitter_sentiment(ticker)
    twitter_section = ("\n\n" + twitter_block) if twitter_block else ""

    # Phase 6.2 — TradingView Ideas context (explicit long/short votes).
    tv_block = _format_tv_ideas(ticker)
    tv_section = ("\n\n" + tv_block) if tv_block else ""

    # Phase 41 — previous holistic context for versioned v1→v2 diff updates.
    # When prev_holistic is provided (loaded from holistic_store), inject a
    # PREVIOUS ANALYSIS section so Claude can produce a meaningful
    # comparison_with_previous + entry_zone_shift diff like MCP TV does.
    prev_section = ""
    if isinstance(prev_holistic, dict) and prev_holistic.get("bias"):
        prev_lines = [
            "\n\n## PREVIOUS ANALYSIS (для сравнения, заполни comparison_with_previous)",
            f"- prev_bias: {prev_holistic.get('bias', 'n/a')}",
            f"- prev_action: {prev_holistic.get('action', 'n/a')}",
            f"- prev_setup_quality: {prev_holistic.get('setup_quality', 'n/a')}",
            f"- prev_entry: {prev_holistic.get('entry', 'n/a')}",
            f"- prev_stop_loss: {prev_holistic.get('stop_loss', 'n/a')}",
            f"- prev_target: {prev_holistic.get('target', 'n/a')}",
            f"- prev_summary: {(prev_holistic.get('summary', '') or '')[:300]}",
            f"- prev_critical_level: {prev_holistic.get('critical_level', '')}",
            f"- prev_golden_entry_zone: {prev_holistic.get('golden_entry_zone', '')}",
        ]
        # If prev had structure_position, include its core
        if prev_holistic.get("structure_position"):
            prev_lines.append(
                f"- prev_structure_position: {prev_holistic['structure_position'][:280]}"
            )
        prev_lines.append(
            "\n📌 ИСПОЛЬЗУЙ это для сравнения: что изменилось с предыдущего "
            "анализа? Заполни comparison_with_previous (что было / стало / "
            "ключевой сдвиг) и entry_zone_shift (сместилась ли buy/sell zone "
            "и куда). Если bias/quality деградировал — открыто признай."
        )
        prev_section = "\n".join(prev_lines)

    # Phase 43 — verified track record of past predictions. Inject only if
    # we have ≥2 entries (1 previous + need at least one verified to be
    # useful). Inspired by MCP TV v5 which celebrates "v4 prediction hit ✓".
    track_section = ""
    if isinstance(track_record, list) and len(track_record) >= 1:
        try:
            from holistic_history import format_for_prompt
            track_section = format_for_prompt(track_record, max_n=5)
            if track_section:
                track_section += (
                    "\n\n📌 Если последние N predictions систематически "
                    "сбывались (hit ✓) — твоя модель калибрована, повышай "
                    "confidence. Если последние N были stopped/ambiguous — "
                    "понижай confidence и пересмотри подход. Открыто пиши в "
                    "summary что 'предыдущий vN hit/missed'."
                )
        except Exception:
            pass

    return f"""Ты — опытный аналитик волн Эллиотта, делаешь top-down обзор
тикера **{ticker}** одновременно по нескольким таймфреймам. В твоём system
prompt есть REFERENCE MATERIAL с карточками паттернов, trading rules
(модель 1-2 setup, какие волны торгуем), правилами Прехтера (alternation,
channeling, fibonacci targets, extension chaining), и реальными примерами
сделок Павелко (silver ×4, USDCAD triple three). ИСПОЛЬЗУЙ их явно — если
ссылаешься на правило, упоминай источник в скобках, например «(impulse_up.
rules_strict)» или «(Прехтер: channeling)».

Отвечай ИСКЛЮЧИТЕЛЬНО по-русски, конкретно и скептически. Если разметка
детектора кажется некорректной — скажи об этом и предложи альтернативную
интерпретацию.

🚨 ЯЗЫК ОТВЕТА — ВСЕ ПОЛЯ ПО-РУССКИ, ВКЛЮЧАЯ ТЕХНИЧЕСКИЕ ТЕРМИНЫ:
- "impulse up"          → "импульс вверх"
- "impulse down"        → "импульс вниз"
- "regular flat"        → "обычная плоская коррекция"
- "expanded flat"       → "расширенная плоская коррекция"
- "running flat"        → "running плоская" (термин Прехтера, оставь латиницей)
- "contracting triangle" → "сжимающийся треугольник"
- "expanding triangle"  → "расширяющийся треугольник"
- "leading diagonal"    → "ведущая диагональ"
- "ending diagonal"     → "конечная диагональ"
- "zigzag"              → "зигзаг"
- "thrust"              → "выброс" (или "толчок")
- "truncation / truncated 5th" → "укорочение / укороченная 5-я"
- "sweet spot"          → "оптимальная зона"
- "retrace / retracement" → "ретрейс / откат"
- "alternation"         → "правило чередования (Прехтер)"
- "channeling"          → "построение канала"
- "completed"           → "завершён / завершена"
- "in progress"         → "в развитии"
- "wait / SIT OUT"      → "ждём / пропускаем"
- "active setup"        → "активный сетап"
- "invalidation"        → "инвалидация / отмена"

ИСКЛЮЧЕНИЯ — английский разрешён ТОЛЬКО для:
- цен и чисел ($82,517)
- технических аббревиатур (W1, W2, R/R, TP, SL, ATH, ATL, BTC, ETH, FOMC)
- coding-style идентификаторов из reference material (impulse_up, flat_down)
  но ТОЛЬКО внутри backticks `slug` для ссылки на правило, не в narrative.

🚨 ACTION ↔ LEVELS — РАЗЛИЧАЙ IMMEDIATE vs PLANNED:

ДВА ТИПА СЕТАПА:
1. **IMMEDIATE (market entry now)**: action = "long" или "short".
   Цена СЕЙЧАС в зоне входа. Трейдер открывает позицию немедленно.
   entry/stop/target/risk_reward все заполнены конкретными числами.

2. **PLANNED (conditional entry)**: action = "wait".
   Сетап видишь, но цена ЕЩЁ НЕ В ЗОНЕ. Жди триггера (например 'pullback
   к $44.5-45.0 + 5m H-L-H reversal'). entry/stop/target можно заполнить
   как **planned levels** — это значения которые активируются ПРИ триггере.

КРИТЕРИЙ выбора:
- Если текущая цена попадает в entry_zone из action_plan/scenarios → IMMEDIATE.
- Если до entry_zone ещё нужно "доехать" + есть конкретный триггер → PLANNED.

  ❌ ПЛОХО: action="wait", entry=90.50, stop=96.30, target=81.61 БЕЗ описания
     триггера в action_plan (просто levels висят без логики).
  ✅ ХОРОШО (immediate): action="short", entry=90.50, stop=96.30,
     текущая цена 91.0, цена внутри entry_zone 90-91.
  ✅ ХОРОШО (planned): action="wait", entry=45.0, stop=43.0, target=50.5,
     текущая цена 47.0 (выше entry_zone 44.5-45.0), action_plan step 3:
     "Если pullback к 44.5-45.0 + 5m H-L-H → LONG entry 45.0".
  ✅ ХОРОШО (нет идеи): action="wait", entry=null, stop=null, target=null
     (рынок неопределённый, ждём clarity).

ПРАВИЛО: если в summary или action_plan есть "СЕЙЧАС не входить" или
"если pullback к..." или "если break <..." — это PLANNED, action="wait".
Иначе если "СЕЙЧАС вход" — IMMEDIATE.

🚨 НЕ ГЕНЕРИРУЙ DEPRECATED ПОЛЯ (Phase 36 lean schema):
Следующие поля были УБРАНЫ из ответа — НЕ генерируй их, даже если
видишь их в reference examples (knowledge_base/trading_examples.yaml):

  ⚠️ entry_alternatives    → переименовано в entry_variants (Phase 42).
     Используй entry_variants с 3 типами: Aggressive/Confirmed/Breakout.
     НЕ tight/wider как было раньше — новый формат с trigger-полем.
  ❌ target_pyramid        → используй scenarios.targets
  ❌ waiting_conditions    → используй action_plan
  ❌ monitoring_levels     → используй monitor_next_hours
  ❌ chart_annotations     → не нужно сейчас
  ❌ scalp_plan_5m         → объединено в action_plan если важно
  ❌ cross_asset_comparison → не нужно для одиночного актива
  ❌ tf_analysis           → объединено в per_tf_notes + structure_position
  ❌ period_context        → не нужно
  ❌ verdict_per_tf        → объединено в per_tf_notes
  ❌ higher_degree_context → объединено в structure_position
  ❌ confluence_factors    → в summary или в scenarios.action
  ❌ fibonacci_check       → объединено в fibonacci_check (можно оставить
     если короткое — это P2 поле)
  ❌ alternation_check     → объединено в rules_check
  ❌ confidence_factors    → в summary или scenarios
  ❌ risk_factors          → в summary или scenarios
  ❌ leader_follower_note  → не нужно для одиночного актива

Если генерируешь эти поля — впустую тратишь токены и провоцируешь
обрезание ответа. Stick to LEAN schema!

🚨 ANTI-WAIT — ЕСЛИ 2+ ТФ ВЫРОВНЕНЫ В ОДНОМ НАПРАВЛЕНИИ — ДАВАЙ ACTIVE SETUP:
Сейчас слишком часто action="wait" даже когда вся структура говорит SHORT
или LONG. Это плохо. Правило:
  - Если 2+ ТФ (например 1d+4h, или 4h+15m) показывают согласованные
    bias и валидные структуры — ДАВАЙ active SHORT/LONG.
  - Если 1 ТФ требует углубления коррекции, но остальные подтверждают
    direction — дай ACTIVE setup с tight stop у текущей цены, не жди.
  - wait разрешён ТОЛЬКО если:
    (a) <2 ТФ выровнены, или
    (b) близкий valid invalidation level (стоп слишком близко = R/R<1),
        или
    (c) только что произошла переразметка (transition в последние 2 свинга).
  - Если выбираешь wait — обязательно укажи КОНКРЕТНОЕ событие/уровень
    которое переключит план из wait в active (поле "trigger_to_activate"
    внутри golden_entry_zone или action_plan).

Текущая цена: {last_price:.4f}

{sections_str}{twitter_section}{tv_section}{prev_section}{track_section}

## CHAIN OF THOUGHT (внутренне пройди по этому чеклисту перед ответом):

**Step 1 — Валидация разметки на каждом ТФ.**
Для top-кандидата каждого ТФ проверь: нарушает ли он жёсткие правила своей
карточки в reference material? (Например для impulse_up: волна 2 не >100%
волны 1; волна 3 не самая короткая; волна 4 не перекрывает 1.) Если хотя
бы одно правило нарушено — это alternative pattern, а не он.

**Step 2 — Определи ТЕКУЩУЮ ПОЗИЦИЮ в структуре каждого ТФ.**
На каком конкретно этапе мы находимся? «Заканчивается волна 2», «середина
волны 3», «возможно truncation волны 5» — это критично потому что
torгуемые волны (3, 5, C) И не-торгуемые (1, 2, 4) определяются именно по
позиции. Примени personality of waves (Прехтер): если это волна 3 — она
должна быть мощной и широкой; если волна 5 — ищи дивергенции с RSI/MACD.

**Step 3 — Cross-TF проверка.**
Совпадает ли направление младших ТФ с волной 3/5 старшего? Top-down
правило Андреева: макро на 1d должно соответствовать ТФ ниже. Если 1d
говорит «волна 3 вниз» а 4h показывает «импульс вверх» — где
противоречие? Возможно 4h это под-волна i-2 внутри волны 3 старшего.

**Step 4 — Fibonacci & alternation check.**
Проверь Fibonacci ratios: волна 2 ретрейснула 50-61.8% волны 1? Волна 3
≈ 1.618× или 2.618× волны 1? Альтернация: если волна 2 была sharp zigzag,
ожидай sideways flat/triangle в волне 4 (и наоборот). Если ожидания не
совпадают — насторожись.

**Step 5 — Alternative interpretations.**
Какие есть ДРУГИЕ возможные разметки? Например: «детектор видит импульс
вниз, но это может быть expanded flat — тогда волна C ещё не закончилась
и target ниже». Если есть равновероятная альтернатива — это сигнал
снизить confidence сетапа.

**Step 6 — Применимость 1-2 setup.**
Если структура подтверждает что мы ВХОДИМ в волну 3 (или волну C) после
завершения волны 2 (или волны B) — это классический 1-2 setup Павелко.
Если нет — ждём.

**Step 7 — Twitter / TV alignment.**
Если есть Twitter sentiment: tier-1 ЗА структурой = усиление, tier-1
ПРОТИВ = скрытый драйвер. Если есть TV Ideas: top-tier авторы с
конкретными уровнями имеют вес; смотри volume long vs short по тиерам.
Явное противоречие TV-смартмани со структурой = ОЧЕНЬ серьёзный сигнал.

**Step 8 — План сделки или wait — ACTIVE SETUP PRIORITY.**

🚨 КРИТИЧНО: НЕ уходи в общий wait если есть валидный signal!
Pavelko strategy ловит именно следующие конкретные паттерны, и при их
обнаружении ОБЯЗАТЕЛЬНО давай ACTIVE setup (не «wait/наблюдаем»):

  ⚡ **TRUNCATED 5th** — W5 не превысила пик W3 (или W5 не опустилась
    ниже низа W3 в bearish). PRIMARY TRIGGER для разворота. Stop за W3
    (НЕ W5!). Reference: tv_mcp_hype_5m_truncated_w5_active_short_setup.

  ⚡ **Ending diagonal** — клин с 1-4 overlap, особенно в 5-й волне.
    Trigger для мощного reversal. Stop за апексом клина.

  ⚡ **Leading diagonal** в волне 1 — клин с 1-4 overlap в начале
    импульса. Trigger для растянутой волны 3 в основном направлении.

  ⚡ **Running flat/zigzag** — B > A end, C < A end. После таких
    коррекций — взрывное продолжение тренда. Reference:
    tv_mcp_btc_running_zigzag (BTC 1d).

  ⚡ **1-2 setup в sweet spot** — после первой импульсной волны видим
    retracement 50-78% во второй. Entry на завершении 2, stop за
    основанием 1, target ≥ 2R. Это базовый Pavelko trade.

  ⚡ **Truncation + ending diagonal в одной структуре** — combined
    signal, по Прехтеру усиливает violence reversal. Stop тёcный, R/R
    обычно 3-5+.

Если видишь любой из вышеперечисленных — НЕ wait, дай конкретный SHORT
или LONG сетап с entry/stop/target. R/R ≥ 1.5 минимум.

Если none of the triggers active — только тогда wait, и объясни ЧТО
ИМЕННО ждём (например «sweet spot retrace для wave 2 на 4h, цена пока
в 38.2%»).

ВАЖНО для entry/stop/target подачи:
  - Дай ПАРУ entry-альтернатив (tight vs wider) с разными stops в
    поле `entry_alternatives` — трейдер выбирает по risk tolerance.
  - Target pyramid с несколькими TP (1.0× / 1.618× / full retrace) в
    поле `target_pyramid` — где каждый TP имеет per-stop R/R.
  - Cross-correlation в `cross_correlation_notes`: что делают
    BTC/ETH/коррелированные инструменты, как это влияет на текущий
    setup (альт может опередить/запоздать с разворотом).

**Step 9 — Higher-degree wave count (super-cycle context).**
НЕ ограничивайся текущей 5-wave structure на 1d. ВСЕГДА думай где мы в
СТАРШЕЙ структуре:
  - Текущий 1d-импульс — это часть какой более крупной волны? Wave (1) of
    monthly super-cycle? Wave (III)? Wave (B) большой коррекции?
  - Был ли недавний cycle peak? Cycle low? На каких уровнях?
  - Если есть подтверждённая историческая разметка (например LuxAlgo Pine
    label на TV-чарте) — используй её как anchor.
Заполни поле `higher_degree_context` в JSON с тремя строками: cycle_peak,
cycle_low, current_phase.

**Step 10 — Monitoring matrix (если setup сейчас = wait).**
Когда swing-плана нет (wait/no_trade) — ЭТО НЕ ОЗНАЧАЕТ конец работы.
Дай 3-4 КОНКРЕТНЫХ price-уровня с if-then действиями:
  - "Break > {{price}}" → "{{bullish/bearish/neutral scenario}}" → "{{action}}"
  - "Pullback к {{range}}" → "{{scenario}}" → "{{action}}"
  - "Break < {{price}}" → "{{scenario}}" → "{{action}}"
  - "Sideways в {{range}}" → "{{scenario}}" → "{{action}}"
Заполни поле `monitoring_levels` массивом объектов с ключами trigger,
scenario, action. Это даст трейдеру конкретные триггеры для следующего хода вместо
просто "ждём". Образец качества — TV MCP analysis BTC running zigzag
case в reference material.

**Step 11 — 5m SCALP план в направлении 1d bias.**
ДАЖЕ если основной swing-сетап = wait (потому что ждём ретеста или
завершения коррекции на 4h-1d), на 5m часто есть scalp-возможность
В НАПРАВЛЕНИИ 1d-bias:
  - bias=short на 1d → ищи 5m short-сетап на отскоке (волна 2 нового
    5m-импульса вниз; пробой 5m support с retest; коррекция к VWAP/EMA).
  - bias=long на 1d → ищи 5m long-сетап на откате (волна 2 нового 5m-
    импульса вверх).
  - bias=neutral → scalp direction = wait.

Заполни поле scalp_plan_5m с детализацией УРОВНЯ TV MCP (см. reference
material `tv_mcp_btc_5m_bear_impulse_complete_sit_out`):

  1. ВАЛИДИРУЙ 5-WAVE STRUCTURE на 5m против hard rules R0-R3
     (W2 не >100% W1, W3 не самая короткая, W4 не перекрывает W1, net
     direction). Если ОБНАРУЖИЛИ только что завершённый 5-wave импульс —
     указать что это finishing wave (NOT new wave 1) → scalp = wait
     до завершения retrace.

  2. РАЗЛИЧАЙ "wave 1 нового импульса" (можно ловить wave 2 для entry)
     vs "wave 5 завершающегося" (НЕ входить в направлении тренда —
     ждать разворота). Это критичное distinction.

  3. ОТМЕЧАЙ EXTENDED WAVES (W3 или W5 с ratio >1.5× W1) — extended
     wave 5 особенно значим как сигнал dramatic reversal (Прехтер).

  4. FIBONACCI RETRACE TARGETS — дай 4-6 уровней (23.6 / 38.2 / 50 /
     61.8 / 78.6 / 100%) для ожидаемого обратного движения. Эти уровни
     становятся monitoring waypoints для следующих часов.

  5. EXPECTED TIMING — "best entry зона $X ± $Y через N часов" если
     scalp = wait но setup ожидается. Это actionable.

  6. Цели меньше swing (2-3R обычно), время в сделке короткое
     (1-6 часов).

  7. Это ОТДЕЛЬНЫЙ план — не пересекается с swing entry/stop/target.
     Если 5m структура не согласована с 1d (например 1d short, но
     5m показывает мощный bull-импульс с extension wave 3) — scalp =
     wait, объясни конфликт в rationale.

  8. Включи 3-4 MONITORING TRIGGERS в scalp.monitoring_5m (если поле
     поддержано) — конкретные условия с if-then действиями именно для
     5m уровня, дополнительно к macro monitoring_levels.

**Step 12 — CROSS-ASSET COMPARISON (для альтов — ОБЯЗАТЕЛЬНО, для BTC — желательно).**
Когда торгуем альт (ETH, HYPE, SOL, и т.д.) — ВСЕГДА сравни структуру
альта с BTC. Reference: TV MCP ETH 5m analysis показал что у ETH на 15m
W3 был SHORTEST → R2 Прехтера НАРУШЕНО → это НЕ импульс, а Y-волна
WXY. У BTC же одновременно W3 был middle → валидный 5-wave bear. Это
КРИТИЧНО для setup: альт со слабой структурой = follower, BTC с чистой
структурой = leader.

  1. ВАЛИДИРУЙ R0-R3 для альта НЕЗАВИСИМО от BTC. Не предполагай что
     структуры идентичны — они часто различаются.

  2. ОПРЕДЕЛИ leader/follower:
     - LEADER = тот у кого чище структура (валидный 5-wave / triangle /
       zigzag по hard rules).
     - FOLLOWER = тот у кого структура нарушает правила (W3 shortest,
       overlap, ambiguous count).
     - Trade LEADER агрессивнее (полный размер позиции, tight stop).
     - Trade FOLLOWER консервативнее ИЛИ пропусти (если leader даёт
       чёткий setup но follower нет — следи только за лидером).

  3. ПОСТРОЙ ТАБЛИЦУ СРАВНЕНИЯ (cross_asset_comparison field) с
     одинаковыми параметрами для обоих активов:
     - 15m структура (валидная/Y-wave/triangle/etc.)
     - Hard rules check (все ✓ vs нарушения)
     - Reversal probability (HIGH/MEDIUM/LOW)
     - Bounce target potential (61.8-78.6% retr vs 23.6-50% retr)
     - Leader/follower роль
     - Trade priority (priority/secondary/skip)
     Минимум 4 строки, идеально 6-8.

  4. CONFLUENCE TRIGGERS — сценарии должны иметь cross-asset триггеры
     явно: "Сценарий A работает только если BTC одновременно держит
     $77,839". Это даёт трейдеру cross-asset checklist.

  5. LAG ANALYSIS — практический совет: входи в follower с задержкой
     10-30 мин после подтверждения лидера. Это даёт лучшую цену и
     подтверждение.

  6. DIVERGENCE WATCH — если лидер делает чистую структуру а follower
     корявую — это сигнал что follower ещё не готов. Не входи только
     потому что лидер пошёл; жди пока структура follower'а тоже
     подтвердится (или работай только лидером).

Заполни поля `cross_asset_comparison` (массив строк таблицы) и
`leader_follower_note` (1-2 предложения вывода). Если торгуем BTC и
cross-asset анализ не нужен — оставь массив пустым [] и note "".

**Step 13 — MCP-ЭТАЛОННЫЙ ФОРМАТ ПОЛНОГО ОТЧЁТА (Phase 34).**
Reference: `tv_mcp_btc_3tf_full_report_w2_golden_zone_setup` в trading_examples.
Стремись к этому уровню детализации когда есть валидные данные на 3+ ТФ.
Заполни ВСЕ восемь полей ниже:

  1. **`period_context`** — context header в начале:
     - current_price (число), current_time (ISO/human-readable)
     - today_low + today_low_time
     - period_low + period_low_time (за окно анализа)
     - period_high + period_high_time
     - period_window (например "20.5 дней")

  2. **`tf_analysis`** — per-TF detailed analysis. Dict с TF как ключом, где
     каждое значение содержит:
     - section_title (например "🔭 1h TF — БОЛЬШАЯ КАРТИНА: Wave 2 в golden zone")
     - thesis (1-2 предложения главного тезиса)
     - wave_trace_ascii (мульти-line ASCII-дерево с `│` `▼` стрелками,
       timestamps, ценами и краткими notes). Это лучше читается чем
       таблица для последовательности событий.
     - fib_table с полями range_from, range_to, range_total, rows (массив
       с level/price/status), current_position. Каждая строка Fib имеет
       Status колонку (например "пробит" / "не пробит" / "invalidation").
     - pavelko_interpretation (что playbook говорит про эту разметку)
     - wave_projections — массив с полями name, price, formula. Wave 3 targets
       1.0×W1 / 1.618×W1 / 2.618×W1 если применимо
     - (для 15m) impulse_decomposition — массив объектов с полями
       impulse_id, label, waves, validation, playbook_outcome. Если на TF
       было >1 impulse'а подряд, разбей каждый ОТДЕЛЬНО.

  3. **`verdict_per_tf`** — таблица сводки. Dict с keys "headers" (массив
     заголовков ["TF", "Структура", "Сигнал", "Trade"]) и "rows" (массив
     строк [[1h-cells], [15m-cells], [5m-cells]]).

  4. **`confluence_factors`** — массив строк с ✓ галочками. Каждая —
     один confluence-фактор. Пример: "✓ 1h — golden 61.8% retrace zone".

  5. **`comparison_with_previous`** — таблица «было / стало». Особенно
     важна когда переразметка (REFRAMED) произошла. Dict с полями headers,
     rows, key_change. Reference: `comparison_with_previous` в эталоне.
     Если предыдущего holistic нет — пустой dict.

  6. **`monitor_next_hours`** — time-bound checklist. Dict с window
     (например "ближайшие 4-8 часов"), headers, rows[[condition, action]].

  7. **`critical_level`** — одна строка с уровнем инвалидации текущего
     thesis'а. Пример: "$77,601 — invalidation of immediate bullish
     thesis (но wave 2 ещё валидна до $74,900 на 1h)".

  8. **`golden_entry_zone`** — финальный actionable summary одной строкой
     с конкретными ценами и R/R. Пример: "🎯 GOLDEN ENTRY zone:
     $77,950-$78,200 если retest развивается типично. Stop $77,500,
     target $80,000+ → R/R 4-5:1."

КОГДА БЕЗ MCP-ФОРМАТА:
- Если данных мало (single TF, нет valid structure) — оставь эти восемь
  полей пустыми / "". Не выдумывай ASCII-trace без timestamps и фактов.
- Если есть данные на 2+ ТФ и хотя бы один valid impulse/correction —
  ОБЯЗАТЕЛЬНО заполни period_context + tf_analysis[main_tf] минимум.
- Полный эталон-формат (все 8 полей) — когда 3+ ТФ + valid structure
  + setup развивается + есть previous holistic для comparison.

## ФОРМАТ ОТВЕТА — СТРОГО JSON:

🚨 КРИТИЧЕСКОЕ ПРАВИЛО #1: каждое поле — ОТДЕЛЬНОЕ значение. НЕ помещай
весь анализ в summary. НЕ возвращай вложенный JSON внутри полей.

🚨 КРИТИЧЕСКОЕ ПРАВИЛО #2 — ТЕЛЕГРАФНЫЙ СТИЛЬ:
Каждое повествовательное поле — МАКСИМАЛЬНО ПЛОТНОЕ. Никаких водянистых
вводных, никаких «вероятно», «возможно», «может быть», «при этом», «также
стоит заметить». Только КОНКРЕТНЫЕ цифры, уровни, факты, выводы.

Примеры качества:
❌ ПЛОХО (водянистое): "На 4h в развитии волна 2 нового импульса вверх,
   но текущий ретрейс 28% является слишком мелким для квалифицированного
   1-2 setup согласно методологии Pavelko, поэтому требуется углубление
   коррекции до зоны 50-61.8% Фибоначчи."
✅ ХОРОШО (плотное): "4h: W2 retrace 28% (мелко, <50% sweet-spot). Ждём
   углубления к 74100-72100 (50-61.8% Fib) — там 1-2 LONG quality entry."

❌ ПЛОХО: "Согласно правилу альтернации, если волна 2 была резкой
   коррекцией, то волна 4 ожидается более плоской по форме."
✅ ХОРОШО: "4h-W2 sharp zigzag → 4h-W4 будет sideways flat/triangle (alt.)."

ЖЁСТКИЕ char-лимиты на повествовательные поля:
- structure_position: ≤280 chars (1d+4h+15m+5m по одному короткому факту)
- rules_check: ≤200 chars (R0-R3 чек одной строкой)
- alternation_check: ≤150 chars
- fibonacci_check: ≤200 chars (с конкретными уровнями)
- alternative_interpretation: ≤200 chars или ""
- summary: ≤350 chars (вердикт + ключевая цифра + действие)
- invalidation_explained: ≤120 chars
- per_tf_notes[tf]: ≤120 chars каждый
- cross_correlation_notes: ≤200 chars
- leader_follower_note: ≤200 chars

❌ ПЛОХО: "summary": "{{\\"bias\\": \\"long\\", \\"structure_position\\": \\"...\\", ...}}"
❌ ПЛОХО: "summary": "Всё в одном поле. structure_position: ... rules_check: ..."
✅ ХОРОШО: каждое поле верхнего уровня заполняется СВОИМ значением.

🚨 КРИТИЧЕСКОЕ ПРАВИЛО #2.5 — ПРИОРИТЕТ ПОЛЕЙ ПРИ ОГРАНИЧЕНИИ ДЛИНЫ:
JSON будет обрезан если не помещается в max_tokens=6000. ВСЕГДА сначала
заполняй ПРИОРИТЕТНЫЕ поля (P1-P3), и ТОЛЬКО потом опциональные.

  **P1 (КРИТИЧЕСКИЕ — заполнять ВСЕГДА в первую очередь):**
  bias, setup_quality, action, summary, action_plan,
  entry, stop_loss, target, risk_reward, invalidation_explained,
  golden_entry_zone, critical_level

  **P2 (ВАЖНЫЕ — заполнять если есть данные на 2+ ТФ):**
  structure_position, rules_check, fibonacci_check,
  per_tf_notes, scenarios, waiting_conditions,
  confluence_factors, monitor_next_hours, comparison_with_previous

  **P3 (ДЕТАЛЬНЫЕ — заполнять если есть данные на 3+ ТФ и хватает места):**
  period_context, tf_analysis (с ASCII wave_trace + fib_table per TF +
  wave_projections + impulse_decomposition), verdict_per_tf,
  entry_alternatives, target_pyramid, higher_degree_context,
  monitoring_levels, scalp_plan_5m, cross_asset_comparison,
  leader_follower_note, alternation_check, alternative_interpretation

  **P4 (ОПЦИОНАЛЬНЫЕ — заполнять только если ОСТАЛОСЬ много места):**
  chart_annotations

Если выйдешь в лимит — лучше пропусти P4 и часть P3, чем оборвать P1/P2
посередине. Не выдумывай данные ради заполнения P3 если их нет.

🚨 КРИТИЧЕСКОЕ ПРАВИЛО #3 — ACTIONABLE PLAN ОБЯЗАТЕЛЕН:
Поле `action_plan` — главный output трейдера. 4-6 пронумерованных шагов
формата «если X → делай Y». КОНКРЕТНЫЕ цены, время, действия.
БЕЗ философии, без «следует учитывать». Чистый telegrafic if-then план.

Пример качества action_plan:
✅ "1. СЕЙЧАС: не входить (W2 retrace 28% мелко). 2. Поставить алерты:
   74100 (38.2% Fib) и 72100 (50%). 3. Если цена в 74100±300 + 5m H-L-H
   reversal → LONG entry, stop 65900. 4. TP1: 90600 (R/R 2.0). 5. Если
   break < 66015 → invalid, сценарий B (extension вниз к 60000). 6.
   Re-check через 4-8h: если bounce без касания 74100 → пропустить."
❌ ПЛОХО: "Рекомендуется дождаться более глубокой коррекции, после чего
   рассмотреть возможность входа в long на основании комплекса факторов."

Структура (СТРОГО соблюдать порядок и наличие всех полей).
LEAN-схема: 14 core полей, без дублирования. Если поле не применимо —
"" для строк, null для чисел, [] для массивов, пустой dict для словарей:

{{
  // === P1 КРИТИЧЕСКИЕ — заполнять ВСЕГДА ===
  "bias": "long" | "short" | "neutral",
  "setup_quality": "high" | "medium" | "low" | "none",
  "action": "long" | "short" | "wait" | "close",
  "summary": "<≤300 chars: вердикт + ключевая цифра + действие. БЕЗ воды>",
  "entry": <число | null>,
  "stop_loss": <число | null>,
  "target": <число | null>,
  "risk_reward": <число | null>,
  "invalidation_explained": "<≤120 chars: конкретный уровень/событие отмены>",
  "golden_entry_zone": "<🎯 одна строка с зоной входа + stop + target + R/R. Например '🎯 GOLDEN ENTRY: $77,950-$78,200, stop $77,500, target $80,000+ → R/R 4-5:1'. Пусто '' если setup=wait без конкретной зоны.>",
  "critical_level": "<≤120 chars: уровень инвалидации thesis'а. Например '$77,601 — break = bear continuation к $74,900'. Пусто '' если N/A.>",
  "action_plan": [
    // 🚨 4-6 шагов. Каждый — конкретное actionable действие с if-then.
    // КОНКРЕТНЫЕ цены, время, R/R. БЕЗ философии. Главный output трейдера.
    {{"step": 1, "action": "<например 'СЕЙЧАС: не входить, W2 retrace 28% мелко'>"}},
    {{"step": 2, "action": "<например 'Поставить алерты: $74,100 (38.2%), $72,100 (50%)'>"}},
    {{"step": 3, "action": "<например 'Если цена $74,100±300 + 5m H-L-H → LONG, stop $65,900'>"}},
    {{"step": 4, "action": "<например 'TP1: $90,600 (R/R 2.0). TP2: $100,200 (R/R 3.0)'>"}},
    {{"step": 5, "action": "<например 'Invalidation: break < $66,015 → закрыть, bearish сценарий'>"}}
  ],
  "scenarios": [
    // 2-3 сценария с probability weights (суммируются ≈100%).
    // Каждый со своим entry_zone/stop/targets — это ЗАМЕНЯЕТ
    // entry_alternatives + target_pyramid.
    {{
      "name":            "<например 'A — Bullish bounce'>",
      "probability_pct": <число 0-100>,
      "trigger":         "<условие активации>",
      "action":          "<что делать>",
      "entry_zone":      "<entry или диапазон, null если wait>",
      "stop":            <число | null>,
      "targets":         "<например 'TP1 $79,400 (R/R 2). TP2 $80,400 (R/R 3.5)'>"
    }}
  ],
  "monitor_next_hours": {{
    // Time-bound condition/action checklist.
    "window":  "<например 'ближайшие 4-8 часов'>",
    "headers": ["Условие", "Действие"],
    "rows": [
      ["<условие>", "<действие>"],
      ["<условие>", "<действие>"]
    ]
  }},

  // === P2 ВАЖНЫЕ — заполнять если есть данные ===
  "structure_position": "<≤280 chars, телеграфно: 'TF: факт. TF: факт.' пер 1d/4h/15m/5m. Без воды.>",
  "rules_check": "<≤200 chars: R0-R3 чек одной строкой 'R3 OK ✓ / W2 56% / W3 ext 1.6× / no overlap'>",
  "per_tf_notes": {{
    // По одному короткому факту на TF (≤120 chars каждый).
    // Заменяет громоздкий tf_analysis с ASCII.
    "1d":  "<например 'ABC up от $62909→$82219 (C=A precisely). Wait для break > $96945.'>",
    "4h":  "<...>",
    "15m": "<...>",
    "5m":  "<...>"
  }},
  "confluence_factors": [
    // 3-7 факторов выровненных в одном направлении. Каждый начинается с ✓.
    "✓ <фактор 1>",
    "✓ <фактор 2>"
  ],
  "comparison_with_previous": {{
    // Опционально — если previous holistic существует и изменилась
    // разметка/bias. Если нет — пустой dict.
    "headers": ["Metric", "Было", "Стало"],
    "rows": [
      ["1h контекст", "<было>", "<стало ← REFRAMED!>"]
    ],
    "key_change": "<1 предложение про сдвиг>"
  }},
  "invalidation_of_previous": "<если предыдущий сохранённый прогноз пробит — открыто признать. Пусто '' если не было invalidation.>",
  "entry_zone_shift": "<≤200 chars: если previous holistic существует и зона входа сместилась — опиши сдвиг. Пример: 'Buy zone сместилась с $43.20 (v2) на $43.80 (v3) — VPOC reshape, volume cluster $43-44 теперь activated'. Пусто '' если первый анализ или зона не изменилась.>",
  "structural_events_timeline": [
    // Phase 47 — таблица ключевых событий с timestamp + ценой.
    // Inspired by MCP v8: показывает прогрессию Breakout → Retest →
    // Bounce → Current. Помогает увидеть структуру без графика.
    // 2-5 events обычно.
    {{"event": "Breakout (v7 trigger)", "time": "2026-05-19 17:00", "price": 48.94, "kind": "H"}},
    {{"event": "Retest support",        "time": "2026-05-20 00:00", "price": 47.10, "kind": "L"}},
    {{"event": "Bounce",                 "time": "2026-05-20 05:00", "price": 48.66, "kind": "H"}},
    {{"event": "Current",                "time": "<now>",            "price": <число>, "kind": "C"}}
  ],
  "invalidation_layers": {{
    // Phase 47 — 3-уровневая инвалидация для риск-менеджмента.
    // Soft = partial close / trim, Hard = full stop, Macro = переразметка.
    // Каждый уровень на своём ТФ.
    "soft":  {{"trigger": "1h close < $47.00", "action": "partial close / trim 50%", "tf": "1h"}},
    "hard":  {{"trigger": "4h close < $46.50", "action": "full stop loss",          "tf": "4h"}},
    "macro": {{"trigger": "1d close < $44.20", "action": "переразметка bias и структуры", "tf": "1d"}}
  }},
  "gut_probability_next_window": {{
    // Phase 47 — time-bound probabilistic forecast на ближайшие 24-48h.
    // Inspired by MCP v8 "Gut probability".
    "window": "next 24-48h",
    "scenarios": [
      {{"name": "Continuation",     "probability_pct": 65, "expected_range": "$50-52"}},
      {{"name": "Sideways",          "probability_pct": 25, "expected_range": "$47.10-48.94"}},
      {{"name": "Deeper pullback",   "probability_pct": 10, "expected_range": "$46.0-46.5"}}
    ]
  }},
  "entry_variants": [
    // Phase 42 — ТРИ варианта входа с разной агрессивностью. Inspired by
    // MCP TV v4. Каждый имеет разный stop, risk, R/R. Трейдер выбирает
    // по своему risk tolerance И уверенности в setup.
    //
    // Если action=wait или setup quality=low — оставь [] пустым.
    // Если есть active/planned LONG или SHORT setup — заполни 2-3
    // варианта (минимум Aggressive + Confirmed).
    {{
      "name":      "Aggressive",
      "entry":     <число>,                // market entry now или в текущей зоне
      "stop":      <число>,
      "risk_pct":  <число | null>,         // % от entry до stop
      "rr":        <число>,                // R/R до primary target
      "trigger":   "<пусто — market entry в зоне>",
      "rationale": "<например 'Pullback to $44.20 уже sweep. C=A equality + capitulation volume = ранний entry'>"
    }},
    {{
      "name":      "Confirmed",
      "entry":     <число>,                // после конкретного break/close
      "stop":      <число>,
      "risk_pct":  <число | null>,
      "rr":        <число>,
      "trigger":   "<например '15m close > $45.20'>",
      "rationale": "<например 'Подтверждение higher-high на 15m, безопаснее но R/R хуже'>"
    }},
    {{
      "name":      "Breakout",
      "entry":     <число>,                // на retest пробитого уровня
      "stop":      <число>,
      "risk_pct":  <число | null>,
      "rr":        <число>,
      "trigger":   "<например '1h close > $46.30, затем retest $45.80-46.00'>",
      "rationale": "<например 'Самый безопасный — wait for VPOC break confirmation'>"
    }}
  ],
  // Phase 38: confidence_factors и risk_factors УБРАНЫ — дублировали
  // summary + scenarios.triggers. Если нужно подчеркнуть фактор
  // ЗА/ПРОТИВ — добавь его прямо в summary одной строкой.
  "sentiment_alignment": "aligned" | "divergent" | "n/a",
  "tv_alignment": "aligned" | "divergent" | "n/a"
}}

ВАЖНО: ответ — это ОДИН JSON-объект, БЕЗ markdown-фенс (``` json), БЕЗ
текста до или после. Просто {{...}}. ВСЕ P1 поля ОБЯЗАТЕЛЬНЫ (12 штук).
P2 заполняй если есть данные. Если поле не применимо — "" для строк,
null для чисел, [] для массивов, пустой dict для словарей.

КОМПАКТНОСТЬ ВЫШЕ ВСЕГО: каждое повествовательное поле строго в
char-лимите. Один факт = одно поле. БЕЗ дублирования между полями
(scenarios уже содержат entry_zone/targets — НЕ дублируй в отдельный
entry_alternatives). max_tokens=2500 жёсткий лимит — не превышай.

🆕 PHASE 47 — три новых ключевых поля из MCP v8:
- **structural_events_timeline**: 2-5 ключевых событий с timestamp +
  ценой (Breakout/Retest/Bounce/Current). Визуальная timeline что
  произошло за последние часы/дни. Помогает увидеть прогрессию.
- **invalidation_layers**: 3-уровневая инвалидация вместо одного
  critical_level. Soft (1h close < X = trim 50%) / Hard (4h close <
  Y = full stop) / Macro (1d close < Z = переразметка). Это даёт
  trader риск-менеджмент в стиле MCP.
- **gut_probability_next_window**: time-bound прогноз («next 24-48h:
  65% continuation $50-52 / 25% sideways / 10% pullback»). Это более
  actionable чем абстрактные scenarios — даёт временное окно.

Эти 3 поля ОПЦИОНАЛЬНЫ — заполняй если есть основание (структура
ясна, события узнаются, временные рамки определимы). Если нет —
оставь пустой dict или пустой массив. БЕЗ выдумок."""


def ask_claude_holistic(
    ticker: str,
    tf_blocks: list[dict],
    model: str = DEFAULT_MODEL,
    use_cache: bool = True,
    prev_holistic: dict | None = None,  # Phase 41 — for versioned v2→v3 diff
    track_record: list[dict] | None = None,  # Phase 43 — verified history
) -> AIHolisticOpinion:
    """Send all 4 TF context in one request, get a unified plan back."""
    api_key = _get_api_key()
    if api_key is None:
        raise RuntimeError(
            "API-ключ Anthropic не настроен. Добавь ANTHROPIC_API_KEY в .env."
        )
    import anthropic

    # Build a stable cache key from all TF signatures.
    sig_parts = [ticker]
    for tfb in tf_blocks:
        sig_parts.append(tfb["name"])
        sig_parts.append(_swings_signature(tfb["swings"], tfb["forecast"]))
    cache_key = ("HOLISTIC", "|".join(sig_parts))
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            _safe_call_log(
                ticker=ticker, call_type="holistic", timeframe="all",
                model=model, prompt="", raw_response=cached.raw_text,
                parsed={"bias": cached.bias, "action": cached.action,
                         "entry": cached.entry, "stop_loss": cached.stop_loss,
                         "target": cached.target, "risk_reward": cached.risk_reward,
                         "setup_quality": cached.setup_quality,
                         "sentiment_alignment": cached.sentiment_alignment},
                tokens_in=0, tokens_out=0, duration_s=0.0, cached=True,
            )
            return cached

    client = anthropic.Anthropic(api_key=api_key, timeout=DEFAULT_TIMEOUT_S)
    prompt = build_holistic_prompt(
        ticker, tf_blocks,
        prev_holistic=prev_holistic,
        track_record=track_record,
    )
    # Collect top candidates from every TF to narrow the knowledge cards
    # to the patterns actually relevant to this analysis.
    holistic_focus = []
    for tfb in tf_blocks:
        for c in (tfb.get("candidates") or [])[:2]:
            holistic_focus.append(c)
    system_block = _system_prompt_with_knowledge(focus_candidates=holistic_focus or None)
    call_started = time.monotonic()

    last_error: Exception | None = None
    msg = None
    for attempt in range(MAX_RETRIES):
        try:
            create_kwargs: dict = {
                "model": model,
                # Phase 36 — обратно к 2500. Anton: «не повышать лимиты,
                # сделать ответ компактнее». Lean schema убрала дубли
                # (entry_alternatives + target_pyramid → scenarios;
                # waiting_conditions → action_plan; monitoring_levels →
                # monitor_next_hours; tf_analysis ASCII → per_tf_notes
                # короткие; убраны: chart_annotations, alternation_check,
                # alternative_interpretation, higher_degree_context,
                # fibonacci_check, scalp_plan_5m, cross_asset_comparison).
                "max_tokens": 2500,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_block:
                create_kwargs["system"] = system_block
            msg = client.messages.create(**create_kwargs)
            break
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_S[attempt])
            continue
        except anthropic.APIStatusError as exc:
            # Phase 45 — retry on transient server errors (529 overloaded,
            # 503 unavailable, 502 bad gateway, 504 gateway timeout, 500
            # internal). For other status codes (400 bad request, 401 auth,
            # 429 rate limit) fall through to the generic handler below.
            status = getattr(exc, "status_code", None)
            if status in (500, 502, 503, 504, 529) and attempt < MAX_RETRIES - 1:
                last_error = exc
                wait = RETRY_BACKOFF_S[attempt] * 2  # longer for 5xx
                time.sleep(wait)
                continue
            # Non-transient 4xx or last attempt — surface to caller
            _safe_call_log(ticker=ticker, call_type="holistic",
                            timeframe="all", model=model, prompt=prompt,
                            error=f"{type(exc).__name__}: {exc}",
                            duration_s=time.monotonic() - call_started)
            raise _humanize_anthropic_error(exc) from exc
        except anthropic.BadRequestError as exc:
            _safe_call_log(ticker=ticker, call_type="holistic",
                            timeframe="all", model=model, prompt=prompt,
                            error=f"{type(exc).__name__}: {exc}",
                            duration_s=time.monotonic() - call_started)
            raise _humanize_anthropic_error(exc) from exc
    if msg is None:
        _safe_call_log(ticker=ticker, call_type="holistic", timeframe="all",
                        model=model, prompt=prompt,
                        error=f"timeout after {MAX_RETRIES} retries: {last_error}",
                        duration_s=time.monotonic() - call_started)
        raise RuntimeError(
            f"Claude недоступен после {MAX_RETRIES} попыток: "
            f"{type(last_error).__name__}: {last_error}"
        )

    raw = ""
    for block in msg.content:
        if hasattr(block, "text"):
            raw += block.text
    raw = raw.strip()

    tokens_in  = getattr(getattr(msg, "usage", None), "input_tokens", 0) or 0
    tokens_out = getattr(getattr(msg, "usage", None), "output_tokens", 0) or 0
    call_duration = time.monotonic() - call_started

    parsed: dict = _extract_json_from_response(raw)

    # Fallback: if Claude stuffed the chain-of-thought fields into `summary`
    # as JSON-text instead of returning them at the top level, try to
    # extract the embedded JSON and merge it back.
    summary_field = parsed.get("summary")
    if isinstance(summary_field, str) and "{" in summary_field \
            and "}" in summary_field \
            and not parsed.get("structure_position"):
        inner = _extract_json_from_response(summary_field)
        if inner:
            # Merge inner over parsed (inner wins for missing keys)
            for k, v in inner.items():
                if k not in parsed or not parsed.get(k):
                    parsed[k] = v
            # Replace the polluted summary with the real one from inner
            real_summary = inner.get("summary")
            if isinstance(real_summary, str) and "{" not in real_summary:
                parsed["summary"] = real_summary

    notes = parsed.get("per_tf_notes", {})
    if not isinstance(notes, dict):
        notes = {}

    # Phase 13 — parse chain-of-thought fields (lists may come as JSON arrays
    # or comma-separated strings; normalize to list[str]).
    def _as_list(v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            return [s.strip() for s in v.split(";") if s.strip()]
        return [str(v)]

    opinion = AIHolisticOpinion(
        bias=str(parsed.get("bias", "neutral")).lower(),
        setup_quality=str(parsed.get("setup_quality", "none")).lower(),
        summary=str(parsed.get("summary", raw if not parsed else "")),
        action=str(parsed.get("action", "wait")).lower(),
        entry=_num_or_none(parsed.get("entry")),
        stop_loss=_num_or_none(parsed.get("stop_loss")),
        target=_num_or_none(parsed.get("target")),
        risk_reward=_num_or_none(parsed.get("risk_reward")),
        invalidation_explained=str(parsed.get("invalidation_explained") or ""),
        per_tf_notes={str(k): str(v) for k, v in notes.items()},
        raw_text=raw,
        sentiment_alignment=str(parsed.get("sentiment_alignment") or "n/a").lower(),
        tv_alignment=str(parsed.get("tv_alignment") or "n/a").lower(),
        structure_position=str(parsed.get("structure_position") or ""),
        rules_check=str(parsed.get("rules_check") or ""),
        alternation_check=str(parsed.get("alternation_check") or ""),
        fibonacci_check=str(parsed.get("fibonacci_check") or ""),
        alternative_interpretation=str(parsed.get("alternative_interpretation") or ""),
        confidence_factors=_as_list(parsed.get("confidence_factors")),
        risk_factors=_as_list(parsed.get("risk_factors")),
        chart_annotations=[
            a for a in (parsed.get("chart_annotations") or [])
            if isinstance(a, dict) and a.get("type")
        ],
        scalp_plan=(parsed.get("scalp_plan_5m")
                    if isinstance(parsed.get("scalp_plan_5m"), dict)
                    else {}),
        higher_degree_context=(parsed.get("higher_degree_context")
                                if isinstance(parsed.get("higher_degree_context"), dict)
                                else {}),
        monitoring_levels=[
            lvl for lvl in (parsed.get("monitoring_levels") or [])
            if isinstance(lvl, dict) and lvl.get("trigger")
        ],
        entry_alternatives=[
            ea for ea in (parsed.get("entry_alternatives") or [])
            if isinstance(ea, dict) and ea.get("name")
        ],
        target_pyramid=[
            t for t in (parsed.get("target_pyramid") or [])
            if isinstance(t, dict) and t.get("name")
            and isinstance(t.get("price"), (int, float))
        ],
        cross_correlation_notes=str(parsed.get("cross_correlation_notes") or ""),
        scenarios=[
            s for s in (parsed.get("scenarios") or [])
            if isinstance(s, dict) and s.get("name")
        ],
        waiting_conditions=[
            w for w in (parsed.get("waiting_conditions") or [])
            if isinstance(w, dict) and w.get("condition")
        ],
        invalidation_of_previous=str(parsed.get("invalidation_of_previous") or ""),
        cross_asset_comparison=[
            row for row in (parsed.get("cross_asset_comparison") or [])
            if isinstance(row, dict) and row.get("parameter")
        ],
        leader_follower_note=str(parsed.get("leader_follower_note") or ""),
        action_plan=[
            s for s in (parsed.get("action_plan") or [])
            if isinstance(s, dict) and s.get("action")
        ],
        period_context=(parsed.get("period_context")
                        if isinstance(parsed.get("period_context"), dict) else {}),
        tf_analysis=(parsed.get("tf_analysis")
                     if isinstance(parsed.get("tf_analysis"), dict) else {}),
        verdict_per_tf=(parsed.get("verdict_per_tf")
                        if isinstance(parsed.get("verdict_per_tf"), dict) else {}),
        confluence_factors=_as_list(parsed.get("confluence_factors")),
        comparison_with_previous=(parsed.get("comparison_with_previous")
                                  if isinstance(parsed.get("comparison_with_previous"), dict) else {}),
        entry_zone_shift=str(parsed.get("entry_zone_shift") or ""),
        entry_variants=[
            v for v in (
                # Phase 44 — accept BOTH names. Claude often emits
                # entry_alternatives (legacy format from HYPE truncated
                # example with tight/wider). Map both into entry_variants.
                (parsed.get("entry_variants") or [])
                + (parsed.get("entry_alternatives") or [])
            )
            if isinstance(v, dict) and v.get("name")
            and isinstance(v.get("entry"), (int, float))
        ],
        was_truncated=bool(parsed.get("_was_truncated")),
        structural_events_timeline=[
            ev for ev in (parsed.get("structural_events_timeline") or [])
            if isinstance(ev, dict) and ev.get("event")
        ],
        invalidation_layers=(parsed.get("invalidation_layers")
                              if isinstance(parsed.get("invalidation_layers"), dict)
                              else {}),
        gut_probability_next_window=(
            parsed.get("gut_probability_next_window")
            if isinstance(parsed.get("gut_probability_next_window"), dict)
            else {}
        ),
        monitor_next_hours=(parsed.get("monitor_next_hours")
                            if isinstance(parsed.get("monitor_next_hours"), dict) else {}),
        critical_level=str(parsed.get("critical_level") or ""),
        golden_entry_zone=str(parsed.get("golden_entry_zone") or ""),
    )
    if use_cache:
        _cache_put(cache_key, opinion)

    # Audit log for backtesting + cost tracking
    _safe_call_log(
        ticker=ticker, call_type="holistic", timeframe="all",
        model=model, prompt=prompt, raw_response=raw,
        parsed=parsed, tokens_in=tokens_in, tokens_out=tokens_out,
        duration_s=call_duration, cached=False,
    )
    return opinion
