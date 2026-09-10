"""News sentiment scoring with VADER (no external API key needed).

Why VADER and not a transformer model?
  - Tiny (~250KB), pure Python, no PyTorch dependency.
  - Specifically tuned for short social-media / headline-style text.
  - Returns a single 'compound' score in [-1, +1] that's intuitive to
    aggregate across many headlines.

News source:
  - yfinance.Ticker(symbol).news — free, no auth, returns recent items
    with title / publisher / publish time / link.
  - Schema is brittle (Yahoo changes it occasionally); we defensively
    look for several field names.
  - Users can also paste their own headlines on the page.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:  # graceful: tests pass even without the package installed
    _VADER_AVAILABLE = False
    SentimentIntensityAnalyzer = None  # type: ignore[assignment,misc]


# Compound-score thresholds (the VADER author's recommended cutoffs).
POSITIVE_THRESHOLD: float = 0.05
NEGATIVE_THRESHOLD: float = -0.05


@dataclass
class HeadlineScore:
    title: str
    score: float                  # VADER compound, [-1, +1]
    label: str                    # 'positive' | 'neutral' | 'negative'
    publisher: str = ""
    timestamp: int = 0            # unix epoch seconds
    link: str = ""

    @property
    def datetime(self) -> datetime | None:
        return datetime.fromtimestamp(self.timestamp) if self.timestamp else None


@dataclass
class TickerSentiment:
    ticker: str
    headlines: list[HeadlineScore]
    mean_score: float
    median_score: float
    label: str                    # bullish | neutral | bearish
    n_pos: int
    n_neu: int
    n_neg: int
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def n(self) -> int:
        return len(self.headlines)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _label_for(score: float) -> str:
    if score >= POSITIVE_THRESHOLD:
        return "positive"
    if score <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def _aggregate_label(mean_score: float) -> str:
    if mean_score >= POSITIVE_THRESHOLD:
        return "bullish"
    if mean_score <= NEGATIVE_THRESHOLD:
        return "bearish"
    return "neutral"


def get_analyzer():
    """Lazily construct a VADER analyzer (or raise a clean error)."""
    if not _VADER_AVAILABLE:
        raise RuntimeError(
            "vaderSentiment is not installed. Add `vaderSentiment>=3.3.2` "
            "to requirements.txt and rebuild the container."
        )
    return SentimentIntensityAnalyzer()


def score_text(text: str, analyzer=None) -> float:
    """Return VADER compound score for a single string."""
    if not text or not text.strip():
        return 0.0
    if analyzer is None:
        analyzer = get_analyzer()
    return float(analyzer.polarity_scores(text)["compound"])


def score_headlines(
    headlines: list[dict], analyzer=None
) -> list[HeadlineScore]:
    """Score a list of headline dicts (with at least a 'title' field)."""
    if analyzer is None:
        analyzer = get_analyzer()
    out: list[HeadlineScore] = []
    for h in headlines:
        title = (h.get("title") or "").strip()
        if not title:
            continue
        s = score_text(title, analyzer)
        out.append(HeadlineScore(
            title=title,
            score=s,
            label=_label_for(s),
            publisher=h.get("publisher", "") or "",
            timestamp=int(h.get("providerPublishTime", 0) or 0),
            link=h.get("link", "") or "",
        ))
    return out


# ---------------------------------------------------------------------------
# News fetch (best-effort)
# ---------------------------------------------------------------------------
def fetch_news(ticker: str, max_items: int = 25) -> list[dict]:
    """Best-effort news fetch via yfinance. Returns empty on failure.

    yfinance's news schema is unstable. We coerce whatever we get into
    {title, publisher, providerPublishTime, link}.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker.strip().upper())
        raw = t.news or []
    except Exception:
        return []

    out: list[dict] = []
    for item in raw[:max_items]:
        # yfinance has had two schemas: flat and nested under 'content'.
        if "content" in item and isinstance(item["content"], dict):
            c = item["content"]
            out.append({
                "title": c.get("title", ""),
                "publisher": (c.get("provider") or {}).get("displayName", "")
                              if isinstance(c.get("provider"), dict) else
                              (c.get("publisher") or ""),
                "providerPublishTime": int(_parse_time(c.get("pubDate", 0))),
                "link": (c.get("canonicalUrl") or {}).get("url", "")
                         if isinstance(c.get("canonicalUrl"), dict) else
                         (c.get("link") or ""),
            })
        else:
            out.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "providerPublishTime": int(item.get("providerPublishTime", 0) or 0),
                "link": item.get("link", ""),
            })
    return [o for o in out if o["title"]]


def _parse_time(raw) -> int:
    """Normalize timestamps in either epoch-int or ISO-string form."""
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
        except Exception:
            return 0
    return 0


# ---------------------------------------------------------------------------
# Top-level: analyze one ticker
# ---------------------------------------------------------------------------
def analyze_ticker(ticker: str, analyzer=None) -> TickerSentiment:
    """Fetch + score news for one ticker."""
    ticker = ticker.strip().upper()
    try:
        items = fetch_news(ticker)
    except Exception as exc:
        return TickerSentiment(
            ticker=ticker, headlines=[],
            mean_score=0.0, median_score=0.0,
            label="neutral", n_pos=0, n_neu=0, n_neg=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    scored = score_headlines(items, analyzer=analyzer)
    return _aggregate(ticker, scored)


def analyze_pasted(
    ticker: str, pasted_text: str, analyzer=None
) -> TickerSentiment:
    """Score user-pasted headlines (one per line). Useful when yfinance.news fails."""
    headlines = [
        {"title": line.strip()}
        for line in pasted_text.splitlines()
        if line.strip()
    ]
    scored = score_headlines(headlines, analyzer=analyzer)
    return _aggregate(ticker.strip().upper() or "MANUAL", scored)


def _aggregate(ticker: str, scored: list[HeadlineScore]) -> TickerSentiment:
    if not scored:
        return TickerSentiment(
            ticker=ticker, headlines=[],
            mean_score=0.0, median_score=0.0,
            label="neutral", n_pos=0, n_neu=0, n_neg=0,
            error="No headlines found.",
        )
    scores = [h.score for h in scored]
    mean = float(sum(scores) / len(scores))
    med = float(pd.Series(scores).median())
    n_pos = sum(1 for h in scored if h.label == "positive")
    n_neg = sum(1 for h in scored if h.label == "negative")
    n_neu = len(scored) - n_pos - n_neg
    return TickerSentiment(
        ticker=ticker, headlines=scored,
        mean_score=mean, median_score=med,
        label=_aggregate_label(mean),
        n_pos=n_pos, n_neu=n_neu, n_neg=n_neg,
    )
