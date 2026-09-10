"""Background refresher for Twitter sentiment.

Runs continuously inside the `twitter-refresher` sidecar container.
Twice per day (default: 06:00 and 18:00 UTC), runs the Apify scraper,
saves a snapshot, and re-aggregates the sentiment index.

Usage:
    python refresh_twitter.py --once    # one cycle, exit
    python refresh_twitter.py --loop    # continuous, sleeps to next slot

If APIFY_API_TOKEN is missing, logs a warning and skips the cycle gracefully.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

# Hours of day to refresh (UTC). Two slots = ~12h spacing.
REFRESH_HOURS_UTC = (6, 18)


def _setup_logging() -> logging.Logger:
    log = logging.getLogger("refresh_twitter")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(h)
    return log


def run_one_cycle(log: logging.Logger) -> dict:
    """One scrape + aggregate cycle. Returns status."""
    from twitter_scraper import scrape, save_snapshot, _apify_token
    from twitter_sentiment import refresh_from_snapshots

    started = time.monotonic()
    log.info("=" * 60)
    log.info("Twitter refresh cycle starting at %s",
              datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

    if not _apify_token():
        log.warning("APIFY_API_TOKEN not set — skipping live scrape. "
                    "(Sentiment can still be computed from existing snapshots.)")
        try:
            agg = refresh_from_snapshots(within_hours=48)
            log.info("Re-aggregated %d tickers from existing snapshots", len(agg))
            return {"ok": True, "tickers": len(agg), "skipped_scrape": True}
        except Exception as exc:
            log.error("Aggregation failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    try:
        tweets = scrape(hours_back=24, max_tweets_per_account=10)
    except Exception as exc:
        log.error("Scrape failed: %s", exc)
        return {"ok": False, "error": str(exc)}

    if not tweets:
        log.warning("Scrape returned 0 tweets — Apify may be rate-limiting "
                    "or the actor configuration is off.")
        return {"ok": True, "tweets": 0}

    snapshot_path = save_snapshot(tweets)
    log.info("Saved snapshot with %d tweets → %s", len(tweets), snapshot_path)

    try:
        agg = refresh_from_snapshots(within_hours=48)
        log.info("Aggregated sentiment for %d tickers", len(agg))
    except Exception as exc:
        log.error("Aggregation failed: %s", exc)
        return {"ok": False, "error": str(exc)}

    elapsed = time.monotonic() - started
    log.info("Cycle complete in %.1fs", elapsed)

    # Heartbeat file
    try:
        os.makedirs("twitter_sentiment", exist_ok=True)
        with open("twitter_sentiment/.heartbeat", "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass

    return {"ok": True, "tweets": len(tweets), "tickers": len(agg)}


def _seconds_until_next_slot() -> float:
    """Return seconds until the next refresh slot (UTC)."""
    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    slots = sorted([
        today + timedelta(hours=h) for h in REFRESH_HOURS_UTC
    ] + [
        today + timedelta(days=1, hours=h) for h in REFRESH_HOURS_UTC
    ])
    for s in slots:
        if s > now:
            return (s - now).total_seconds()
    return 12 * 3600  # fallback 12h


def run_loop(log: logging.Logger) -> None:
    log.info("refresh_twitter starting in --loop mode "
              "(slots UTC: %s)", REFRESH_HOURS_UTC)
    # First cycle on startup if no recent heartbeat
    hb_path = "twitter_sentiment/.heartbeat"
    needs_first_cycle = True
    if os.path.isfile(hb_path):
        age_hours = (time.time() - os.path.getmtime(hb_path)) / 3600
        if age_hours < 8:
            log.info("Recent heartbeat (%.1fh old), skipping immediate cycle.",
                      age_hours)
            needs_first_cycle = False
    if needs_first_cycle:
        run_one_cycle(log)

    while True:
        sleep_s = _seconds_until_next_slot()
        log.info("Sleeping %.1fh until next slot…", sleep_s / 3600)
        time.sleep(sleep_s)
        try:
            run_one_cycle(log)
        except Exception as exc:
            log.exception("Unhandled error in cycle: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true")
    g.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    log = _setup_logging()
    if args.once:
        status = run_one_cycle(log)
        return 0 if status.get("ok") else 1
    if args.loop:
        run_loop(log)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
