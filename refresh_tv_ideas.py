"""Background sidecar — fetches TradingView Ideas via Apify, aggregates,
appends history, refreshes index. Runs twice a day (similar to twitter-refresher).
"""
from __future__ import annotations
import argparse, logging, os, sys, time
from datetime import datetime, timedelta, timezone

REFRESH_HOURS_UTC = (6, 18)


def _setup_logging() -> logging.Logger:
    log = logging.getLogger("refresh_tv_ideas")
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
    from tv_ideas_scraper import scrape, save_snapshot, _apify_token
    from tv_ideas_aggregate import refresh_from_snapshots

    started = time.monotonic()
    log.info("=" * 60)
    log.info("TV Ideas refresh starting at %s",
              datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

    if not _apify_token():
        log.warning("APIFY_API_TOKEN missing — skipping scrape, only re-aggregating.")
        try:
            agg = refresh_from_snapshots(within_hours=48)
            return {"ok": True, "tickers": len(agg), "skipped_scrape": True}
        except Exception as exc:
            log.error("Aggregation failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    try:
        ideas = scrape(max_per_source=40)
    except Exception as exc:
        log.error("TV scrape failed: %s", exc)
        return {"ok": False, "error": str(exc)}

    if not ideas:
        log.warning("Scrape returned 0 ideas.")
        return {"ok": True, "ideas": 0}

    snap = save_snapshot(ideas)
    log.info("Saved %d ideas → %s", len(ideas), snap)
    try:
        agg = refresh_from_snapshots(within_hours=48)
        log.info("Aggregated TV ideas for %d tickers", len(agg))
    except Exception as exc:
        log.error("Aggregation failed: %s", exc)
        return {"ok": False, "error": str(exc)}

    try:
        os.makedirs("tv_ideas", exist_ok=True)
        with open("tv_ideas/.heartbeat", "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass

    elapsed = time.monotonic() - started
    log.info("Cycle complete in %.1fs", elapsed)
    return {"ok": True, "ideas": len(ideas), "tickers": len(agg)}


def _sleep_until_next_slot() -> float:
    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    slots = sorted([today + timedelta(hours=h) for h in REFRESH_HOURS_UTC]
                    + [today + timedelta(days=1, hours=h) for h in REFRESH_HOURS_UTC])
    for s in slots:
        if s > now:
            return (s - now).total_seconds()
    return 12 * 3600


def run_loop(log: logging.Logger) -> None:
    log.info("refresh_tv_ideas --loop, slots UTC: %s", REFRESH_HOURS_UTC)
    hb = "tv_ideas/.heartbeat"
    needs_first = True
    if os.path.isfile(hb):
        age_h = (time.time() - os.path.getmtime(hb)) / 3600
        if age_h < 8:
            needs_first = False
            log.info("Heartbeat %.1fh old, skipping immediate cycle.", age_h)
    if needs_first:
        run_one_cycle(log)
    while True:
        sleep_s = _sleep_until_next_slot()
        log.info("Sleeping %.1fh until next slot…", sleep_s / 3600)
        time.sleep(sleep_s)
        try:
            run_one_cycle(log)
        except Exception as exc:
            log.exception("Unhandled error: %s", exc)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true")
    g.add_argument("--loop", action="store_true")
    args = p.parse_args()
    log = _setup_logging()
    if args.once:
        return 0 if run_one_cycle(log).get("ok") else 1
    run_loop(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
