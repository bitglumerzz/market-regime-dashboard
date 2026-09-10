"""Ingest new knowledge-base material — annotate screenshots via Claude vision.

Workflow:
  1. User drops .png / .jpg files into
     ``knowledge_base/sources/<slug>/screenshots/``.
  2. This script reads each image, sends it to Claude with a structured
     instruction ("describe the pattern you see, what TF, what timeframe"),
     and appends/updates ``descriptions.yaml`` next to the images.
  3. Idempotent: if an entry for that filename already exists, the script
     skips it unless ``--force`` is passed.

Usage:
  python wave_knowledge_ingest.py                          # all sources
  python wave_knowledge_ingest.py --source pavelko_fxpro_advanced
  python wave_knowledge_ingest.py --source <slug> --force

Cost: ~$0.015 per image at Sonnet (~1.5k tokens vision + ~500 reply).
Skips if no ANTHROPIC_API_KEY is set — prints a warning and exits 0 so
this is safe to wire into a sidecar.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from wave_knowledge import (
    SOURCES_DIR, _read_yaml, _write_yaml, load_sources, screenshot_descriptions,
)


VISION_MODEL = "claude-opus-4-7"
MAX_TOKENS = 700


VISION_PROMPT = """Ты — эксперт по волновой теории Эллиотта. На этом изображении
показан финансовый график с волновой разметкой. Опиши его в JSON формате:

{
  "patterns": ["impulse_up", "leading_diagonal", ...],   // pattern slugs (snake_case)
  "timeframe": "1D" | "4H" | "1W" | "15m" | ...,
  "symbol": "EURUSD" | "BTC" | ...,
  "description": "детальное описание того что видно на графике на русском",
  "teaching_point": "главная мысль которую иллюстрирует этот пример"
}

Допустимые pattern slugs:
  impulse_up / impulse_down / leading_diagonal / ending_diagonal /
  zigzag_correction / flat_correction / expanded_flat / triangle_correction /
  double_three / triple_three

Верни ТОЛЬКО JSON, без преамбулы и комментариев."""


def _encode_image(path: str) -> tuple[str, str]:
    """Return (media_type, base64_data) for an image file."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    if ext not in ("png", "jpeg", "gif", "webp"):
        raise ValueError(f"Unsupported image format: {ext}")
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    return f"image/{ext}", b64


def _ingest_one(client, image_path: str) -> dict:
    """Send one screenshot to Claude vision, return parsed JSON dict."""
    media_type, b64 = _encode_image(image_path)
    msg = client.messages.create(
        model=VISION_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                }},
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
    )
    raw = ""
    for block in msg.content:
        if hasattr(block, "text"):
            raw += block.text
    raw = raw.strip()
    # Extract JSON
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return {"raw": raw, "parse_error": True}
    try:
        return json.loads(raw[start: end + 1])
    except json.JSONDecodeError:
        return {"raw": raw, "parse_error": True}


def ingest_source(slug: str, *, force: bool = False, dry_run: bool = False) -> dict:
    """Ingest all images for one source. Returns counts dict."""
    folder = os.path.join(SOURCES_DIR, slug, "screenshots")
    if not os.path.isdir(folder):
        return {"error": f"no screenshots folder for source '{slug}'"}

    # Load existing descriptions
    desc_path = os.path.join(folder, "descriptions.yaml")
    data = _read_yaml(desc_path) or {}
    existing = list(data.get("screenshots", []) or [])
    by_file = {e["file"]: e for e in existing if isinstance(e, dict) and "file" in e}

    images = sorted(fn for fn in os.listdir(folder)
                    if fn.lower().endswith((".png", ".jpg", ".jpeg")))
    if not images:
        return {"slug": slug, "n_images": 0, "skipped": 0, "ingested": 0,
                 "no_images": True}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"slug": slug, "error": "ANTHROPIC_API_KEY not set"}

    if dry_run:
        return {"slug": slug, "n_images": len(images), "dry_run": True}

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    counts = {"slug": slug, "n_images": len(images),
              "skipped": 0, "ingested": 0, "errors": 0}

    for fn in images:
        if not force and fn in by_file and "description" in by_file[fn]:
            counts["skipped"] += 1
            continue
        path = os.path.join(folder, fn)
        try:
            parsed = _ingest_one(client, path)
        except Exception as exc:
            counts["errors"] += 1
            by_file[fn] = {
                "file": fn,
                "ingest_error": f"{type(exc).__name__}: {exc}",
            }
            continue
        if "parse_error" in parsed:
            counts["errors"] += 1
            by_file[fn] = {
                "file": fn,
                "ingest_error": "json parse failed",
                "raw": parsed.get("raw", ""),
            }
            continue
        by_file[fn] = {
            "file": fn,
            "patterns": parsed.get("patterns", []),
            "timeframe": parsed.get("timeframe", ""),
            "symbol": parsed.get("symbol", ""),
            "description": parsed.get("description", ""),
            "teaching_point": parsed.get("teaching_point", ""),
        }
        counts["ingested"] += 1

    # Write back, preserving ordering by filename
    new_list = [by_file[fn] for fn in sorted(by_file.keys())]
    _write_yaml(desc_path, {"screenshots": new_list})
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest screenshots into the knowledge base.")
    parser.add_argument("--source", help="Slug of one source to process (default: all)")
    parser.add_argument("--force", action="store_true",
                         help="Re-ingest even if description already exists.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Don't call API, just print what would happen.")
    args = parser.parse_args()

    if args.source:
        slugs = [args.source]
    else:
        slugs = [s.slug for s in load_sources()]

    if not slugs:
        print("No sources to ingest.", file=sys.stderr)
        return 0

    total = {"sources": 0, "ingested": 0, "skipped": 0, "errors": 0}
    for slug in slugs:
        result = ingest_source(slug, force=args.force, dry_run=args.dry_run)
        print(f"[{slug}] {result}")
        if "ingested" in result:
            total["sources"] += 1
            total["ingested"] += result["ingested"]
            total["skipped"] += result["skipped"]
            total["errors"] += result.get("errors", 0)
    print(f"\nTotal: {total}")
    return 0 if total["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
