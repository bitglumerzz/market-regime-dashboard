"""Wave-knowledge base — expert source material compiled into Claude prompts.

The base lives under ``knowledge_base/`` (bind-mounted from the host):

  knowledge_base/
  ├── patterns.yaml          ← pattern cards (impulse, zigzag, flat, …)
  ├── trading_rules.yaml     ← which waves to trade, 1-2 setup, stop rules
  └── sources/
      └── <slug>/
          ├── source.yaml    ← author / title / publisher / type
          ├── transcript.txt ← optional
          └── screenshots/
              ├── *.png
              └── descriptions.yaml  ← per-image annotations

What this module gives the rest of the codebase:

  * ``load_sources()``  — list of every source with metadata
  * ``load_patterns()`` — pattern cards as dicts
  * ``load_trading_rules()`` — system-level rules dict
  * ``build_knowledge_prompt(max_chars=8000)`` — compact Markdown block,
    ready to be glued into Claude's system prompt. Filters down patterns
    by ``focus_patterns`` (slugs) when given.
  * ``screenshot_descriptions(source_slug)`` — annotations for vision
    few-shot or for the UI page.

Design choices:

  * **Always compile fresh.** Reading a few YAML files is cheap; we don't
    cache and we don't gate on Streamlit. This keeps editing-from-Finder
    workflow snappy.
  * **Compact compile is text-only.** Vision few-shot (sending screenshots
    to Claude) is handled separately by the caller — kept out of this
    module so token-cost stays predictable.
  * **Tolerant to missing pieces.** Source folder without source.yaml is
    skipped with a warning, not a crash.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover — pyyaml is a hard dependency
    yaml = None


# --------------------------------------------------------------------------
# Storage paths (bind mount: ./knowledge_base:/app/knowledge_base)
# --------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = "knowledge_base"
PATTERNS_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "patterns.yaml")
TRADING_RULES_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "trading_rules.yaml")
TRADING_EXAMPLES_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "trading_examples.yaml")
SOURCES_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "sources")


# --------------------------------------------------------------------------
# YAML helpers
# --------------------------------------------------------------------------
def _read_yaml(path: str) -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML not installed — knowledge base unavailable.")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return None


def _write_yaml(path: str, data: Any) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML not installed — cannot write knowledge.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
@dataclass
class Source:
    slug: str
    title: str = ""
    author: str = ""
    publisher: str = ""
    type: str = "video"
    language: str = "ru"
    url: str = ""
    year: int | None = None
    quality_tier: int = 2
    summary: str = ""
    key_takeaways: list[str] = field(default_factory=list)
    recommended_books: list[dict] = field(default_factory=list)
    path: str = ""       # absolute folder path
    raw: dict = field(default_factory=dict)


def load_sources() -> list[Source]:
    """List every source folder under ``knowledge_base/sources/``."""
    if not os.path.isdir(SOURCES_DIR):
        return []
    out: list[Source] = []
    for name in sorted(os.listdir(SOURCES_DIR)):
        folder = os.path.join(SOURCES_DIR, name)
        if not os.path.isdir(folder):
            continue
        src_yaml = os.path.join(folder, "source.yaml")
        data = _read_yaml(src_yaml) or {}
        slug = data.get("slug") or name
        out.append(Source(
            slug=slug,
            title=data.get("title", ""),
            author=data.get("author", ""),
            publisher=data.get("publisher", ""),
            type=data.get("type", "video"),
            language=data.get("language", "ru"),
            url=data.get("url", ""),
            year=data.get("year"),
            quality_tier=int(data.get("quality_tier", 2)),
            summary=data.get("summary", ""),
            key_takeaways=list(data.get("key_takeaways", []) or []),
            recommended_books=list(data.get("recommended_books", []) or []),
            path=folder,
            raw=data,
        ))
    return out


def load_patterns() -> list[dict]:
    """Pattern cards from patterns.yaml. Empty list if file is missing."""
    data = _read_yaml(PATTERNS_PATH)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "patterns" in data:
        return list(data["patterns"])
    return []


def load_trading_rules() -> dict:
    """System-level trading rules from trading_rules.yaml."""
    data = _read_yaml(TRADING_RULES_PATH)
    return data if isinstance(data, dict) else {}


def load_trading_examples() -> dict:
    """Real-world trade examples + setup templates from trading_examples.yaml.

    Returns ``{"examples": [...], "setup_templates": [...]}``.
    """
    data = _read_yaml(TRADING_EXAMPLES_PATH)
    if not isinstance(data, dict):
        return {"examples": [], "setup_templates": []}
    return {
        "examples":        list(data.get("examples", []) or []),
        "setup_templates": list(data.get("setup_templates", []) or []),
    }


def screenshot_descriptions(source_slug: str) -> list[dict]:
    """Per-screenshot annotations for a source. Empty if file is missing."""
    folder = os.path.join(SOURCES_DIR, source_slug, "screenshots")
    yml = os.path.join(folder, "descriptions.yaml")
    data = _read_yaml(yml) or {}
    return list(data.get("screenshots", []) or [])


def transcript_for(source_slug: str) -> str:
    """Full transcript text if available."""
    path = os.path.join(SOURCES_DIR, source_slug, "transcript.txt")
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


# --------------------------------------------------------------------------
# Prompt compilation — the single most important thing this module does
# --------------------------------------------------------------------------
def _format_pattern_card(p: dict, *, compact: bool = True) -> str:
    """One pattern → a short Markdown card."""
    ru = p.get("ru_name") or p.get("slug", "?")
    en = p.get("en_name", "")
    waves = p.get("waves", "")
    head = f"**{ru}** ({en}) — структура {waves}"
    lines: list[str] = [head]

    rules = p.get("rules_strict") or []
    if rules:
        lines.append("ЖЁСТКИЕ ПРАВИЛА:")
        for r in rules:
            lines.append(f"  • {r}")

    if not compact:
        guide = p.get("guidelines_soft") or []
        if guide:
            lines.append("GUIDELINES:")
            for g in guide:
                lines.append(f"  • {g}")

    ext = p.get("common_extensions") or []
    if ext:
        for e in ext:
            lines.append(f"  ↳ {e}")

    ta = p.get("trading_advice") or {}
    if ta:
        primary = ta.get("primary_trade")
        if primary:
            lines.append(f"  ▶ PRIMARY TRADE: {primary}")
        do_not = ta.get("do_not_trade")
        if do_not:
            lines.append(f"  ✗ НЕ ТОРГОВАТЬ: {', '.join(do_not)}")
        entry = ta.get("entry_model")
        if entry:
            lines.append(f"  ⊕ ENTRY: {entry}")
        stop = ta.get("stop_placement")
        if stop:
            lines.append(f"  ⊗ STOP: {stop}")

    return "\n".join(lines)


def _format_trading_rules(rules: dict) -> str:
    """Trading-rules section as compact Markdown."""
    out: list[str] = []
    sys = rules.get("system")
    if sys:
        out.append(f"### Trading system: {sys}")

    tradable = rules.get("tradable_waves") or []
    if tradable:
        out.append("\n**Торгуемые волны (PRIMARY):**")
        for w in tradable:
            conf = w.get("confidence", "")
            conf_tag = f" ({conf})" if conf else ""
            out.append(f"  ✓ {w.get('location','?')}{conf_tag} — {w.get('why','')}")

    non_tradable = rules.get("non_tradable_waves") or []
    if non_tradable:
        out.append("\n**НЕ торгуемые волны:**")
        for w in non_tradable:
            out.append(f"  ✗ {w.get('location','?')} — {w.get('why','')}")

    entry = rules.get("primary_entry_model") or {}
    if entry:
        out.append(f"\n**Primary entry model: «{entry.get('name','?')}»**")
        desc = entry.get("description")
        if desc:
            out.append(desc.strip())
        pre = entry.get("preconditions") or []
        if pre:
            out.append("Preconditions:")
            for p in pre:
                out.append(f"  • {p}")
        out.append(f"  Entry: {entry.get('entry','—')}")
        out.append(f"  Stop:  {entry.get('stop','—')}")
        out.append(f"  R/R target: {entry.get('initial_risk_reward_target','—')}")

    sm = rules.get("stop_management") or {}
    if sm:
        out.append("\n**Stop management:**")
        for r in sm.get("rules", []) or []:
            out.append(f"  • {r}")

    mistakes = rules.get("common_mistakes") or []
    if mistakes:
        out.append("\n**Frequent mistakes to avoid:**")
        for m in mistakes:
            out.append(f"  ⚠️ {m.get('mistake','?')} → {m.get('fix','')}")

    return "\n".join(out)


def _format_sources_list(sources: list[Source]) -> str:
    """Short bibliography for the prompt."""
    if not sources:
        return ""
    lines = ["### Reference sources used in this analysis:"]
    for s in sources:
        bits = [s.title or s.slug]
        if s.author:
            bits.append(f"({s.author}")
            if s.publisher:
                bits[-1] += f", {s.publisher}"
            bits[-1] += ")"
        tier_tag = f" [tier-{s.quality_tier}]" if s.quality_tier else ""
        lines.append(f"  - {' '.join(bits)}{tier_tag}")
    return "\n".join(lines)


def _format_trading_examples(examples: list[dict], setup_templates: list[dict],
                              *, max_examples: int = 6) -> str:
    """Compact trade-example block — few-shot learning material."""
    if not examples and not setup_templates:
        return ""
    out: list[str] = []

    if examples:
        out.append("### 🎯 Real trade examples (few-shot reference):")
        for ex in examples[:max_examples]:
            head = (f"**{ex.get('symbol','?')} {ex.get('timeframe','?')} "
                    f"{ex.get('direction','?').upper()}**")
            r = ex.get("final_result_r")
            if r:
                head += f" — итог: ×{r}R"
            out.append(head)
            pat = ex.get("pattern_observed")
            if pat:
                out.append(f"Паттерн: `{pat}`")
            # Compress steps to single-line summary
            steps = ex.get("setup_steps") or []
            for s in steps:
                out.append(f"  {s.get('step','?')}. {s.get('action','')}")
            ts = ex.get("teaching_summary")
            if ts:
                out.append(ts.strip())
            out.append("")  # spacer

    if setup_templates:
        out.append("### 📋 Setup templates (geometry shortcuts):")
        for tpl in setup_templates:
            out.append(f"**{tpl.get('name','?')}** ({tpl.get('direction','?')})")
            wtu = tpl.get("when_to_use")
            if wtu:
                out.append(f"When: {wtu}")
            geo = tpl.get("geometry") or []
            for g in geo[:5]:
                out.append(f"  • {g}")
            inv = tpl.get("invalidation") or []
            if inv:
                out.append(f"Invalidation: {'; '.join(inv[:2])}")
            out.append("")

    return "\n".join(out)


def build_knowledge_prompt(
    *,
    focus_patterns: Iterable[str] | None = None,
    include_trading_rules: bool = True,
    include_sources_list: bool = True,
    include_examples: bool = True,
    max_chars: int = 22000,
) -> str:
    """Compile a Markdown block ready to embed in Claude's system prompt.

    Parameters
    ----------
    focus_patterns : Iterable[str] or None
        If given, only pattern cards whose slug appears here will be included
        (saves tokens when the analysis is about a known set of patterns).
    include_trading_rules : bool
        Whether to embed the trading-system block (when/where to trade).
    include_sources_list : bool
        Whether to embed the bibliography line at the end.
    max_chars : int
        Soft cap — if the assembled block exceeds this, pattern cards are
        emitted in compact form. Hard truncation at the end as last resort.
    """
    patterns = load_patterns()
    if focus_patterns:
        wanted = {p.lower() for p in focus_patterns}
        patterns = [p for p in patterns
                    if (p.get("slug") or "").lower() in wanted]

    rules = load_trading_rules() if include_trading_rules else {}
    sources = load_sources()

    parts: list[str] = [
        "## 🌊 Reference material — Elliott Wave knowledge base",
        "Use these expert-derived rules as your primary playbook. They reflect "
        "consensus from professional wave traders (Pavelko/FxPro, Prechter, "
        "Voznoy). When the rules below conflict with your prior knowledge, "
        "PREFER THE RULES BELOW."
    ]

    if patterns:
        parts.append("\n### Pattern reference cards:")
        # First pass — compact mode
        for p in patterns:
            parts.append(_format_pattern_card(p, compact=True))

    if rules:
        parts.append("\n" + _format_trading_rules(rules))

    if include_examples:
        ex_data = load_trading_examples()
        block = _format_trading_examples(
            ex_data["examples"], ex_data["setup_templates"],
        )
        if block:
            parts.append("\n" + block)

    if include_sources_list and sources:
        parts.append("\n" + _format_sources_list(sources))

    blob = "\n\n".join(parts)
    if len(blob) <= max_chars:
        return blob

    # Hard truncate as last resort
    return blob[:max_chars - 60] + "\n…[truncated by build_knowledge_prompt]"


# --------------------------------------------------------------------------
# Stats / introspection
# --------------------------------------------------------------------------
def stats() -> dict:
    """Quick overview of what's in the base."""
    sources = load_sources()
    patterns = load_patterns()
    rules = load_trading_rules()
    examples_data = load_trading_examples()

    by_type: dict[str, int] = {}
    by_language: dict[str, int] = {}
    total_screenshots = 0
    annotated_screenshots = 0
    for s in sources:
        by_type[s.type] = by_type.get(s.type, 0) + 1
        by_language[s.language] = by_language.get(s.language, 0) + 1
        descs = screenshot_descriptions(s.slug)
        total_screenshots += len(descs)
        sf = os.path.join(s.path, "screenshots")
        if os.path.isdir(sf):
            for fn in os.listdir(sf):
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    annotated_screenshots += 1
    return {
        "n_sources":           len(sources),
        "n_patterns":          len(patterns),
        "has_trading_rules":   bool(rules),
        "n_trade_examples":    len(examples_data["examples"]),
        "n_setup_templates":   len(examples_data["setup_templates"]),
        "by_type":             by_type,
        "by_language":         by_language,
        "n_screenshot_annotations": total_screenshots,
        "n_screenshot_files":      annotated_screenshots,
    }


# --------------------------------------------------------------------------
# Pattern → focus_patterns helpers (used by wave_ai for RAG-style narrowing)
# --------------------------------------------------------------------------
def slugs_for_candidates(candidates: Iterable[Any]) -> set[str]:
    """Map ``elliott.WaveCandidate.pattern`` values to ``patterns.yaml`` slugs.

    The wave-engine emits names like 'impulse_up' or 'leading_diagonal_down'.
    Our pattern cards generalize those (one card per pattern type, regardless
    of direction). This helper does that normalization.
    """
    out: set[str] = set()
    direction_suffixes = ("_up", "_down")
    for c in candidates or []:
        pat = getattr(c, "pattern", None) or (c.get("pattern") if isinstance(c, dict) else None)
        if not pat:
            continue
        pat = str(pat).lower()
        # impulse_up → impulse_up (keep both directions as separate cards in yaml)
        out.add(pat)
        # leading_diagonal_down → leading_diagonal
        for suf in direction_suffixes:
            if pat.endswith(suf):
                out.add(pat[: -len(suf)])
        # correction_up → zigzag_correction (common rename)
        if pat in ("correction_up", "correction_down"):
            out.add("zigzag_correction")
        # complex_correction_up → double_three
        if pat in ("complex_correction_up", "complex_correction_down"):
            out.add("double_three")
        # flat_up/flat_down → flat_correction + expanded_flat
        if pat in ("flat_up", "flat_down"):
            out.add("flat_correction")
            out.add("expanded_flat")
        # triangle_up/down → triangle_correction
        if pat in ("triangle_up", "triangle_down"):
            out.add("triangle_correction")
    return out
