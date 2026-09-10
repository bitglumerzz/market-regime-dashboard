"""Smoke tests for wave_knowledge.

Covers:
  * sources load successfully and parse YAML
  * patterns.yaml parses; each entry has required fields
  * trading_rules.yaml parses; key sections exist
  * build_knowledge_prompt produces a non-empty Markdown block under cap
  * focus_patterns narrowing works
  * empty-knowledge case doesn't crash
  * slugs_for_candidates maps elliott names correctly
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from types import SimpleNamespace


def setup_isolated_kb() -> str:
    """Point wave_knowledge at a fresh empty kb dir, return the path."""
    tmp = tempfile.mkdtemp(prefix="wk_test_")
    import wave_knowledge as wk
    wk.KNOWLEDGE_BASE_DIR = tmp
    wk.PATTERNS_PATH = os.path.join(tmp, "patterns.yaml")
    wk.TRADING_RULES_PATH = os.path.join(tmp, "trading_rules.yaml")
    wk.TRADING_EXAMPLES_PATH = os.path.join(tmp, "trading_examples.yaml")
    wk.SOURCES_DIR = os.path.join(tmp, "sources")
    return tmp


def test_real_knowledge_loads():
    """Against the actual knowledge_base/ in the repo, everything must load."""
    import wave_knowledge as wk
    sources = wk.load_sources()
    assert sources, "expected at least one source (pavelko_fxpro_advanced)"
    slugs = {s.slug for s in sources}
    assert "pavelko_fxpro_advanced" in slugs

    patterns = wk.load_patterns()
    assert len(patterns) >= 8, f"expected 8+ patterns, got {len(patterns)}"
    for p in patterns:
        assert p.get("slug"), f"pattern missing slug: {p}"
        # Reference/anomaly/modifier patterns may have empty rules_strict —
        # they're catalogs, not tradable structures. They must have guidelines instead.
        is_reference = (p.get("degree_group", "").endswith(("_reference",
                                                              "_modifier",
                                                              "_anomaly")) or
                        p.get("degree_group") == "anomaly_reference")
        if is_reference:
            assert p.get("guidelines_soft"), \
                f"reference pattern {p['slug']} has no guidelines_soft either"
        else:
            assert p.get("rules_strict"), \
                f"pattern {p['slug']} has no rules_strict"

    rules = wk.load_trading_rules()
    assert rules.get("tradable_waves"), "trading_rules.yaml missing tradable_waves"
    assert rules.get("primary_entry_model"), "missing primary_entry_model"
    print("  ✓ real knowledge_base parses fully")


def test_build_prompt_non_empty():
    import wave_knowledge as wk
    prompt = wk.build_knowledge_prompt()
    assert prompt, "expected non-empty prompt"
    assert "Elliott Wave knowledge base" in prompt
    assert "ЖЁСТКИЕ ПРАВИЛА" in prompt
    print(f"  ✓ prompt builds ({len(prompt):,} chars)")


def test_focus_narrowing():
    import wave_knowledge as wk
    full = wk.build_knowledge_prompt()
    narrow = wk.build_knowledge_prompt(focus_patterns=["impulse_up"])
    assert len(narrow) < len(full), \
        f"narrowed prompt should be smaller: {len(narrow)} vs {len(full)}"
    # Should still contain the focused pattern
    assert "Импульс вверх" in narrow or "impulse_up" in narrow.lower()
    print(f"  ✓ focus narrowing: full={len(full)} narrow={len(narrow)}")


def test_prompt_respects_max_chars():
    import wave_knowledge as wk
    p = wk.build_knowledge_prompt(max_chars=500)
    assert len(p) <= 500, f"prompt exceeded max_chars: {len(p)}"
    print("  ✓ max_chars respected")


def test_slugs_for_candidates():
    import wave_knowledge as wk
    cands = [
        SimpleNamespace(pattern="impulse_up"),
        SimpleNamespace(pattern="leading_diagonal_down"),
        SimpleNamespace(pattern="correction_up"),
        SimpleNamespace(pattern="flat_down"),
        SimpleNamespace(pattern="complex_correction_up"),
        SimpleNamespace(pattern="triangle_up"),
    ]
    slugs = wk.slugs_for_candidates(cands)
    expected = {"impulse_up", "leading_diagonal_down", "leading_diagonal",
                "correction_up", "zigzag_correction",
                "flat_down", "flat_correction", "expanded_flat",
                "complex_correction_up", "double_three",
                "triangle_up", "triangle_correction"}
    missing = expected - slugs
    assert not missing, f"missing slug mappings: {missing}"
    print(f"  ✓ slugs_for_candidates maps all variants ({len(slugs)} slugs)")


def test_empty_knowledge_dont_crash():
    tmp = setup_isolated_kb()
    try:
        import wave_knowledge as wk
        # Force the changes to take effect — reimport-safe operations
        assert wk.load_sources() == []
        assert wk.load_patterns() == []
        assert wk.load_trading_rules() == {}
        prompt = wk.build_knowledge_prompt()
        # Should still return at least the header (empty knowledge → minimal block)
        assert isinstance(prompt, str)
        st = wk.stats()
        assert st["n_sources"] == 0
        assert st["n_patterns"] == 0
        print("  ✓ empty knowledge graceful")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_screenshot_descriptions():
    import wave_knowledge as wk
    descs = wk.screenshot_descriptions("pavelko_fxpro_advanced")
    assert len(descs) >= 9, f"expected 9+ screenshot annotations, got {len(descs)}"
    # Every annotation must have file + description
    for d in descs:
        assert d.get("file"), f"missing file in {d}"
        assert d.get("description"), f"missing description in {d.get('file')}"
    print(f"  ✓ {len(descs)} screenshot descriptions loaded")


def test_stats_overview():
    import wave_knowledge as wk
    s = wk.stats()
    assert s["n_sources"] >= 1
    assert s["n_patterns"] >= 8
    assert s["has_trading_rules"] is True
    print(f"  ✓ stats: {s}")


def test_trading_examples_load():
    """trading_examples.yaml parses + has expected fields."""
    import wave_knowledge as wk
    data = wk.load_trading_examples()
    assert data["examples"], "expected at least one trade example"
    assert data["setup_templates"], "expected at least one setup template"
    ex = data["examples"][0]
    assert ex.get("symbol")
    assert ex.get("setup_steps")
    print(f"  ✓ {len(data['examples'])} examples + "
          f"{len(data['setup_templates'])} templates loaded")


def test_prompt_includes_examples_when_enabled():
    import wave_knowledge as wk
    with_ex = wk.build_knowledge_prompt(include_examples=True)
    without_ex = wk.build_knowledge_prompt(include_examples=False)
    assert len(with_ex) > len(without_ex), \
        f"prompt with examples should be longer: {len(with_ex)} vs {len(without_ex)}"
    # Must mention SILVER or USDCAD — the actual symbols from examples
    assert ("SILVER" in with_ex or "USDCAD" in with_ex), \
        "expected real trade symbols in prompt"
    print(f"  ✓ examples toggle works: with={len(with_ex)} without={len(without_ex)}")


def test_multiple_sources_in_prompt():
    """After adding 3 books, sources list should mention all of them."""
    import wave_knowledge as wk
    prompt = wk.build_knowledge_prompt(include_sources_list=True, max_chars=20000)
    sources = wk.load_sources()
    slugs = {s.slug for s in sources}
    expected = {"pavelko_fxpro_advanced", "prechter_wave_principle",
                "voznoy_elliott_code", "schwager_technical_analysis"}
    missing = expected - slugs
    assert not missing, f"missing sources: {missing}"
    # At least 2 author names should appear in the prompt
    author_hits = sum(1 for name in ("Pavelko", "Prechter", "Vozny", "Schwager",
                                       "Возный", "Прехтер", "Павелко", "Швагер",
                                       "Frost", "Roman")
                       if name.lower() in prompt.lower())
    assert author_hits >= 2, f"expected ≥2 author mentions, got {author_hits}"
    print(f"  ✓ {len(slugs)} sources registered, {author_hits} author mentions in prompt")


def test_wave_ai_imports_knowledge():
    """Ensure wave_ai.py picks up build_knowledge_prompt without crashing.

    Skipped in environments without streamlit (e.g. CI sandboxes); wave_ai
    imports streamlit at module top because it uses st.cache."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("  ⊘ wave_ai integration: skipped (streamlit not available in this env)")
        return
    import importlib
    import wave_ai
    importlib.reload(wave_ai)
    assert hasattr(wave_ai, "_system_prompt_with_knowledge")
    result = wave_ai._system_prompt_with_knowledge()
    assert result is None or isinstance(result, str)
    if result:
        assert "Reference material" in result or "Elliott" in result
        print(f"  ✓ wave_ai integration produces system block "
              f"({len(result):,} chars)")
    else:
        print("  ✓ wave_ai integration: knowledge optional (returned None)")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print("Running wave_knowledge smoke tests…")
    test_real_knowledge_loads()
    test_build_prompt_non_empty()
    test_focus_narrowing()
    test_prompt_respects_max_chars()
    test_slugs_for_candidates()
    test_screenshot_descriptions()
    test_stats_overview()
    test_trading_examples_load()
    test_prompt_includes_examples_when_enabled()
    test_multiple_sources_in_prompt()
    test_empty_knowledge_dont_crash()
    test_wave_ai_imports_knowledge()
    print("\nAll wave_knowledge smoke tests passed ✓")
