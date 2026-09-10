"""MCP-style markdown report builder for holistic Claude opinions.

Renders the full ``st.session_state['last_holistic_full']`` dict as a single
long markdown document in the visual style of TradingView MCP chat responses
(emoji-headed sections, bullet lists, tables). This is the Phase 32 visual
overhaul — replaces the previous widget-grid layout with a coherent narrative
report similar to what the user sees from TV MCP in chat.

Usage:

    from holistic_report import build_mcp_style_report
    md = build_mcp_style_report(st.session_state["last_holistic_full"])
    st.markdown(md, unsafe_allow_html=True)
"""
from __future__ import annotations

from typing import Any

try:
    from translation_helpers import translate_detector_text as _tr
except Exception:
    def _tr(s: str) -> str:
        return s


def _fmt_num(v: Any, decimals: int = 4) -> str:
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v.is_integer():
            return f"{int(v):,}"
        if abs(v) >= 100:
            return f"{v:,.2f}"
        return f"{v:,.{decimals}f}"
    if v is None:
        return "—"
    return str(v)


def _section(title: str, body: str | None) -> str:
    """Wrap a section with an emoji header. Skip if body is empty."""
    if not body or not body.strip():
        return ""
    return f"\n\n## {title}\n\n{body.strip()}\n"


def _bias_header(h: dict) -> str:
    bias = (h.get("bias") or "neutral").lower()
    action = (h.get("action") or "wait").lower()
    sq = h.get("setup_quality", "low")
    bias_emoji = {"long": "🟢", "short": "🔴", "neutral": "⚪️"}.get(bias, "⚪️")
    action_emoji = {"long": "▶ ЛОНГ", "short": "▶ ШОРТ",
                    "wait": "⏸ WAIT", "close": "✖ CLOSE"}.get(action, action.upper())
    summary = _tr(h.get("summary", ""))
    truncated_warning = ""
    if h.get("was_truncated"):
        truncated_warning = (
            "\n\n> ⚠️ **Ответ Claude был обрезан** (max_tokens=2500). "
            "Восстановлено через tolerant парсер — главные поля на месте, но "
            "часть scenarios/comparison может отсутствовать. Для полного "
            "ответа измени threshold-slider и перезапусти."
        )
    return (
        f"# {bias_emoji} BIAS: {bias.upper()} · {action_emoji} · качество: {sq}\n\n"
        f"> {summary}{truncated_warning}"
    )


def _trade_levels(h: dict) -> str:
    entry = h.get("entry")
    stop = h.get("stop_loss")
    target = h.get("target")
    rr = h.get("risk_reward")
    inv = h.get("invalidation_explained") or ""

    lines = ["| Параметр | Значение |",
             "|---|---|"]
    if entry is not None:
        lines.append(f"| 🎯 **Entry** | `{_fmt_num(entry)}` |")
    if stop is not None:
        lines.append(f"| ⊗ **Stop** | `{_fmt_num(stop)}` |")
    if target is not None:
        lines.append(f"| 🏁 **Target** | `{_fmt_num(target)}` |")
    if rr is not None:
        lines.append(f"| 📊 **R/R** | `{_fmt_num(rr, 2)}` |")

    body = "\n".join(lines)
    if inv:
        body += f"\n\n**⚠ Инвалидация:** {_tr(inv)}"
    return body


def _action_plan(h: dict) -> str:
    steps = h.get("action_plan") or []
    if not steps:
        return ""
    lines = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        n = s.get("step", "?")
        act = s.get("action", "")
        if act:
            lines.append(f"{n}. {_tr(act)}")
    if not lines:
        return ""
    return "\n".join(lines)


def _structural_context(h: dict) -> str:
    parts = []
    sp = h.get("structure_position") or ""
    if sp:
        parts.append(f"**📍 Структурный контекст:** {sp}")
    rc = h.get("rules_check") or ""
    if rc:
        parts.append(f"**✅ Валидация R0-R3:** {rc}")
    alt = h.get("alternation_check") or ""
    if alt:
        parts.append(f"**🔄 Альтернация:** {alt}")
    fib = h.get("fibonacci_check") or ""
    if fib:
        parts.append(f"**📐 Fibonacci:** {fib}")
    alt_int = h.get("alternative_interpretation") or ""
    if alt_int:
        parts.append(f"**⚠ Альтернативная разметка:** {alt_int}")
    return "\n\n".join(parts)


def _per_tf_notes(h: dict) -> str:
    notes = h.get("per_tf_notes") or {}
    if not isinstance(notes, dict) or not notes:
        return ""
    order = ["1d", "4h", "15m", "5m"]
    lines = ["| TF | Заметки |", "|---|---|"]
    for tf in order:
        v = notes.get(tf)
        if v:
            lines.append(f"| **{tf}** | {v} |")
    return "\n".join(lines)


def _confidence_risk(h: dict) -> str:
    cf = h.get("confidence_factors") or []
    rf = h.get("risk_factors") or []
    out = []
    if cf:
        out.append("**✅ За сетап:**\n" + "\n".join(f"- {x}" for x in cf))
    if rf:
        out.append("**❌ Против сетапа:**\n" + "\n".join(f"- {x}" for x in rf))
    return "\n\n".join(out)


def _entry_alternatives(h: dict) -> str:
    alts = h.get("entry_alternatives") or []
    if not alts:
        return ""
    lines = ["| Версия | Entry | Stop | Risk | Обоснование |",
             "|---|---|---|---|---|"]
    for a in alts:
        if not isinstance(a, dict):
            continue
        lines.append(
            f"| **{a.get('name', '—')}** | "
            f"`{_fmt_num(a.get('entry'))}` | "
            f"`{_fmt_num(a.get('stop'))}` | "
            f"`{_fmt_num(a.get('risk'), 2)}` | "
            f"{a.get('rationale', '—')} |"
        )
    return "\n".join(lines) if len(lines) > 2 else ""


def _entry_variants_md(h: dict) -> str:
    """Phase 42 — three-tier entry variants (Aggressive/Confirmed/Breakout).
    Phase 44 — also tolerates legacy tight/wider format (risk vs risk_pct,
    no trigger field) so old reference examples still render correctly.
    """
    variants = h.get("entry_variants") or []
    if not variants:
        return ""
    lines = ["| Вариант | Entry | Stop | Risk | R/R | Триггер | Обоснование |",
             "|---|---|---|---|---|---|---|"]
    for v in variants:
        if not isinstance(v, dict):
            continue
        # Risk: prefer risk_pct, fall back to risk (abs amount)
        risk_pct = v.get("risk_pct")
        risk_abs = v.get("risk")
        if isinstance(risk_pct, (int, float)):
            risk_str = f"{risk_pct}%"
        elif isinstance(risk_abs, (int, float)):
            risk_str = f"`{_fmt_num(risk_abs, 2)}`"
        else:
            risk_str = "—"
        # R/R: prefer 'rr', fallback to '—'
        rr = v.get("rr")
        rr_str = f"`{_fmt_num(rr, 2)}`" if isinstance(rr, (int, float)) else "—"
        trigger = v.get("trigger") or v.get("trigger_condition") or "—"
        lines.append(
            f"| **{v.get('name', '—')}** | "
            f"`{_fmt_num(v.get('entry'))}` | "
            f"`{_fmt_num(v.get('stop'))}` | "
            f"{risk_str} | "
            f"{rr_str} | "
            f"{_tr(str(trigger))} | "
            f"{_tr(str(v.get('rationale', '—')))} |"
        )
    return "\n".join(lines) if len(lines) > 2 else ""


def _target_pyramid(h: dict) -> str:
    targets = h.get("target_pyramid") or []
    if not targets:
        return ""
    lines = ["| Target | Цена | R/R tight | R/R wider |",
             "|---|---|---|---|"]
    for t in targets:
        if not isinstance(t, dict):
            continue
        lines.append(
            f"| **{t.get('name', '—')}** | "
            f"`{_fmt_num(t.get('price'))}` | "
            f"`{_fmt_num(t.get('rr_tight'), 2)}` | "
            f"`{_fmt_num(t.get('rr_wider'), 2)}` |"
        )
    return "\n".join(lines) if len(lines) > 2 else ""


def _scenarios(h: dict) -> str:
    scenarios = h.get("scenarios") or []
    if not scenarios:
        return ""
    parts = []
    for sc in scenarios:
        if not isinstance(sc, dict):
            continue
        name = sc.get("name", "—")
        prob = sc.get("probability_pct")
        prob_str = f" ({prob}%)" if isinstance(prob, (int, float)) else ""
        parts.append(f"### {name}{prob_str}")
        if sc.get("trigger"):
            parts.append(f"**🚨 Триггер:** {_tr(str(sc['trigger']))}")
        if sc.get("action"):
            parts.append(f"**▶ Действие:** {_tr(str(sc['action']))}")
        ez = sc.get("entry_zone")
        if ez:
            parts.append(f"**🎯 Entry zone:** `{ez}`")
        stop = sc.get("stop")
        if stop is not None:
            parts.append(f"**⊗ Stop:** `{_fmt_num(stop)}`")
        tgts = sc.get("targets")
        if tgts:
            parts.append(f"**🏁 Targets:** {tgts}")
        parts.append("")  # blank line between scenarios
    return "\n\n".join(parts)


def _waiting_conditions(h: dict) -> str:
    waiting = h.get("waiting_conditions") or []
    if not waiting:
        return ""
    lines = []
    for w in waiting:
        if isinstance(w, dict):
            step = w.get("step", "?")
            cond = w.get("condition", "")
            if cond:
                lines.append(f"{step}. {cond}")
    return "\n".join(lines)


def _higher_degree(h: dict) -> str:
    hdc = h.get("higher_degree_context") or {}
    if not isinstance(hdc, dict) or not any(hdc.values()):
        return ""
    parts = []
    if hdc.get("cycle_peak"):
        parts.append(f"**🔝 Cycle peak:** {hdc['cycle_peak']}")
    if hdc.get("cycle_low"):
        parts.append(f"**📉 Cycle low:** {hdc['cycle_low']}")
    if hdc.get("current_phase"):
        parts.append(f"**📍 Current phase:** {hdc['current_phase']}")
    return "\n\n".join(parts)


def _monitoring_matrix(h: dict) -> str:
    monitoring = h.get("monitoring_levels") or []
    if not monitoring:
        return ""
    lines = ["| 🚨 Триггер | 📋 Сценарий | ▶ Действие |",
             "|---|---|---|"]
    for m in monitoring:
        if not isinstance(m, dict):
            continue
        lines.append(
            f"| {m.get('trigger', '—')} | "
            f"{m.get('scenario', '—')} | "
            f"{m.get('action', '—')} |"
        )
    return "\n".join(lines) if len(lines) > 2 else ""


def _cross_correlation(h: dict) -> str:
    parts = []
    cc = h.get("cross_correlation_notes") or ""
    if cc:
        parts.append(cc)
    lfn = h.get("leader_follower_note") or ""
    if lfn:
        parts.append(f"**👑 Leader/Follower:** {lfn}")
    cac = h.get("cross_asset_comparison") or []
    if cac:
        lines = ["", "| Параметр | Primary | Secondary |", "|---|---|---|"]
        for row in cac:
            if isinstance(row, dict):
                lines.append(
                    f"| **{row.get('parameter', '—')}** | "
                    f"{row.get('primary', '—')} | "
                    f"{row.get('secondary', '—')} |"
                )
        if len(lines) > 3:
            parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _period_context(h: dict) -> str:
    """Phase 34 — context header at top of MCP-style report."""
    pc = h.get("period_context") or {}
    if not isinstance(pc, dict) or not pc:
        return ""
    parts = []
    if pc.get("current_price") is not None:
        parts.append(
            f"**Текущая цена:** `${_fmt_num(pc['current_price'])}` "
            f"({pc.get('current_time', '—')})"
        )
    if pc.get("today_low") is not None:
        parts.append(
            f"**Сегодняшний low:** `${_fmt_num(pc['today_low'])}` "
            f"({pc.get('today_low_time', '—')})"
        )
    if pc.get("period_low") is not None:
        parts.append(
            f"**Period low ({pc.get('period_window', '—')}):** "
            f"`${_fmt_num(pc['period_low'])}` "
            f"({pc.get('period_low_time', '—')})"
        )
    if pc.get("period_high") is not None:
        parts.append(
            f"**Period high ({pc.get('period_window', '—')}):** "
            f"`${_fmt_num(pc['period_high'])}` "
            f"({pc.get('period_high_time', '—')})"
        )
    return "\n\n".join(parts)


def _fib_table_md(fib: dict) -> str:
    """Render Phase 34 fib_table dict as markdown table with Status column."""
    if not isinstance(fib, dict) or not fib.get("rows"):
        return ""
    lines = []
    rf = fib.get("range_from")
    rt = fib.get("range_to")
    if rf is not None and rt is not None:
        lines.append(
            f"_Range: `${_fmt_num(rf)}` → `${_fmt_num(rt)}` "
            f"(range $`{_fmt_num(fib.get('range_total', '—'))}`)_\n"
        )
    lines.append("| Level | Price | Status |")
    lines.append("|---|---|---|")
    for row in fib["rows"]:
        if isinstance(row, dict):
            lines.append(
                f"| {row.get('level', '—')} | "
                f"`${_fmt_num(row.get('price'))}` | "
                f"{row.get('status', '')} |"
            )
    if fib.get("current_position"):
        lines.append(f"\n**Текущая позиция:** {fib['current_position']}")
    return "\n".join(lines)


def _impulse_decomposition_md(imps: list) -> str:
    """Render impulse_decomposition list as labelled sub-sections."""
    if not isinstance(imps, list) or not imps:
        return ""
    parts = []
    for imp in imps:
        if not isinstance(imp, dict):
            continue
        if imp.get("label"):
            parts.append(f"### {imp['label']}")
        waves = imp.get("waves") or []
        if waves:
            lines = ["| Wave | Time | Price | Length | Retrace | Wave ratio | Note |",
                     "|---|---|---|---|---|---|---|"]
            for w in waves:
                if isinstance(w, dict):
                    lines.append(
                        f"| **{w.get('wave', '—')}** | "
                        f"{w.get('time', '—')} | "
                        f"`${_fmt_num(w.get('price'))}` | "
                        f"`{_fmt_num(w.get('length_usd'), 2) if w.get('length_usd') is not None else '—'}` | "
                        f"{(str(w['retrace_pct']) + '%') if w.get('retrace_pct') is not None else '—'} | "
                        f"{w.get('wave_ratio', '—')} | "
                        f"{w.get('note', '')} |"
                    )
            parts.append("\n".join(lines))
        validation = imp.get("validation") or []
        if validation:
            parts.append("**Валидация:**\n" + "\n".join(f"- {v}" for v in validation))
        if imp.get("playbook_outcome"):
            parts.append(f"**Playbook outcome:** {imp['playbook_outcome']}")
    return "\n\n".join(parts)


def _wave_projections_md(projs: list) -> str:
    if not isinstance(projs, list) or not projs:
        return ""
    lines = ["| Projection | Price | Formula |", "|---|---|---|"]
    for p in projs:
        if isinstance(p, dict):
            lines.append(
                f"| **{p.get('name', '—')}** | "
                f"`${_fmt_num(p.get('price'))}` | "
                f"{p.get('formula', '—')} |"
            )
    return "\n".join(lines) if len(lines) > 2 else ""


def _tf_analysis(h: dict) -> str:
    """Phase 34 — per-TF detailed sections with wave_trace + fib_table + projections."""
    tfa = h.get("tf_analysis") or {}
    if not isinstance(tfa, dict) or not tfa:
        return ""
    parts = []
    for tf in ("1h", "4h", "15m", "5m"):
        details = tfa.get(tf)
        if not isinstance(details, dict) or not details:
            continue
        sec = []
        title = details.get("section_title") or f"📊 {tf} TF"
        sec.append(f"### {title}")
        if details.get("thesis"):
            sec.append(f"_{details['thesis']}_")
        wta = details.get("wave_trace_ascii") or details.get("bounce_trace_ascii")
        if wta:
            sec.append(f"```\n{wta}\n```")
        fib = details.get("fib_table") or {}
        fib_md = _fib_table_md(fib)
        if fib_md:
            sec.append("**📐 Fibonacci retrace:**\n\n" + fib_md)
        if details.get("pavelko_interpretation"):
            sec.append("**🎯 Pavelko interpretation:**\n\n" + details["pavelko_interpretation"])
        projs = details.get("wave_projections") or []
        projs_md = _wave_projections_md(projs)
        if projs_md:
            sec.append("**🚀 Wave projections:**\n\n" + projs_md)
        imps = details.get("impulse_decomposition") or []
        imps_md = _impulse_decomposition_md(imps)
        if imps_md:
            sec.append("**📈 Impulse decomposition:**\n\n" + imps_md)
        if details.get("warning"):
            sec.append(f"⚠ {details['warning']}")
        if details.get("contrary_setup_filter"):
            sec.append(f"**🚫 Контр-trade filter:** {details['contrary_setup_filter']}")
        parts.append("\n\n".join(sec))
    return "\n\n---\n\n".join(parts)


def _verdict_per_tf(h: dict) -> str:
    """Phase 34 — summary table per TF."""
    vp = h.get("verdict_per_tf") or {}
    if not isinstance(vp, dict):
        return ""
    headers = vp.get("headers") or ["TF", "Структура", "Сигнал", "Trade"]
    rows = vp.get("rows") or []
    if not rows:
        return ""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "---|" * len(headers)]
    for row in rows:
        if isinstance(row, list):
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) if len(lines) > 2 else ""


def _confluence_factors_md(h: dict) -> str:
    cf = h.get("confluence_factors") or []
    if not cf:
        return ""
    return "\n".join(f"- {x}" for x in cf)


def _comparison_with_previous(h: dict) -> str:
    """Phase 34 — was/now reframing table."""
    cwp = h.get("comparison_with_previous") or {}
    if not isinstance(cwp, dict) or not cwp.get("rows"):
        return ""
    headers = cwp.get("headers") or ["Metric", "Was", "Now"]
    rows = cwp.get("rows") or []
    parts = []
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "---|" * len(headers)]
    for row in rows:
        if isinstance(row, list):
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
    if len(lines) > 2:
        parts.append("\n".join(lines))
    if cwp.get("key_change"):
        parts.append(f"\n**Ключевой сдвиг:** {_tr(str(cwp['key_change']))}")
    return "\n\n".join(parts)


def _entry_zone_shift_md(h: dict) -> str:
    """Phase 41 — render entry_zone_shift if previous holistic existed."""
    ezs = h.get("entry_zone_shift") or ""
    if not ezs.strip():
        return ""
    return f"**🎯 Сдвиг зоны входа:** {_tr(str(ezs))}"


def _events_timeline_md(h: dict) -> str:
    """Phase 47 — structural events timeline (Breakout/Retest/Bounce/Current)."""
    events = h.get("structural_events_timeline") or []
    if not isinstance(events, list) or not events:
        return ""
    lines = ["| Событие | Время | Цена | Тип |",
             "|---|---|---|---|"]
    for ev in events:
        if not isinstance(ev, dict):
            continue
        kind = ev.get("kind", "")
        kind_emoji = {"H": "🔝 high", "L": "🔻 low", "C": "📍 close",
                      "O": "🚪 open"}.get(kind, kind)
        lines.append(
            f"| **{_tr(str(ev.get('event', '—')))}** | "
            f"{ev.get('time', '—')} | "
            f"`{_fmt_num(ev.get('price'))}` | "
            f"{kind_emoji} |"
        )
    return "\n".join(lines) if len(lines) > 2 else ""


def _invalidation_layers_md(h: dict) -> str:
    """Phase 47 — 3-layer invalidation (soft/hard/macro)."""
    layers = h.get("invalidation_layers") or {}
    if not isinstance(layers, dict) or not layers:
        return ""
    parts = []
    layer_specs = [
        ("soft",  "🟡 SOFT",  "partial close / trim"),
        ("hard",  "🔴 HARD",  "full stop loss"),
        ("macro", "⚫ MACRO", "переразметка структуры"),
    ]
    for key, label, default_action in layer_specs:
        layer = layers.get(key)
        if not isinstance(layer, dict):
            continue
        trigger = layer.get("trigger", "—")
        action = layer.get("action", default_action)
        tf = layer.get("tf", "—")
        parts.append(
            f"**{label}** ({tf}): {_tr(str(trigger))} → "
            f"{_tr(str(action))}"
        )
    return "\n\n".join(parts)


def _gut_probability_md(h: dict) -> str:
    """Phase 47 — time-bound probability forecast for next N hours."""
    gp = h.get("gut_probability_next_window") or {}
    if not isinstance(gp, dict) or not gp.get("scenarios"):
        return ""
    window = gp.get("window", "next window")
    parts = [f"_Окно прогноза: **{window}**_"]
    lines = ["", "| Сценарий | Вероятность | Ожидаемый диапазон |",
             "|---|---|---|"]
    for sc in gp.get("scenarios", []):
        if not isinstance(sc, dict):
            continue
        prob = sc.get("probability_pct")
        prob_str = f"**{prob}%**" if isinstance(prob, (int, float)) else "—"
        lines.append(
            f"| {_tr(str(sc.get('name', '—')))} | "
            f"{prob_str} | "
            f"{_tr(str(sc.get('expected_range', '—')))} |"
        )
    if len(lines) > 3:
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _track_record_md(h: dict) -> str:
    """Phase 43 — render verified track record of past predictions."""
    track = h.get("track_record") or []
    if not isinstance(track, list) or not track:
        return ""
    verdict_emoji = {
        "hit": "✓", "stopped": "✗", "ambiguous": "⚠",
        "pending": "⏳",
    }
    lines = ["| Версия | Bias | Entry | Stop | Target | Реальность |",
             "|---|---|---|---|---|---|"]
    for e in track[-5:]:
        if not isinstance(e, dict):
            continue
        ver = e.get("version", "?")
        bias = e.get("bias", "?")
        entry = e.get("entry")
        stop = e.get("stop_loss")
        target = e.get("target")
        verdict = e.get("verdict", "pending")
        emoji = verdict_emoji.get(verdict, "?")
        note = (e.get("verdict_note") or "")[:60]
        bias_emoji = {"long": "🟢", "short": "🔴",
                      "neutral": "⚪"}.get(bias, "⚪")
        lines.append(
            f"| **v{ver}** | {bias_emoji} {bias} | "
            f"`{_fmt_num(entry)}` | `{_fmt_num(stop)}` | "
            f"`{_fmt_num(target)}` | {emoji} {verdict} — {note} |"
        )
    return "\n".join(lines) if len(lines) > 2 else ""


def _monitor_next_hours(h: dict) -> str:
    """Phase 34 — time-bound monitor checklist."""
    mnh = h.get("monitor_next_hours") or {}
    if not isinstance(mnh, dict) or not mnh.get("rows"):
        return ""
    parts = []
    if mnh.get("window"):
        parts.append(f"_Окно мониторинга: **{mnh['window']}**_")
    headers = mnh.get("headers") or ["Условие", "Действие"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "---|" * len(headers)]
    for row in mnh.get("rows", []):
        if isinstance(row, list):
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
    if len(lines) > 2:
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _critical_and_golden(h: dict) -> str:
    """Phase 34 — critical_level + golden_entry_zone as final actionable block."""
    parts = []
    if h.get("critical_level"):
        parts.append(f"**🔒 Critical level:** {h['critical_level']}")
    if h.get("golden_entry_zone"):
        # Render in a callout box style
        parts.append(f"\n> {h['golden_entry_zone']}")
    return "\n\n".join(parts)


def _scalp_plan(h: dict) -> str:
    scalp = h.get("scalp_plan") or {}
    if not isinstance(scalp, dict) or not scalp.get("direction"):
        return ""
    direction = (scalp.get("direction") or "wait").lower()
    dir_emoji = {"long": "🟢", "short": "🔴", "wait": "⏸"}.get(direction, "⏸")
    parts = [f"**{dir_emoji} Направление:** {direction.upper()}"]
    if scalp.get("wave_structure_5m"):
        parts.append(f"**📐 Структура 5m:** {scalp['wave_structure_5m']}")
    if scalp.get("rationale"):
        parts.append(f"**💡 Логика:** {scalp['rationale']}")
    if scalp.get("trigger_condition"):
        parts.append(f"**🚨 Триггер:** {scalp['trigger_condition']}")
    e = scalp.get("entry")
    s = scalp.get("stop_loss")
    t1 = scalp.get("target_1")
    t2 = scalp.get("target_2")
    rr = scalp.get("risk_reward")
    if any(v is not None for v in [e, s, t1, t2, rr]):
        lvl_lines = ["", "| Уровень | Значение |", "|---|---|"]
        if e is not None:
            lvl_lines.append(f"| 🎯 Entry | `{_fmt_num(e)}` |")
        if s is not None:
            lvl_lines.append(f"| ⊗ Stop | `{_fmt_num(s)}` |")
        if t1 is not None:
            lvl_lines.append(f"| 🏁 TP1 | `{_fmt_num(t1)}` |")
        if t2 is not None:
            lvl_lines.append(f"| 🏁 TP2 | `{_fmt_num(t2)}` |")
        if rr is not None:
            lvl_lines.append(f"| 📊 R/R | `{_fmt_num(rr, 2)}` |")
        parts.append("\n".join(lvl_lines))
    if scalp.get("expected_timing"):
        parts.append(f"**⏰ Ожидаемое время:** {scalp['expected_timing']}")
    if scalp.get("invalidation"):
        parts.append(f"**⚠ Invalidation:** {scalp['invalidation']}")

    # Fib retrace 5m
    fib5m = scalp.get("fib_retrace_5m") or {}
    if isinstance(fib5m, dict) and any(fib5m.values()):
        flines = ["", "**📐 Fibonacci retrace (5m):**", "",
                  "| Уровень | Цена |", "|---|---|"]
        for k, v in fib5m.items():
            if v is not None:
                flines.append(f"| {k} | `{_fmt_num(v)}` |")
        parts.append("\n".join(flines))

    # Monitoring 5m
    mon5m = scalp.get("monitoring_5m") or []
    if mon5m:
        mlines = ["", "**🎯 Monitoring 5m:**", "",
                  "| Триггер | Сценарий | Действие |", "|---|---|---|"]
        for m in mon5m:
            if isinstance(m, dict):
                mlines.append(
                    f"| {m.get('trigger', '—')} | "
                    f"{m.get('scenario', '—')} | "
                    f"{m.get('action', '—')} |"
                )
        if len(mlines) > 5:
            parts.append("\n".join(mlines))

    return "\n\n".join(parts)


def build_mcp_style_report(h: dict) -> str:
    """Build the full MCP-style markdown report from a session-state holistic dict.

    Sections are emitted in order:
      1. Bias header (banner + summary)
      2. Invalidation of previous (if any)
      3. ▶ ПЛАН ДЕЙСТВИЙ (action_plan) — топ блок
      4. 🎯 УРОВНИ СДЕЛКИ (entry/stop/target/R/R + invalidation)
      5. 📐 СТРУКТУРНЫЙ КОНТЕКСТ (structure_position + rules + alternation + fib + alt)
      6. 📜 PER-TF НОТЫ
      7. ✅ ЗА / ❌ ПРОТИВ
      8. 🎯 ENTRY АЛЬТЕРНАТИВЫ
      9. 📈 TARGET PYRAMID
      10. 🎲 СЦЕНАРИИ
      11. 📋 WAITING CHECKLIST
      12. 🌐 HIGHER-DEGREE CONTEXT
      13. 🎯 MONITORING MATRIX
      14. 🔗 CROSS-CORRELATION + CROSS-ASSET
      15. 📐 5M SCALP PLAN

    Sections without data are skipped automatically.
    """
    if not isinstance(h, dict):
        return "_Нет данных holistic_"

    # Phase 36 — LEAN render. Only the 14 core fields. No duplicates.
    # Order: bias header → invalidation_of_previous → action_plan →
    # golden_entry_zone + critical_level → trade levels → scenarios →
    # monitor_next_hours → structure_position + per_tf_notes →
    # confluence_factors + confidence/risk → comparison_with_previous.
    out = [_bias_header(h)]

    inv_prev = h.get("invalidation_of_previous") or ""
    if inv_prev:
        out.append(f"\n\n> ⚠️ **Invalidation предыдущего setup'а:** {inv_prev}")

    # 1. Главный actionable блок — план действий
    out.append(_section("▶ ПЛАН ДЕЙСТВИЙ", _action_plan(h)))

    # 2. Golden entry zone + critical level (короткие и самые важные)
    cg = _critical_and_golden(h)
    if cg:
        out.append(_section("🔑 КЛЮЧЕВЫЕ УРОВНИ", cg))

    # Phase 47 — 3-layer invalidation (soft / hard / macro)
    out.append(_section("🛡 МНОГОУРОВНЕВАЯ ИНВАЛИДАЦИЯ", _invalidation_layers_md(h)))

    # Phase 47 — Structural events timeline (Breakout / Retest / Bounce)
    out.append(_section("⏱ СТРУКТУРНАЯ ВРЕМЕННАЯ ЛИНИЯ", _events_timeline_md(h)))

    # 3. Уровни сделки (entry/stop/target/R-R)
    out.append(_section("🎯 УРОВНИ СДЕЛКИ", _trade_levels(h)))

    # 4. Phase 42 — Entry variants (Aggressive/Confirmed/Breakout)
    out.append(_section("🎯 ВАРИАНТЫ ВХОДА", _entry_variants_md(h)))

    # 5. Сценарии с probability (для общего if-this-then-that плана)
    out.append(_section("🎲 СЦЕНАРИИ С ВЕРОЯТНОСТЯМИ", _scenarios(h)))

    # Phase 47 — Time-bound probability for next 24-48h
    out.append(_section("🔮 ПРОГНОЗ НА БЛИЖАЙШЕЕ ОКНО", _gut_probability_md(h)))

    # 5. Что мониторить (time-bound checklist)
    out.append(_section("⏰ ЧТО МОНИТОРИТЬ", _monitor_next_hours(h)))

    # 6. Структурный контекст + краткие ноты по ТФ
    sp = h.get("structure_position") or ""
    rc = h.get("rules_check") or ""
    struct_parts = []
    if sp:
        struct_parts.append(f"**📍 Структура:** {sp}")
    if rc:
        struct_parts.append(f"**✅ Валидация R0-R3:** {rc}")
    out.append(_section("📐 СТРУКТУРНЫЙ КОНТЕКСТ", "\n\n".join(struct_parts)))
    out.append(_section("📜 ПО ТАЙМФРЕЙМАМ", _per_tf_notes(h)))

    # 7. Confluence (без confidence_factors/risk_factors — убраны в Phase 38)
    cf_md = _confluence_factors_md(h)
    if cf_md:
        out.append(_section("✅ CONFLUENCE FACTORS", cf_md))

    # 8. Phase 43 — Track record verified vs OHLC (выше comparison)
    out.append(_section("📊 ТРЕК-РЕКОРД ПРЕДСКАЗАНИЙ", _track_record_md(h)))

    # 9. Comparison with previous analysis (опционально, Phase 41)
    cwp_md = _comparison_with_previous(h)
    ezs_md = _entry_zone_shift_md(h)
    if cwp_md or ezs_md:
        combined = []
        if ezs_md:
            combined.append(ezs_md)
        if cwp_md:
            combined.append(cwp_md)
        out.append(_section("🔄 СРАВНЕНИЕ С ПРЕДЫДУЩИМ АНАЛИЗОМ",
                             "\n\n".join(combined)))

    return "".join(out)
