"""Forecast the next price move from the current Elliott structure.

This is what Igor Andreev does in his head: "the impulse is done at wave 5,
expect ABC correction down, A target around the 38.2% Fib of the whole leg,
C target around 100% of A, the whole correction probably takes half the
time of the impulse, invalidation if price breaks back above wave 5 high".

We codify those rules as a deterministic function. Every forecast comes with:
  - the predicted pattern (correction_down / impulse_up / etc.)
  - direction (up / down)
  - confidence (high / medium / low)
  - 1-3 concrete price targets with Fibonacci labels
  - an invalidation level — where the forecast breaks if reached
  - time estimate in bars
  - a short Russian rationale

If the current structure is ambiguous or incomplete, we DON'T forecast —
we say so. Lying is worse than silence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from elliott import WaveCandidate
from probability import probability_of_reaching
from zigzag import Swing


def _categorize(p: float | None) -> Literal["high", "medium", "low"]:
    """Bin an empirical probability into a qualitative label."""
    if p is None:
        return "medium"
    if p >= 0.55:
        return "high"
    if p >= 0.30:
        return "medium"
    return "low"


def _attach_empirical(target: "ForecastTarget", fib_ratio: float) -> None:
    """Look up empirical P(reach) for this Fib level and update target."""
    p = probability_of_reaching(fib_ratio)
    if p is not None:
        target.empirical_p = p
        target.probability = _categorize(p)


# Fibonacci ratios used by the wave-completion rules.
FIB_236 = 0.236
FIB_382 = 0.382
FIB_500 = 0.500
FIB_618 = 0.618
FIB_786 = 0.786
FIB_1000 = 1.000
FIB_1272 = 1.272
FIB_1618 = 1.618


@dataclass
class ForecastTarget:
    """One projected price level after the current structure."""
    label: str         # e.g. "конец волны A", "цель волны 3"
    price: float
    fib_label: str     # e.g. "38.2%", "1.618×"
    probability: Literal["high", "medium", "low"] = "medium"
    # If empirical data is available, this is the measured P(reach).
    # None when probability comes from heuristic priors only.
    empirical_p: float | None = None


@dataclass
class WaveForecast:
    """Prediction of the next move."""
    next_pattern: str
    direction: Literal["up", "down", "unclear"]
    confidence: Literal["high", "medium", "low"]
    rationale: str

    targets: list[ForecastTarget] = field(default_factory=list)
    invalidation_level: float | None = None
    invalidation_reason: str = ""

    # Time estimate as (min_bars, max_bars) — None if too uncertain.
    expected_duration_bars: tuple[int, int] | None = None

    @property
    def is_actionable(self) -> bool:
        """True when there's a usable direction with reasonable confidence."""
        return (self.direction != "unclear"
                and self.confidence != "low"
                and len(self.targets) > 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bars_between(s_start: Swing, s_end: Swing) -> int:
    """Bar distance between two swings."""
    return max(1, abs(s_end.bar_index - s_start.bar_index))


def _no_forecast(reason: str) -> WaveForecast:
    return WaveForecast(
        next_pattern="unknown", direction="unclear", confidence="low",
        rationale=reason,
    )


# ---------------------------------------------------------------------------
# Branch: completed 5-wave impulse → expect ABC correction
# ---------------------------------------------------------------------------
def _forecast_after_completed_impulse(top: WaveCandidate) -> WaveForecast:
    """An impulse just ended (last labeled swing is the wave-5 endpoint).
    Predict an A-B-C correction in the opposite direction."""
    s = top.swings_used
    if len(s) < 6:
        return _no_forecast("Недостаточно swing-точек для полного импульса.")

    s0, s1, s2, s3, s4, s5 = s
    impulse_up = "up" in top.pattern
    full_range = abs(s5.price - s0.price)
    if full_range <= 0:
        return _no_forecast("Нулевой размер импульса — невозможно проецировать.")

    # Time estimate — correction usually 0.38–0.62 of impulse duration.
    impulse_bars = _bars_between(s0, s5)
    expected_duration = (
        max(1, int(impulse_bars * 0.38)),
        max(2, int(impulse_bars * 0.62)),
    )

    if impulse_up:
        next_pattern = "correction_down"
        direction = "down"
        # A target: retrace 38.2% of the entire impulse from wave-5 high.
        a_target = s5.price - FIB_382 * full_range
        # Deeper A: 50% retrace.
        a_deep_target = s5.price - FIB_500 * full_range
        # C target: assume ≈ A in size (typical zigzag) — measured from
        # a yet-unknown wave-B high. We project assuming B retraces 50% of A:
        a_size = s5.price - a_target  # positive — magnitude of expected A
        b_high = a_target + 0.5 * a_size  # 50% retrace of A upward
        c_target = b_high - 1.0 * a_size  # C ≈ A
        c_target_alt = b_high - FIB_1618 * a_size  # extended C

        targets = [
            ForecastTarget("Зона цели волны A (откат 38.2% импульса)",
                           round(a_target, 6), "38.2%", "high"),
            ForecastTarget("Глубже — конец волны A (откат 50% импульса)",
                           round(a_deep_target, 6), "50%", "medium"),
            ForecastTarget("Цель волны C (≈ A) — конец коррекции",
                           round(c_target, 6), "C=A", "high"),
            ForecastTarget("Глубокая цель волны C (1.618×A)",
                           round(c_target_alt, 6), "1.618×A", "low"),
        ]
        _attach_empirical(targets[0], 0.382)
        _attach_empirical(targets[1], 0.500)
        _attach_empirical(targets[2], 1.000)
        _attach_empirical(targets[3], 1.618)
        # Invalidation: price breaks BACK ABOVE wave 5 high — would mean
        # the impulse is actually still extending (truncation or 5th-extension).
        invalidation = s5.price * 1.005  # 0.5% above wave-5 high
        invalidation_reason = (
            f"Цена пробивает максимум волны 5 ({s5.price:.4f} +0.5%). "
            "Это означает что импульс продолжается (5-я волна расширяется), "
            "а не закончился."
        )
    else:
        next_pattern = "correction_up"
        direction = "up"
        a_target = s5.price + FIB_382 * full_range
        a_deep_target = s5.price + FIB_500 * full_range
        a_size = a_target - s5.price
        b_low = a_target - 0.5 * a_size
        c_target = b_low + 1.0 * a_size
        c_target_alt = b_low + FIB_1618 * a_size

        targets = [
            ForecastTarget("Зона цели волны A (откат 38.2% импульса)",
                           round(a_target, 6), "38.2%", "high"),
            ForecastTarget("Глубже — конец волны A (откат 50% импульса)",
                           round(a_deep_target, 6), "50%", "medium"),
            ForecastTarget("Цель волны C (≈ A) — конец коррекции",
                           round(c_target, 6), "C=A", "high"),
            ForecastTarget("Глубокая цель волны C (1.618×A)",
                           round(c_target_alt, 6), "1.618×A", "low"),
        ]
        _attach_empirical(targets[0], 0.382)
        _attach_empirical(targets[1], 0.500)
        _attach_empirical(targets[2], 1.000)
        _attach_empirical(targets[3], 1.618)
        invalidation = s5.price * 0.995
        invalidation_reason = (
            f"Цена пробивает минимум волны 5 ({s5.price:.4f} −0.5%). "
            "Импульс вниз продолжается (5-я волна расширяется)."
        )

    rationale = (
        f"5-волновой {'импульс вверх' if impulse_up else 'импульс вниз'} "
        f"завершён. По теории Эллиотта далее ожидаем A-B-C "
        f"{'коррекцию вниз' if impulse_up else 'коррекцию вверх'}. "
        f"Цели A: 38.2–50% отката всего импульса. C обычно ≈ A или 1.618×A."
    )

    return WaveForecast(
        next_pattern=next_pattern,
        direction=direction,
        confidence="high" if top.score >= 70 and not top.rule_violations else "medium",
        rationale=rationale,
        targets=targets,
        invalidation_level=round(invalidation, 6),
        invalidation_reason=invalidation_reason,
        expected_duration_bars=expected_duration,
    )


# ---------------------------------------------------------------------------
# Branch: completed A-B-C correction → expect new impulse
# ---------------------------------------------------------------------------
def _forecast_after_completed_correction(top: WaveCandidate) -> WaveForecast:
    s = top.swings_used
    if len(s) < 4:
        return _no_forecast("Недостаточно swing-точек для A-B-C структуры.")

    s0, sa, sb, sc = s[-4:]
    correction_up = "up" in top.pattern
    correction_range = abs(sc.price - s0.price)
    if correction_range <= 0:
        return _no_forecast("Нулевой размер коррекции.")

    correction_bars = _bars_between(s0, sc)
    expected_duration = (
        max(2, int(correction_bars * 1.0)),
        max(3, int(correction_bars * 2.5)),
    )

    if correction_up:
        # An A-B-C UP correction (countertrend in a downtrend) just finished.
        # The next move is a fresh down-impulse continuing the larger trend.
        next_pattern = "impulse_down"
        direction = "down"
        # Wave 1 of new impulse: typically retraces 50–62% of the prior correction.
        w1_target = sc.price - FIB_500 * correction_range
        # Wave 3 target: 1.618 × wave 1, projected from a yet-unknown wave 2.
        # Simplification: project an entire 5-wave move equal to 1.618 × ABC.
        w5_target = sc.price - FIB_1618 * correction_range

        targets = [
            ForecastTarget("Цель волны 1 нового импульса (откат 50% ABC)",
                           round(w1_target, 6), "50%", "medium"),
            ForecastTarget("Цель всего импульса (1.618× ABC)",
                           round(w5_target, 6), "1.618×", "high"),
        ]
        invalidation = sc.price * 1.005
        invalidation_reason = (
            f"Цена пробивает максимум {sc.price:.4f} +0.5%. "
            "Тогда коррекция, видимо, ещё не завершена — расширяется (B-волна продолжается)."
        )
    else:
        # A-B-C DOWN correction completed — expect new up-impulse.
        next_pattern = "impulse_up"
        direction = "up"
        w1_target = sc.price + FIB_500 * correction_range
        w5_target = sc.price + FIB_1618 * correction_range

        targets = [
            ForecastTarget("Цель волны 1 нового импульса (откат 50% ABC)",
                           round(w1_target, 6), "50%", "medium"),
            ForecastTarget("Цель всего импульса (1.618× ABC)",
                           round(w5_target, 6), "1.618×", "high"),
        ]
        invalidation = sc.price * 0.995
        invalidation_reason = (
            f"Цена пробивает минимум {sc.price:.4f} −0.5%. "
            "Коррекция ещё не завершена, идёт расширение."
        )

    rationale = (
        f"A-B-C {'коррекция вверх' if correction_up else 'коррекция вниз'} "
        f"завершена. По теории Эллиотта далее начинается новый "
        f"{'импульс вниз' if correction_up else 'импульс вверх'} продолжающий "
        f"больший тренд. Цель импульса обычно 1.618× размера коррекции."
    )

    return WaveForecast(
        next_pattern=next_pattern,
        direction=direction,
        confidence="medium" if top.score >= 60 else "low",
        rationale=rationale,
        targets=targets,
        invalidation_level=round(invalidation, 6),
        invalidation_reason=invalidation_reason,
        expected_duration_bars=expected_duration,
    )


# ---------------------------------------------------------------------------
# Branch: impulse / correction in progress (not yet complete)
# ---------------------------------------------------------------------------
def _forecast_impulse_in_progress(top: WaveCandidate) -> WaveForecast:
    """The detector found a 5-wave structure but flagged it as 'completed N
    swing(s) ago' or with rule violations. Forecast is weaker."""
    msg = (top.current_position or "").lower()
    if "active" in msg or "wave 5" in msg:
        return WaveForecast(
            next_pattern="unknown",
            direction="up" if "up" in top.pattern else "down",
            confidence="medium",
            rationale=(
                "Импульс ещё активен (волна 5 идёт). До окончания волны 5 "
                "конкретные цели задавать рано — следи за достижением "
                "1.618× от волны 1."
            ),
        )
    # The window is in the past — the market is presumably in a correction now.
    return _forecast_after_completed_impulse(top)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def forecast_next(top: WaveCandidate) -> WaveForecast:
    """Return a prediction of the next move based on the top hypothesis."""
    if top is None or top.score <= 0:
        return _no_forecast("Нет структуры — сначала нужно определить волны.")

    if top.rule_violations:
        # Forecasting on a structure that violates hard rules is risky.
        return WaveForecast(
            next_pattern="unknown", direction="unclear", confidence="low",
            rationale=(
                f"Найденная структура нарушает правила: "
                f"{top.rule_violations[0]}. Прогноз делать ненадёжно — "
                "сначала нужна чистая разметка."
            ),
        )

    # Phase 2.1: Ending diagonal → STRONG reversal. Check FIRST because
    # pattern name "ending_diagonal_up" also contains "impulse" semantics.
    if "ending_diagonal" in top.pattern:
        return _forecast_after_ending_diagonal(top)

    # Phase 2.2: Leading diagonal → wave 1 of larger structure → expect wave 2 correction
    if "leading_diagonal" in top.pattern:
        return _forecast_after_leading_diagonal(top)

    if "impulse" in top.pattern:
        # Phase 2.3: Truncated impulses → reversal signal, even stronger than normal.
        if getattr(top, "is_truncated", False):
            return _forecast_after_truncated_impulse(top)
        position = (top.current_position or "").lower()
        if "completed" in position or "ago" in position:
            return _forecast_after_completed_impulse(top)
        return _forecast_impulse_in_progress(top)

    # Phase 1.1: Flats behave like zigzag for the forecast — they're
    # 3-wave corrections that complete and are followed by a strong impulse.
    # The "post-flat thrust" tends to be stronger than after a zigzag.
    if "flat" in top.pattern:
        return _forecast_after_completed_flat(top)

    # Phase 1.2: Triangles are CONTINUATION patterns. After E, price thrusts
    # in the direction labeled by triangle_up/down for a distance ≈ widest leg.
    if "triangle" in top.pattern:
        return _forecast_after_completed_triangle(top)

    # Phase 2.4: WXY complex correction → strong impulse opposite direction
    if "complex_correction" in top.pattern:
        return _forecast_after_complex_correction(top)

    if "correction" in top.pattern:
        return _forecast_after_completed_correction(top)

    return _no_forecast("Неизвестный тип паттерна.")


# ---------------------------------------------------------------------------
# Phase 2.1 — Forecast after ending diagonal
# ---------------------------------------------------------------------------
def _forecast_after_ending_diagonal(top: WaveCandidate) -> WaveForecast:
    """Ending diagonal completes wave 5/C — STRONG reversal expected,
    often retracing entire diagonal back to its starting point quickly."""
    s = top.swings_used
    if len(s) < 6:
        return _no_forecast("Недостаточно swing-точек для диагонали.")
    s0, s1, s2, s3, s4, s5 = s
    diag_up = "up" in top.pattern
    diag_length = abs(s5.price - s0.price)
    if diag_length <= 0:
        return _no_forecast("Нулевой размер диагонали.")

    # Reversal direction is OPPOSITE the diagonal direction
    if diag_up:
        next_pattern = "impulse_down"
        direction = "down"
        primary_target = s0.price                     # retrace to start
        deep_target = s0.price - 0.236 * diag_length  # extend slightly beyond
        invalidation = s5.price + 0.05 * diag_length  # 5% above the top
    else:
        next_pattern = "impulse_up"
        direction = "up"
        primary_target = s0.price
        deep_target = s0.price + 0.236 * diag_length
        invalidation = s5.price - 0.05 * diag_length

    bars_in_diag = _bars_between(s0, s5)
    # Ending diagonals reverse FAST — typically 0.3-0.7× the time spent building
    expected_duration = (
        max(2, int(bars_in_diag * 0.3)),
        max(3, int(bars_in_diag * 0.7)),
    )

    targets = [
        ForecastTarget(
            label="Возврат к началу диагонали",
            fib_label="100% retrace",
            price=float(primary_target),
            probability="high",
            empirical_p=None,
        ),
        ForecastTarget(
            label="Глубокий разворот",
            fib_label="123.6% retrace",
            price=float(deep_target),
            probability="medium",
            empirical_p=None,
        ),
    ]

    rationale = (
        f"Завершён ending diagonal {('вверх' if diag_up else 'вниз')} — "
        f"конец крупного импульса/коррекции, ожидается резкий разворот. "
        f"После ending diagonal цена обычно быстро возвращается к началу клина "
        f"({s0.price:.4f}) — это classic Эллиоттовский reversal-сигнал. "
        f"Инвалидация при пробое последнего хая/лоу клина."
    )

    return WaveForecast(
        next_pattern=next_pattern,
        direction=direction,
        confidence="high",        # ending diagonals are some of the most reliable reversal signals
        rationale=rationale,
        expected_duration_bars=expected_duration,
        targets=targets,
        invalidation_level=float(invalidation),
    )


# ---------------------------------------------------------------------------
# Phase 2.2 — Forecast after leading diagonal
# ---------------------------------------------------------------------------
def _forecast_after_leading_diagonal(top: WaveCandidate) -> WaveForecast:
    """Leading diagonal = wave 1 of new trend. Expect wave 2 correction
    (50-78.6% retrace) BEFORE the main impulse continues."""
    s = top.swings_used
    if len(s) < 6:
        return _no_forecast("Недостаточно swing-точек для диагонали.")
    s0, s1, s2, s3, s4, s5 = s
    diag_up = "up" in top.pattern
    diag_length = abs(s5.price - s0.price)
    if diag_length <= 0:
        return _no_forecast("Нулевой размер диагонали.")

    # Wave 2 is a correction OPPOSITE the diagonal direction
    if diag_up:
        next_pattern = "correction_down"
        direction = "down"
        wave2_50  = s5.price - FIB_500 * diag_length
        wave2_786 = s5.price - FIB_786 * diag_length
        invalidation = s0.price       # wave 2 must not exceed start of wave 1
    else:
        next_pattern = "correction_up"
        direction = "up"
        wave2_50  = s5.price + FIB_500 * diag_length
        wave2_786 = s5.price + FIB_786 * diag_length
        invalidation = s0.price

    bars_in_diag = _bars_between(s0, s5)
    expected_duration = (
        max(2, int(bars_in_diag * 0.5)),
        max(3, int(bars_in_diag * 1.5)),
    )

    targets = [
        ForecastTarget(
            label="Wave 2 — классическая цель",
            fib_label="50% retrace",
            price=float(wave2_50),
            probability="high",
            empirical_p=probability_of_reaching(0.5),
        ),
        ForecastTarget(
            label="Wave 2 — глубокая цель",
            fib_label="78.6% retrace",
            price=float(wave2_786),
            probability="medium",
            empirical_p=probability_of_reaching(0.786),
        ),
    ]

    rationale = (
        f"Завершён leading diagonal {('вверх' if diag_up else 'вниз')} — это "
        f"волна 1 более крупной структуры. Перед продолжением тренда ожидается "
        f"коррекция wave 2 на 50-78.6% от пройденной диагонали. "
        f"Инвалидация: пробой начала wave 1 ({s0.price:.4f}) полностью "
        f"отменяет сценарий."
    )

    return WaveForecast(
        next_pattern=next_pattern,
        direction=direction,
        confidence="medium",      # diagonals at start are less reliable than impulses
        rationale=rationale,
        expected_duration_bars=expected_duration,
        targets=targets,
        invalidation_level=float(invalidation),
    )


# ---------------------------------------------------------------------------
# Phase 2.3 — Forecast after truncated fifth wave
# ---------------------------------------------------------------------------
def _forecast_after_truncated_impulse(top: WaveCandidate) -> WaveForecast:
    """Truncation (failed fifth) = STRONG reversal — buyers/sellers exhausted
    before wave 5 even reached wave 3 high/low. Expect sharp counter-move."""
    s = top.swings_used
    if len(s) < 6:
        return _no_forecast("Недостаточно swing-точек для импульса.")
    s0, s1, s2, s3, s4, s5 = s
    impulse_up = "up" in top.pattern
    full_range = abs(s5.price - s0.price)
    if full_range <= 0:
        return _no_forecast("Нулевой размер импульса.")

    # Reversal opposite to (failed) impulse direction.
    # Targets are DEEPER than normal post-impulse correction because exhaustion was extreme.
    if impulse_up:
        next_pattern = "correction_down"
        direction = "down"
        primary = s5.price - FIB_618 * full_range  # 61.8% retrace (deep)
        deep    = s0.price                          # full retrace (very deep)
        invalidation = s3.price + 0.01 * s3.price  # break of wave 3 high cancels truncation
    else:
        next_pattern = "correction_up"
        direction = "up"
        primary = s5.price + FIB_618 * full_range
        deep    = s0.price
        invalidation = s3.price - 0.01 * s3.price

    bars_in_impulse = _bars_between(s0, s5)
    expected_duration = (
        max(2, int(bars_in_impulse * 0.4)),
        max(3, int(bars_in_impulse * 1.0)),
    )

    targets = [
        ForecastTarget(
            label="Глубокий откат (truncation reversal)",
            fib_label="61.8% retrace",
            price=float(primary),
            probability="high",
            empirical_p=probability_of_reaching(0.618),
        ),
        ForecastTarget(
            label="Полный откат к началу",
            fib_label="100% retrace",
            price=float(deep),
            probability="medium",
            empirical_p=probability_of_reaching(1.0),
        ),
    ]

    truncation_pct = getattr(top, "truncation_distance", 0) or 0
    rationale = (
        f"⚠️ TRUNCATED FIFTH — волна 5 не пробила вершину/дно волны 3 "
        f"(не дошла на {truncation_pct:.2%}). Это сильнейший сигнал истощения: "
        f"покупатели/продавцы не смогли довести импульс. Ожидается резкий и "
        f"глубокий разворот, часто к началу всего импульса. Инвалидация — "
        f"возврат за вершину/дно wave 3."
    )

    return WaveForecast(
        next_pattern=next_pattern,
        direction=direction,
        confidence="high",
        rationale=rationale,
        expected_duration_bars=expected_duration,
        targets=targets,
        invalidation_level=float(invalidation),
    )


# ---------------------------------------------------------------------------
# Phase 2.4 — Forecast after WXY complex correction
# ---------------------------------------------------------------------------
def _forecast_after_complex_correction(top: WaveCandidate) -> WaveForecast:
    """WXY double-three — full correction complete, expect strong impulse
    in the opposite direction."""
    s = top.swings_used
    if len(s) < 8:
        return _no_forecast("Недостаточно swing-точек для WXY.")
    s0 = s[0]
    sc2 = s[-1]
    correction_up = "up" in top.pattern
    full_range = abs(sc2.price - s0.price)
    if full_range <= 0:
        return _no_forecast("Нулевой размер WXY.")

    # Post-WXY impulse OPPOSITE to correction direction
    if correction_up:
        next_pattern = "impulse_down"
        direction = "down"
        primary = sc2.price - FIB_1000 * full_range
        deep    = sc2.price - FIB_1618 * full_range
        invalidation = sc2.price
    else:
        next_pattern = "impulse_up"
        direction = "up"
        primary = sc2.price + FIB_1000 * full_range
        deep    = sc2.price + FIB_1618 * full_range
        invalidation = sc2.price

    bars_in_wxy = _bars_between(s0, sc2)
    expected_duration = (
        max(3, int(bars_in_wxy * 0.5)),
        max(5, int(bars_in_wxy * 1.5)),
    )

    targets = [
        ForecastTarget(
            label="Цель импульса 1× WXY",
            fib_label="1.0×",
            price=float(primary),
            probability="high",
            empirical_p=probability_of_reaching(1.0),
        ),
        ForecastTarget(
            label="Глубокая цель 1.618× WXY",
            fib_label="1.618×",
            price=float(deep),
            probability="medium",
            empirical_p=probability_of_reaching(1.618),
        ),
    ]

    rationale = (
        f"Завершён WXY double-three {('вверх' if correction_up else 'вниз')} — "
        f"крупная коррекция, состоящая из двух простых корректирующих структур "
        f"через X-волну. После такой коррекции обычно идёт мощный импульс в "
        f"противоположном направлении, цели проецируются от длины всей WXY."
    )

    return WaveForecast(
        next_pattern=next_pattern,
        direction=direction,
        confidence="high",
        rationale=rationale,
        expected_duration_bars=expected_duration,
        targets=targets,
        invalidation_level=float(invalidation),
    )


# ---------------------------------------------------------------------------
# Phase 1.1: Forecast after a completed flat correction
# ---------------------------------------------------------------------------
def _forecast_after_completed_flat(top: WaveCandidate) -> WaveForecast:
    """A flat correction just ended. Expect a strong impulse in the
    direction OPPOSITE to the flat's nominal direction.

    Flats — especially expanded flats — typically precede stronger thrusts
    than zigzags. We bump confidence and target magnitudes accordingly.
    """
    s = top.swings_used
    if len(s) < 4:
        return _no_forecast("Недостаточно swing-точек для плоскости.")

    s0, sa, sb, sc = s
    flat_up = "up" in top.pattern
    a_length = abs(sa.price - s0.price)
    if a_length <= 0:
        return _no_forecast("Нулевой размер A-волны — нельзя проецировать.")

    bars_in_flat = _bars_between(s0, sc)
    expected_duration = (
        max(2, int(bars_in_flat * 1.0)),
        max(3, int(bars_in_flat * 2.0)),
    )

    # Thrust direction is OPPOSITE to flat label (because flats are
    # corrections — the impulse afterwards reverses the corrective direction).
    if flat_up:
        next_pattern = "impulse_down"
        direction = "down"
        # Targets project DOWN from the completion point (sc)
        t1 = sc.price - 1.0 * a_length
        t2 = sc.price - FIB_1618 * a_length
    else:
        next_pattern = "impulse_up"
        direction = "up"
        t1 = sc.price + 1.0 * a_length
        t2 = sc.price + FIB_1618 * a_length

    is_expanded = any("EXPANDED" in n for n in top.guideline_notes)
    confidence = "high" if is_expanded else "medium"

    targets = [
        ForecastTarget(
            label="Post-flat thrust (классическая цель)",
            fib_label="1.0×A",
            price=float(t1), probability="medium",
            empirical_p=probability_of_reaching(1.0),
        ),
        ForecastTarget(
            label="Extended thrust",
            fib_label="1.618×A",
            price=float(t2), probability=("high" if is_expanded else "medium"),
            empirical_p=probability_of_reaching(1.618),
        ),
    ]
    invalidation = sb.price  # any move beyond B negates the flat

    rationale = (
        f"{'Расширенная' if is_expanded else 'Обычная'} плоскость "
        f"{'вверх' if flat_up else 'вниз'} завершена на {sc.price:.4f}. "
        f"После плоскости ожидается сильный {'импульс вниз' if flat_up else 'импульс вверх'}. "
        f"Плоскости (особенно расширенные) предваряют более мощные движения "
        f"чем зигзаги — цели проецируются на 1.0× и 1.618× длины волны A."
    )

    return WaveForecast(
        next_pattern=next_pattern,
        direction=direction,
        confidence=confidence,
        rationale=rationale,
        expected_duration_bars=expected_duration,
        targets=targets,
        invalidation_level=float(invalidation),
    )


# ---------------------------------------------------------------------------
# Phase 1.2: Forecast after a completed triangle
# ---------------------------------------------------------------------------
def _forecast_after_completed_triangle(top: WaveCandidate) -> WaveForecast:
    """A contracting triangle completed at point E. Expect a thrust in the
    direction labeled by triangle_up/triangle_down, with magnitude ≈ widest leg.
    """
    s = top.swings_used
    if len(s) < 6:
        return _no_forecast("Недостаточно swing-точек для треугольника.")

    s0, sa, sb, sc, sd, se = s
    triangle_up = "up" in top.pattern
    direction = "up" if triangle_up else "down"

    a_length = abs(sa.price - s0.price)
    if a_length <= 0:
        return _no_forecast("Нулевой размер леги A — нельзя проецировать thrust.")

    # The thrust phase is usually quick — 0.3-0.6× the time spent in the triangle.
    triangle_bars = _bars_between(s0, se)
    expected_duration = (
        max(2, int(triangle_bars * 0.3)),
        max(3, int(triangle_bars * 0.7)),
    )

    # Triangle thrust target = length of A added to breakout point E.
    primary_target = top.thrust_target if top.thrust_target is not None else (
        se.price + a_length if triangle_up else se.price - a_length
    )
    extended_target = (
        se.price + FIB_1618 * a_length if triangle_up
        else se.price - FIB_1618 * a_length
    )

    # Invalidation: break beyond C boundary cancels the triangle.
    invalidation = sc.price

    targets = [
        ForecastTarget(
            label="Классическая цель thrust",
            fib_label="≈A",
            price=float(primary_target), probability="high",
            empirical_p=None,    # triangle thrusts are not in the impulse-based prob table
        ),
        ForecastTarget(
            label="Мощный thrust",
            fib_label="1.618×A",
            price=float(extended_target), probability="medium",
            empirical_p=None,
        ),
    ]

    next_pattern = "impulse_up" if triangle_up else "impulse_down"
    rationale = (
        f"Сжимающийся треугольник A-B-C-D-E завершён на {se.price:.4f}. "
        f"Высота самой широкой ноги (A) ≈ {a_length:.4f}. После треугольников "
        f"типично следует резкий thrust в направлении {'вверх' if triangle_up else 'вниз'} "
        f"на длину A от точки E. Инвалидация — закрытие за границу C ({sc.price:.4f})."
    )

    return WaveForecast(
        next_pattern=next_pattern,
        direction=direction,
        confidence="high",       # triangles have highly directional thrust signals
        rationale=rationale,
        expected_duration_bars=expected_duration,
        targets=targets,
        invalidation_level=float(invalidation),
    )
