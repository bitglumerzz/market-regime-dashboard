"""Elliott Wave classifier — score competing hypotheses on a swing sequence.

This is NOT a magic oracle. Full Elliott wave analysis is an open research
problem; professional wave-counters maintain multiple parallel hypotheses
that update as new bars arrive. We mimic that posture: given the last few
swings, we score each plausible labeling against the Elliott rules and
return a ranked list.

Standard rules used here:
  Impulse (1-2-3-4-5):
    R0) Net direction sanity: an up impulse must end ABOVE where it started
        (s5 > s0). Alternation alone is not enough — without this rule a
        sideways sequence of 6 alternating swings can score as an impulse.
    R1) Wave 2 does not retrace more than 100% of wave 1.
    R2) Wave 3 is never the shortest of waves 1, 3, 5.
    R3) Wave 4 does not overlap the price territory of wave 1.
  Common guidelines (scored, not strict):
    G1) Wave 2 commonly retraces 50–61.8% of wave 1.
    G2) Wave 3 commonly equals 1.618 × wave 1 (extension).
    G3) Wave 4 commonly retraces 38.2% of wave 3.
    G4) Wave 5 commonly equals wave 1 (or 0.618 × wave 1).

  Zigzag correction (A-B-C):
    Z0) Net direction sanity (same idea as R0).
    Z1) B retraces 38.2–78.6% of A.
    Z2) C is roughly equal to A, or 1.618 × A.

Travel bonus: bigger moves score higher than micro-structures with the
same ratios. A 30% impulse is more meaningful than a 1% one.

Window search: we don't just look at the very last 6 swings. An impulse
may have completed N swings ago; we evaluate several recent windows and
return the best-scoring one.

Higher final score = more consistent with the rules + larger / more recent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from zigzag import Swing


WavePattern = Literal[
    # Five-wave motive structures
    "impulse_up", "impulse_down",
    # Three-wave corrections — zigzag (5-3-5)
    "correction_up", "correction_down",
    # Three-wave corrections — flat (3-3-5, B retraces ≥0.9× A)
    "flat_up", "flat_down",
    # Five-wave triangle corrections (ABCDE)
    "triangle_up", "triangle_down",
    # Diagonal wedges — overlap of wave 1-4 allowed
    "leading_diagonal_up", "leading_diagonal_down",
    "ending_diagonal_up", "ending_diagonal_down",
    # Complex correction WXY (double three)
    "complex_correction_up", "complex_correction_down",
    "undefined",
]


# Fibonacci levels used in scoring.
FIB_236 = 0.236
FIB_382 = 0.382
FIB_500 = 0.500
FIB_618 = 0.618
FIB_786 = 0.786
FIB_1000 = 1.000
FIB_1272 = 1.272
FIB_1618 = 1.618
FIB_2618 = 2.618


@dataclass
class WaveCandidate:
    """One scored hypothesis about the current wave structure."""
    pattern: WavePattern
    swings_used: list[Swing]                 # the labeled swing endpoints
    labels: list[str]                        # e.g. ['0', '1', '2', '3', '4', '5']
    rule_violations: list[str] = field(default_factory=list)
    guideline_notes: list[str] = field(default_factory=list)
    score: float = 0.0                       # higher = more plausible (0-100)
    current_position: str = ""               # "completing wave 5", "in wave 2", etc.
    fibs: dict[str, float] = field(default_factory=dict)
    next_targets: dict[str, float] = field(default_factory=dict)
    # Phase-1 metadata extensions
    is_extended: bool = False                # True if wave 3 ≥ 1.5× wave 1 (extended impulse)
    extension_ratio: float | None = None     # actual wave3/wave1 ratio when extended
    triangle_kind: str = ""                  # "contracting" / "expanding" / "" if not a triangle
    thrust_target: float | None = None       # post-triangle expected price after E breakout
    # Phase-2 metadata extensions
    is_truncated: bool = False               # True if wave 5 fails to break wave 3 (failed fifth)
    truncation_distance: float | None = None # how far wave 5 fell short, as fraction of price
    is_diagonal: bool = False                # True for leading or ending diagonal
    diagonal_kind: str = ""                  # "leading" / "ending" / ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _abs_len(start: Swing, end: Swing) -> float:
    return abs(end.price - start.price)


def _is_up_move(a: Swing, b: Swing) -> bool:
    return b.price > a.price


def _fib_distance(actual_ratio: float, target_ratio: float) -> float:
    """Score how close an observed ratio is to a Fibonacci target.

    Returns a value in [0, 1] — 1 if exactly on target, falling off
    smoothly to 0 at ±50% deviation.
    """
    deviation = abs(actual_ratio - target_ratio) / max(target_ratio, 1e-6)
    return float(max(0.0, 1.0 - deviation / 0.5))


def _nearest_fib_score(actual: float, targets: tuple[float, ...]) -> tuple[float, float]:
    """Return (best_score, best_target) for the closest Fibonacci level."""
    best_target = targets[0]
    best_score = _fib_distance(actual, best_target)
    for t in targets[1:]:
        s = _fib_distance(actual, t)
        if s > best_score:
            best_score = s
            best_target = t
    return best_score, best_target


# ---------------------------------------------------------------------------
# Impulse scorer
# ---------------------------------------------------------------------------
def _score_impulse_window(
    swings: list[Swing], direction: Literal["up", "down"]
) -> WaveCandidate:
    """Score exactly 6 consecutive swings as an impulse in the given direction.

    Swings are labeled 0 → 1 → 2 → 3 → 4 → 5.
    For an UP impulse: 0=low, 1=high, 2=low, 3=high, 4=low, 5=high.
    """
    if len(swings) != 6:
        return WaveCandidate(
            pattern=f"impulse_{direction}", swings_used=swings,
            labels=[], score=0.0,
            current_position="window size != 6",
        )

    s0, s1, s2, s3, s4, s5 = swings
    labels = ["0", "1", "2", "3", "4", "5"]

    # Alternation must match the impulse direction.
    if direction == "up":
        kinds_expected = ["low", "high", "low", "high", "low", "high"]
    else:
        kinds_expected = ["high", "low", "high", "low", "high", "low"]
    if [s.kind for s in (s0, s1, s2, s3, s4, s5)] != kinds_expected:
        return WaveCandidate(
            pattern=f"impulse_{direction}", swings_used=[s0, s1, s2, s3, s4, s5],
            labels=labels, score=0.0,
            rule_violations=["swing alternation does not match impulse direction"],
            current_position="structural mismatch",
        )

    wave1 = _abs_len(s0, s1)
    wave2 = _abs_len(s1, s2)
    wave3 = _abs_len(s2, s3)
    wave4 = _abs_len(s3, s4)
    wave5 = _abs_len(s4, s5)

    violations: list[str] = []
    notes: list[str] = []
    score = 0.0

    # R0 — net direction sanity. The MOST IMPORTANT rule we previously missed.
    # An up impulse must END above where it began; a down impulse must end below.
    # Without this check, sideways alternating sequences score as impulses.
    if direction == "up" and s5.price <= s0.price:
        violations.append(
            f"R0 violated: 'up' impulse ends below start "
            f"({s5.price:.4f} ≤ {s0.price:.4f}) — net direction is not up"
        )
    elif direction == "down" and s5.price >= s0.price:
        violations.append(
            f"R0 violated: 'down' impulse ends above start "
            f"({s5.price:.4f} ≥ {s0.price:.4f}) — net direction is not down"
        )
    else:
        score += 25  # R0 satisfied — most important single check

    # R1 — wave 2 doesn't fully retrace wave 1.
    retrace_2 = wave2 / max(wave1, 1e-9)
    if retrace_2 >= 1.0:
        violations.append(f"R1 violated: wave 2 retraced {retrace_2:.0%} of wave 1 (>= 100%)")
    else:
        score += 15

    # R2 — wave 3 not the shortest of (1, 3, 5).
    if wave3 < wave1 and wave3 < wave5:
        violations.append("R2 violated: wave 3 is the shortest of waves 1, 3, 5")
    else:
        score += 15

    # R3 — wave 4 doesn't overlap wave 1 territory.
    if direction == "up":
        overlap = s4.price <= s1.price
    else:
        overlap = s4.price >= s1.price
    if overlap:
        violations.append("R3 violated: wave 4 entered wave 1 territory")
    else:
        score += 15

    # G1 — wave 2 retracement close to 0.5–0.618.
    g1_score, g1_target = _nearest_fib_score(retrace_2, (FIB_382, FIB_500, FIB_618, FIB_786))
    score += 8 * g1_score
    notes.append(f"wave 2 retraced {retrace_2:.1%} (closest Fib: {g1_target:.3f})")

    # G2 — wave 3 length vs wave 1 (target 1.618, accept 1.272).
    ratio_31 = wave3 / max(wave1, 1e-9)
    g2_score, g2_target = _nearest_fib_score(ratio_31, (FIB_1000, FIB_1272, FIB_1618, FIB_2618))
    score += 12 * g2_score
    notes.append(f"wave 3 / wave 1 = {ratio_31:.2f}× (closest Fib: {g2_target:.3f})")

    # G3 — wave 4 retracement of wave 3 (target 0.382).
    retrace_4 = wave4 / max(wave3, 1e-9)
    g3_score, g3_target = _nearest_fib_score(retrace_4, (FIB_236, FIB_382, FIB_500))
    score += 8 * g3_score
    notes.append(f"wave 4 retraced {retrace_4:.1%} of wave 3 (closest Fib: {g3_target:.3f})")

    # G4 — wave 5 ≈ wave 1 (or 0.618 × wave 1).
    ratio_51 = wave5 / max(wave1, 1e-9)
    g4_score, g4_target = _nearest_fib_score(ratio_51, (FIB_618, FIB_1000, FIB_1618))
    score += 8 * g4_score
    notes.append(f"wave 5 / wave 1 = {ratio_51:.2f}× (closest Fib: {g4_target:.3f})")

    # B0 — travel bonus: reward larger absolute moves. Up to +6 points.
    # A 30%+ move gets the full bonus; smaller moves scale linearly.
    if s0.price > 0:
        travel_pct = abs(s5.price - s0.price) / s0.price
        travel_bonus = min(6.0, travel_pct * 20.0)
        score += travel_bonus
        notes.append(f"net travel {direction}: "
                      f"{(s5.price - s0.price) / s0.price * 100:+.1f}%")

    # Fibonacci levels relative to the 0→1 leg, useful for the next move.
    base = s1.price - s0.price  # signed
    fibs = {
        "0%":   s1.price,
        "23.6%": s1.price - FIB_236 * base,
        "38.2%": s1.price - FIB_382 * base,
        "50%":   s1.price - FIB_500 * base,
        "61.8%": s1.price - FIB_618 * base,
        "78.6%": s1.price - FIB_786 * base,
        "100%":  s0.price,
    }

    # Targets for hypothetical extension beyond wave 5 (modest projection).
    last_price = s5.price
    move_03 = s3.price - s0.price
    next_targets = {
        "wave-5 ext 100% (next pivot)": last_price,
        "0→5 extension 161.8%": s0.price + FIB_1618 * move_03,
        "wave 6 (=A) ≈ 38.2% retrace": last_price - 0.382 * (last_price - s0.price),
    }

    if not violations:
        position = (
            f"completing wave 5 of {direction} impulse — next likely "
            f"correction A-B-C"
        )
    else:
        position = f"{direction} impulse hypothesis (with violations)"

    # Phase 1.3 — Wave 3 extension detection.
    # In extended impulses, wave 3 is 1.618-3× wave 1.
    # Calibration insight: 49% of impulses flagged as extended in earlier
    # threshold (1.5×) — too loose. Tightened to 1.618× minimum.
    # Also require the impulse itself to be valid (no R0 violation).
    has_r0_violation = any("R0 violated" in v for v in violations)
    is_extended = (not has_r0_violation) and bool(1.618 <= ratio_31 <= 3.0)
    if is_extended:
        notes.append(
            f"⚡ EXTENDED IMPULSE — wave 3 = {ratio_31:.2f}× wave 1 "
            f"(typical wave 5 ≈ wave 1 ≈ {wave1:.4f})"
        )

    # Phase 2.3 — Truncation: wave 5 fails to make a new high/low beyond wave 3.
    # Require ≥1.0% gap (not just any tiny failure) — micro-misses are noise.
    # Calibration insight: too-loose threshold caused 51% of impulses to flag
    # as truncated in the 6399-classification report.
    TRUNCATION_MIN_PCT = 0.010
    if direction == "up":
        raw_gap = (s3.price - s5.price) / max(s3.price, 1e-9)
    else:
        raw_gap = (s5.price - s3.price) / max(s3.price, 1e-9)
    is_truncated = raw_gap >= TRUNCATION_MIN_PCT
    truncation_distance = raw_gap if is_truncated else 0.0
    if is_truncated:
        notes.append(
            f"⚠️ TRUNCATED FIFTH — wave 5 failed to break wave 3 high/low "
            f"by {truncation_distance:.2%} — STRONG reversal signal expected"
        )
        score += 5

    return WaveCandidate(
        pattern=f"impulse_{direction}",
        swings_used=[s0, s1, s2, s3, s4, s5],
        labels=labels,
        rule_violations=violations,
        guideline_notes=notes,
        score=float(score),
        current_position=position,
        fibs=fibs,
        next_targets=next_targets,
        is_extended=is_extended,
        extension_ratio=ratio_31 if is_extended else None,
        is_truncated=is_truncated,
        truncation_distance=truncation_distance if is_truncated else None,
    )


# ---------------------------------------------------------------------------
# Correction scorer (A-B-C zigzag)
# ---------------------------------------------------------------------------
def _score_correction_window(
    swings: list[Swing], direction: Literal["up", "down"]
) -> WaveCandidate:
    """Score exactly 4 consecutive swings as an A-B-C correction."""
    if len(swings) != 4:
        return WaveCandidate(
            pattern=f"correction_{direction}", swings_used=swings,
            labels=[], score=0.0,
            current_position="window size != 4",
        )

    s0, sa, sb, sc = swings
    labels = ["0", "A", "B", "C"]

    if direction == "up":
        kinds_expected = ["low", "high", "low", "high"]
    else:
        kinds_expected = ["high", "low", "high", "low"]
    if [s.kind for s in (s0, sa, sb, sc)] != kinds_expected:
        return WaveCandidate(
            pattern=f"correction_{direction}", swings_used=[s0, sa, sb, sc],
            labels=labels, score=0.0,
            rule_violations=["swing alternation does not match correction direction"],
            current_position="structural mismatch",
        )

    wave_a = _abs_len(s0, sa)
    wave_b = _abs_len(sa, sb)
    wave_c = _abs_len(sb, sc)

    violations: list[str] = []
    notes: list[str] = []
    score = 0.0

    # Z0 — net direction sanity (same idea as R0 for impulses).
    if direction == "up" and sc.price <= s0.price:
        violations.append(
            f"Z0 violated: 'up' correction ends below start — net direction wrong"
        )
    elif direction == "down" and sc.price >= s0.price:
        violations.append(
            f"Z0 violated: 'down' correction ends above start — net direction wrong"
        )
    else:
        score += 15  # Z0 satisfied

    # Z1 — B retraces 38.2-78.6% of A.
    retrace_b = wave_b / max(wave_a, 1e-9)
    if FIB_236 <= retrace_b <= 0.9:
        score += 20
        z1_score, z1_target = _nearest_fib_score(retrace_b, (FIB_382, FIB_500, FIB_618, FIB_786))
        score += 15 * z1_score
        notes.append(f"B retraced {retrace_b:.1%} of A (closest Fib: {z1_target:.3f})")
    else:
        violations.append(f"Z1 violated: B retraced {retrace_b:.0%} of A "
                          f"(outside 23.6–90%)")

    # Z2 — C ≈ A or 1.618 × A.
    ratio_ca = wave_c / max(wave_a, 1e-9)
    z2_score, z2_target = _nearest_fib_score(ratio_ca, (FIB_618, FIB_1000, FIB_1272, FIB_1618))
    score += 25 * z2_score
    notes.append(f"C / A = {ratio_ca:.2f}× (closest Fib: {z2_target:.3f})")

    # Travel bonus — same as impulse.
    if s0.price > 0:
        travel_pct = abs(sc.price - s0.price) / s0.price
        score += min(5.0, travel_pct * 20.0)

    # Light baseline so corrections with mediocre fits still register.
    score += 10

    fibs = {
        "100%": s0.price,
        "61.8%": sa.price - FIB_618 * (sa.price - s0.price),
        "50%":   sa.price - FIB_500 * (sa.price - s0.price),
        "38.2%": sa.price - FIB_382 * (sa.price - s0.price),
    }
    next_targets = {
        "next impulse 161.8% (if reversal)": s0.price + FIB_1618 * (sa.price - s0.price),
    }

    return WaveCandidate(
        pattern=f"correction_{direction}",
        swings_used=[s0, sa, sb, sc],
        labels=labels,
        rule_violations=violations,
        guideline_notes=notes,
        score=float(score),
        current_position=(
            f"completed A-B-C {direction} correction — next likely impulse "
            f"in opposite direction"
            if not violations else
            f"{direction} A-B-C correction hypothesis (with violations)"
        ),
        fibs=fibs,
        next_targets=next_targets,
    )


# ---------------------------------------------------------------------------
# Window search — find the strongest impulse/correction across recent windows
# ---------------------------------------------------------------------------
# How many windows back from the most recent swing to search. A larger value
# means we can recognize impulses that ended a few swings ago and are now
# being corrected.
WINDOW_SEARCH_DEPTH: int = 6


def _best_impulse(
    swings: list[Swing], direction: Literal["up", "down"]
) -> WaveCandidate:
    """Slide a 6-swing window across recent history; return the best score."""
    if len(swings) < 6:
        return WaveCandidate(
            pattern=f"impulse_{direction}", swings_used=swings,
            labels=[], score=0.0,
            current_position="not enough swings for full impulse",
        )
    best: WaveCandidate | None = None
    # End indices to try: the most recent first, then 1 back, 2 back, etc.
    # End index `e` means we evaluate swings[e-6:e].
    n = len(swings)
    e_min = max(6, n - WINDOW_SEARCH_DEPTH)
    for e in range(n, e_min - 1, -1):
        window = swings[e - 6:e]
        if len(window) != 6:
            continue
        cand = _score_impulse_window(window, direction)
        if best is None or cand.score > best.score:
            best = cand
            # Annotate the position with how recent the window ends.
            bars_back = n - e
            if bars_back == 0:
                best.current_position = (
                    f"impulse {direction} active — last swing is wave 5"
                )
            else:
                best.current_position = (
                    f"impulse {direction} completed {bars_back} swing(s) ago — "
                    f"market likely in subsequent correction now"
                )
    return best or WaveCandidate(
        pattern=f"impulse_{direction}", swings_used=[], labels=[], score=0.0,
    )


def _best_correction(
    swings: list[Swing], direction: Literal["up", "down"]
) -> WaveCandidate:
    """Same idea for 4-swing A-B-C correction windows."""
    if len(swings) < 4:
        return WaveCandidate(
            pattern=f"correction_{direction}", swings_used=swings,
            labels=[], score=0.0,
            current_position="not enough swings for A-B-C",
        )
    best: WaveCandidate | None = None
    n = len(swings)
    e_min = max(4, n - WINDOW_SEARCH_DEPTH)
    for e in range(n, e_min - 1, -1):
        window = swings[e - 4:e]
        if len(window) != 4:
            continue
        cand = _score_correction_window(window, direction)
        if best is None or cand.score > best.score:
            best = cand
            bars_back = n - e
            if bars_back == 0:
                best.current_position = (
                    f"A-B-C {direction} correction completed — "
                    f"next likely impulse in opposite direction"
                )
            else:
                best.current_position = (
                    f"A-B-C {direction} ended {bars_back} swing(s) ago"
                )
    return best or WaveCandidate(
        pattern=f"correction_{direction}", swings_used=[], labels=[], score=0.0,
    )


# ---------------------------------------------------------------------------
# Phase 1.1 — Flat correction (3-3-5)
# ---------------------------------------------------------------------------
def _score_flat_window(
    swings: list[Swing], direction: Literal["up", "down"]
) -> WaveCandidate:
    """Score a 4-swing window as a FLAT correction (distinct from zigzag).

    Key Flat rules:
      F0) Direction sanity (same as Z0).
      F1) B retraces ≥0.9× A (deep retrace — the *defining* feature vs zigzag).
          - 0.9-1.0 = regular flat
          - >1.0    = expanded flat (B exceeds start of A)
      F2) C ≈ A in length (regular) or 1.272-1.618× A (expanded flat C-extension).
      F3) Travel bonus: smaller than a zigzag of the same swings — flats are sideways.
    """
    if len(swings) != 4:
        return WaveCandidate(
            pattern=f"flat_{direction}", swings_used=swings,
            labels=[], score=0.0,
            current_position="window size != 4",
        )

    s0, sa, sb, sc = swings
    labels = ["0", "A", "B", "C"]

    if direction == "up":
        kinds_expected = ["low", "high", "low", "high"]
    else:
        kinds_expected = ["high", "low", "high", "low"]
    if [s.kind for s in (s0, sa, sb, sc)] != kinds_expected:
        return WaveCandidate(
            pattern=f"flat_{direction}", swings_used=[s0, sa, sb, sc],
            labels=labels, score=0.0,
            rule_violations=["swing alternation does not match flat direction"],
            current_position="structural mismatch",
        )

    wave_a = _abs_len(s0, sa)
    wave_b = _abs_len(sa, sb)
    wave_c = _abs_len(sb, sc)

    violations: list[str] = []
    notes: list[str] = []
    score = 0.0
    flat_subtype = "regular"

    # F0 — direction sanity. Flats CAN be "running" (net contradicting B) but
    # for v1 we keep strict net-direction-matches-label.
    if direction == "up" and sc.price <= s0.price:
        violations.append("F0 violated: up flat ends below start")
    elif direction == "down" and sc.price >= s0.price:
        violations.append("F0 violated: down flat ends above start")
    else:
        score += 12

    # F1 — B retraces ≥0.9× A. This is the SIGNATURE flat feature.
    retrace_b = wave_b / max(wave_a, 1e-9)
    if retrace_b < 0.85:
        # Not enough B retracement — it's a zigzag, not a flat.
        return WaveCandidate(
            pattern=f"flat_{direction}", swings_used=[s0, sa, sb, sc],
            labels=labels, score=0.0,
            rule_violations=[f"F1 violated: B retraced only {retrace_b:.0%} "
                             f"of A — too shallow for flat (need ≥85%)"],
            current_position="this is a zigzag, not a flat",
        )
    if 0.85 <= retrace_b < 1.0:
        flat_subtype = "regular"
        score += 22
        notes.append(f"REGULAR flat — B retraced {retrace_b:.0%} of A")
    elif 1.0 <= retrace_b <= 1.4:
        flat_subtype = "expanded"
        score += 28  # expanded flats are stronger reversal signals
        notes.append(f"EXPANDED flat — B exceeded start by {(retrace_b - 1):.1%}")
    else:
        violations.append(
            f"F1 violated: B retraced {retrace_b:.0%} of A — too deep for flat"
        )

    # F2 — C-to-A relationship.
    ratio_ca = wave_c / max(wave_a, 1e-9)
    if flat_subtype == "regular":
        f2_score, f2_target = _nearest_fib_score(
            ratio_ca, (FIB_1000, FIB_1272))
        score += 25 * f2_score
        notes.append(f"C / A = {ratio_ca:.2f}× (regular flat target: 1.0)")
    else:
        f2_score, f2_target = _nearest_fib_score(
            ratio_ca, (FIB_1272, FIB_1618, FIB_1000))
        score += 22 * f2_score
        notes.append(f"C / A = {ratio_ca:.2f}× (expanded flat target: 1.272–1.618)")

    # F3 — Travel bonus is smaller for flats (they're sideways structures).
    if s0.price > 0:
        travel_pct = abs(sc.price - s0.price) / s0.price
        score += min(3.0, travel_pct * 15.0)

    # Calibration boost: flats need to compete with 6-swing impulses on the
    # same swing series. Without this they almost never win (4/6399 in report).
    score += 30  # baseline

    fibs = {
        "100%": s0.price,
        "B level": sb.price,
        "A level": sa.price,
    }
    # Flat completion → expect strong impulse in opposite direction
    next_targets = {
        "next impulse 161.8% (post-flat thrust)":
            s0.price + FIB_1618 * (sa.price - s0.price),
    }

    return WaveCandidate(
        pattern=f"flat_{direction}",
        swings_used=[s0, sa, sb, sc],
        labels=labels,
        rule_violations=violations,
        guideline_notes=notes,
        score=float(score),
        current_position=(
            f"completed {flat_subtype} {direction} flat — "
            f"next likely strong impulse in opposite direction"
            if not violations else
            f"{direction} flat hypothesis (with violations)"
        ),
        fibs=fibs,
        next_targets=next_targets,
    )


def _best_flat(
    swings: list[Swing], direction: Literal["up", "down"]
) -> WaveCandidate:
    if len(swings) < 4:
        return WaveCandidate(
            pattern=f"flat_{direction}", swings_used=swings,
            labels=[], score=0.0,
            current_position="not enough swings for flat",
        )
    best: WaveCandidate | None = None
    n = len(swings)
    e_min = max(4, n - WINDOW_SEARCH_DEPTH)
    for e in range(n, e_min - 1, -1):
        window = swings[e - 4:e]
        if len(window) != 4:
            continue
        cand = _score_flat_window(window, direction)
        if best is None or cand.score > best.score:
            best = cand
            bars_back = n - e
            if best.score > 0 and bars_back > 0:
                best.current_position += f" — completed {bars_back} swing(s) ago"
    return best or WaveCandidate(
        pattern=f"flat_{direction}", swings_used=[], labels=[], score=0.0,
    )


# ---------------------------------------------------------------------------
# Phase 1.2 — Contracting triangle (A-B-C-D-E)
# ---------------------------------------------------------------------------
def _score_triangle_window(
    swings: list[Swing], direction: Literal["up", "down"]
) -> WaveCandidate:
    """Score a 6-swing window (0+A+B+C+D+E) as a contracting triangle.

    `direction` here is the THRUST direction expected AFTER the triangle
    completes — opposite of the final leg E's direction.

    Triangle rules (Elliott):
      T0) 5 alternating sub-waves A-B-C-D-E.
      T1) Each successive leg shorter than the previous (contracting):
          len(A) > len(C) > len(E)  AND  len(B) > len(D)
          — i.e. the same-direction legs decrease, AND counter legs decrease.
      T2) E ≤ C (E never breaks beyond C) and B ≤ A — converging.
      T3) C is not the longest leg.
      T4) Travel: triangles are SIDEWAYS — net displacement should be small.

    Post-triangle thrust target: the height of the widest part of the
    triangle (≈ length of A) added to the breakout point (last swing E).
    """
    if len(swings) != 6:
        return WaveCandidate(
            pattern=f"triangle_{direction}", swings_used=swings,
            labels=[], score=0.0,
            current_position="window size != 6",
        )

    s0, sa, sb, sc, sd, se = swings
    labels = ["0", "A", "B", "C", "D", "E"]

    # For triangle_up (thrust UP after E), final leg E goes DOWN.
    # So the swing alternation starts with whatever, but at E we need:
    #   - triangle_up → E.kind = "low"   (price down → up thrust)
    #   - triangle_down → E.kind = "high" (price up → down thrust)
    # Working backwards: if E=low, then D=high, C=low, B=high, A=low, 0=high.
    # That's [high, low, high, low, high, low] for triangle_up.
    if direction == "up":
        kinds_expected = ["high", "low", "high", "low", "high", "low"]
    else:
        kinds_expected = ["low", "high", "low", "high", "low", "high"]

    if [s.kind for s in (s0, sa, sb, sc, sd, se)] != kinds_expected:
        return WaveCandidate(
            pattern=f"triangle_{direction}", swings_used=swings,
            labels=labels, score=0.0,
            rule_violations=["swing alternation does not match triangle direction"],
            current_position="structural mismatch",
        )

    leg_a = _abs_len(s0, sa)
    leg_b = _abs_len(sa, sb)
    leg_c = _abs_len(sb, sc)
    leg_d = _abs_len(sc, sd)
    leg_e = _abs_len(sd, se)

    violations: list[str] = []
    notes: list[str] = []
    score = 0.0

    # T1 — same-direction legs decrease (contracting).
    if not (leg_a > leg_c > leg_e):
        violations.append(
            f"T1a violated: same-direction legs not contracting "
            f"(A={leg_a:.4f}, C={leg_c:.4f}, E={leg_e:.4f})"
        )
    else:
        score += 18
        notes.append(f"contracting same-direction: A>C>E ({leg_a:.4f}>{leg_c:.4f}>{leg_e:.4f})")
    if not (leg_b > leg_d):
        violations.append(f"T1b violated: B ≤ D ({leg_b:.4f} ≤ {leg_d:.4f})")
    else:
        score += 12
        notes.append(f"contracting counter-direction: B>D ({leg_b:.4f}>{leg_d:.4f})")

    # T2 — E doesn't break beyond C (containment).
    if direction == "up":
        # triangle_up: C is a low, E is a low, E should be ABOVE C
        e_breaks_c = se.price < sc.price
    else:
        # triangle_down: C is a high, E is a high, E should be BELOW C
        e_breaks_c = se.price > sc.price
    if e_breaks_c:
        violations.append("T2 violated: E pierced C boundary — triangle broken")
    else:
        score += 12

    # T3 — C not the longest leg.
    if leg_c >= max(leg_a, leg_b, leg_d, leg_e):
        violations.append("T3 violated: C is the longest leg — atypical")
    else:
        score += 8

    # T4 — sideways travel. Net move should be small relative to widest leg.
    net = abs(se.price - s0.price)
    widest = max(leg_a, leg_b)
    if widest > 0 and net / widest < 0.5:
        score += 10
        notes.append(f"sideways structure: net travel {net/widest:.0%} of widest leg")

    # Fibonacci-style: each leg ~0.618 of the prior. Reward proximity.
    for label, prior, cur in [("B/A", leg_a, leg_b), ("C/B", leg_b, leg_c),
                                ("D/C", leg_c, leg_d), ("E/D", leg_d, leg_e)]:
        if prior > 0:
            ratio = cur / prior
            fib_score, fib_target = _nearest_fib_score(ratio, (FIB_618, FIB_500, FIB_786))
            score += 4 * fib_score
            notes.append(f"{label} = {ratio:.2f}× (Fib target {fib_target:.3f})")

    # Travel bonus is intentionally small or zero for triangles.
    # Calibration boost: triangles need to compete with impulses on same window.
    score += 25  # baseline

    # Post-triangle thrust target = breakout point + height of widest leg.
    # The thrust typically goes in the direction OPPOSITE to leg E.
    if direction == "up":
        thrust_target = se.price + leg_a    # thrust up
    else:
        thrust_target = se.price - leg_a    # thrust down

    fibs = {
        "A peak/trough": sa.price,
        "C peak/trough": sc.price,
        "E (breakout point)": se.price,
    }
    next_targets = {
        f"thrust target (= length of A from E)": thrust_target,
        f"thrust 1.618× extension": (
            se.price + FIB_1618 * leg_a if direction == "up"
            else se.price - FIB_1618 * leg_a
        ),
    }

    return WaveCandidate(
        pattern=f"triangle_{direction}",
        swings_used=[s0, sa, sb, sc, sd, se],
        labels=labels,
        rule_violations=violations,
        guideline_notes=notes,
        score=float(score),
        current_position=(
            f"contracting triangle complete — expect {direction.upper()} thrust "
            f"to {thrust_target:.4f}"
            if not violations else
            f"{direction} triangle hypothesis (with violations)"
        ),
        fibs=fibs,
        next_targets=next_targets,
        triangle_kind="contracting",
        thrust_target=thrust_target,
    )


def _best_triangle(
    swings: list[Swing], direction: Literal["up", "down"]
) -> WaveCandidate:
    if len(swings) < 6:
        return WaveCandidate(
            pattern=f"triangle_{direction}", swings_used=swings,
            labels=[], score=0.0,
            current_position="not enough swings for triangle",
        )
    best: WaveCandidate | None = None
    n = len(swings)
    e_min = max(6, n - WINDOW_SEARCH_DEPTH)
    for e in range(n, e_min - 1, -1):
        window = swings[e - 6:e]
        if len(window) != 6:
            continue
        cand = _score_triangle_window(window, direction)
        if best is None or cand.score > best.score:
            best = cand
            bars_back = n - e
            if best.score > 0 and bars_back > 0:
                best.current_position += f" — completed {bars_back} swing(s) ago"
    return best or WaveCandidate(
        pattern=f"triangle_{direction}", swings_used=[], labels=[], score=0.0,
    )


# ---------------------------------------------------------------------------
# Phase 2.1+2.2 — Diagonal wedges (leading + ending)
# ---------------------------------------------------------------------------
def _score_diagonal_window(
    swings: list[Swing],
    direction: Literal["up", "down"],
    kind: Literal["leading", "ending"],
) -> WaveCandidate:
    """Score a 6-swing window as a diagonal (wedge) — leading or ending.

    Diagonal characteristics:
      D0) 5-wave motive structure (same alternation as impulse).
      D1) Net direction sanity (same as R0).
      D2) Wave 2 doesn't fully retrace wave 1 (same as R1).
      D3) Wave 4 OVERLAPS wave 1 — this is the SIGNATURE diagonal feature.
          For an ending diagonal, overlap is required.
          For a leading diagonal, overlap is allowed but rare.
      D4) Wave 3 is NOT the shortest (same as R2).
      D5) Converging trendlines — wave 3 < wave 1 typically, wave 5 < wave 3.
          (Each successive same-direction wave shorter than prior.)
      D6) Travel: a wedge moves modestly — net price change is contained.

    Distinction leading vs ending:
      - Leading: appears at the START of an impulse or correction (wave 1 / A).
      - Ending: appears at the END of an impulse or correction (wave 5 / C).
      - At classification time we don't know which without broader context;
        for v1 we accept BOTH and let the higher-degree analysis pick.
    """
    if len(swings) != 6:
        return WaveCandidate(
            pattern=f"{kind}_diagonal_{direction}", swings_used=swings,
            labels=[], score=0.0,
            current_position="window size != 6",
        )

    s0, s1, s2, s3, s4, s5 = swings
    labels = ["0", "1", "2", "3", "4", "5"]

    if direction == "up":
        kinds_expected = ["low", "high", "low", "high", "low", "high"]
    else:
        kinds_expected = ["high", "low", "high", "low", "high", "low"]
    if [s.kind for s in (s0, s1, s2, s3, s4, s5)] != kinds_expected:
        return WaveCandidate(
            pattern=f"{kind}_diagonal_{direction}", swings_used=swings,
            labels=labels, score=0.0,
            rule_violations=["swing alternation does not match diagonal direction"],
            current_position="structural mismatch",
        )

    wave1 = _abs_len(s0, s1)
    wave2 = _abs_len(s1, s2)
    wave3 = _abs_len(s2, s3)
    wave4 = _abs_len(s3, s4)
    wave5 = _abs_len(s4, s5)

    violations: list[str] = []
    notes: list[str] = []
    score = 0.0

    # D1 — direction sanity
    if direction == "up" and s5.price <= s0.price:
        violations.append("D1 violated: up diagonal ends below start")
    elif direction == "down" and s5.price >= s0.price:
        violations.append("D1 violated: down diagonal ends above start")
    else:
        score += 18

    # D2 — wave 2 doesn't fully retrace wave 1
    retrace_2 = wave2 / max(wave1, 1e-9)
    if retrace_2 >= 1.0:
        violations.append(f"D2 violated: wave 2 retraced {retrace_2:.0%}")
    else:
        score += 10

    # D3 — Wave 4 overlap with wave 1. SIGNATURE feature.
    # In a normal impulse this kills the structure. In a diagonal it's REQUIRED
    # (for ending) or common (for leading).
    # Calibration insight: ending got +22, leading only +10 — caused 0 leading
    # diagonals to ever win. Equalized so each can win when its kind fits.
    if direction == "up":
        overlap = s4.price <= s1.price
    else:
        overlap = s4.price >= s1.price
    if overlap:
        score += 22 if kind == "ending" else 20
        notes.append(f"✓ wave 4 overlaps wave 1 — diagonal signature confirmed")
    else:
        if kind == "ending":
            # No overlap → this is not a diagonal, it's a normal impulse
            return WaveCandidate(
                pattern=f"{kind}_diagonal_{direction}", swings_used=swings,
                labels=labels, score=0.0,
                rule_violations=["D3 ending: no wave 4 / wave 1 overlap — "
                                  "this is a normal impulse, not an ending diagonal"],
                current_position="not a diagonal",
            )

    # D4 — wave 3 not the shortest
    if wave3 < wave1 and wave3 < wave5:
        violations.append("D4 violated: wave 3 is shortest")
    else:
        score += 8

    # D5 — converging structure: each successive same-direction wave smaller
    if not (wave1 >= wave3 >= wave5):
        # Not strictly converging — partial credit
        if wave5 < wave3:
            score += 5
            notes.append(f"partial convergence (5<3 but 1<3)")
        else:
            violations.append(
                f"D5 violated: waves not converging "
                f"(1={wave1:.4f}, 3={wave3:.4f}, 5={wave5:.4f})"
            )
    else:
        score += 15
        notes.append(f"converging: 1≥3≥5 ({wave1:.4f}≥{wave3:.4f}≥{wave5:.4f})")

    # D6 — modest travel (wedges move slowly)
    if s0.price > 0:
        travel_pct = abs(s5.price - s0.price) / s0.price
        score += min(4.0, travel_pct * 10.0)

    # Fibonacci wave 2 retracement is typically deep in diagonals (0.5-0.78)
    g_score, _ = _nearest_fib_score(retrace_2, (FIB_500, FIB_618, FIB_786))
    score += 6 * g_score
    notes.append(f"wave 2 retraced {retrace_2:.1%}")

    # Wave 4 retracement of wave 3
    retrace_4 = wave4 / max(wave3, 1e-9)
    g4_score, _ = _nearest_fib_score(retrace_4, (FIB_500, FIB_618, FIB_786))
    score += 5 * g4_score
    notes.append(f"wave 4 retraced {retrace_4:.1%} of wave 3")

    # Trading implication of diagonals:
    # - Ending diagonal: WAVE 5 IS COMPLETE → expect SHARP reversal
    # - Leading diagonal: this is wave 1 / A → expect correction then continuation
    if kind == "ending":
        position = (
            f"ending {direction} diagonal complete — STRONG reversal signal: "
            f"expect sharp {'decline' if direction == 'up' else 'rally'} "
            f"often back to start ({s0.price:.4f}) or beyond"
        )
        # Target after ending diagonal: typically retrace to start of wave 1
        next_targets = {
            "ending-diag target (retrace to start)": s0.price,
            "ending-diag deep target (-23.6%)":
                s0.price - 0.236 * (s5.price - s0.price),
        }
    else:
        position = (
            f"leading {direction} diagonal complete — wave 1 of larger structure; "
            f"expect correction (wave 2) retracing 50-78% before continuation"
        )
        if direction == "up":
            wave2_target = s5.price - FIB_500 * (s5.price - s0.price)
            wave2_deep = s5.price - FIB_786 * (s5.price - s0.price)
        else:
            wave2_target = s5.price + FIB_500 * (s0.price - s5.price)
            wave2_deep = s5.price + FIB_786 * (s0.price - s5.price)
        next_targets = {
            "wave 2 target (50% retrace)": wave2_target,
            "wave 2 deep (78.6% retrace)":  wave2_deep,
        }

    fibs = {
        "0%":   s5.price,
        "50%":  s5.price - 0.5 * (s5.price - s0.price),
        "100%": s0.price,
    }

    return WaveCandidate(
        pattern=f"{kind}_diagonal_{direction}",
        swings_used=[s0, s1, s2, s3, s4, s5],
        labels=labels,
        rule_violations=violations,
        guideline_notes=notes,
        score=float(score),
        current_position=position,
        fibs=fibs,
        next_targets=next_targets,
        is_diagonal=True,
        diagonal_kind=kind,
    )


def _best_diagonal(
    swings: list[Swing],
    direction: Literal["up", "down"],
    kind: Literal["leading", "ending"],
) -> WaveCandidate:
    if len(swings) < 6:
        return WaveCandidate(
            pattern=f"{kind}_diagonal_{direction}", swings_used=swings,
            labels=[], score=0.0,
        )
    best: WaveCandidate | None = None
    n = len(swings)
    e_min = max(6, n - WINDOW_SEARCH_DEPTH)
    for e in range(n, e_min - 1, -1):
        window = swings[e - 6:e]
        if len(window) != 6:
            continue
        cand = _score_diagonal_window(window, direction, kind)
        if best is None or cand.score > best.score:
            best = cand
            bars_back = n - e
            if best.score > 0 and bars_back > 0:
                best.current_position += f" — formed {bars_back} swing(s) ago"
    return best or WaveCandidate(
        pattern=f"{kind}_diagonal_{direction}",
        swings_used=[], labels=[], score=0.0,
    )


# ---------------------------------------------------------------------------
# Phase 2.4 — WXY complex correction (double three / double zigzag)
# ---------------------------------------------------------------------------
def _score_wxy_window(
    swings: list[Swing], direction: Literal["up", "down"]
) -> WaveCandidate:
    """Score an 8-swing window as a WXY complex correction.

    WXY structure: two corrections (W, Y) joined by a small connector (X).
      W = first ABC zigzag/flat (4 swings: 0 → A_w → B_w → C_w)
      X = single counter-move connector (1 swing: C_w → X)
      Y = second ABC zigzag/flat (3 swings: X → A_y → B_y → C_y)
    Total: 8 swings.

    Rules:
      WX0) Direction sanity.
      WX1) W is a valid 3-wave correction (ABC).
      WX2) X is shallow — retraces 0.382-0.786 of W.
      WX3) Y is roughly equal to W or 1.618× W.
    """
    if len(swings) != 8:
        return WaveCandidate(
            pattern=f"complex_correction_{direction}", swings_used=swings,
            labels=[], score=0.0,
            current_position="window size != 8",
        )

    s0, sa1, sb1, sc1, sx, sa2, sb2, sc2 = swings
    labels = ["0", "A", "B", "C(W)", "X", "A", "B", "C(Y)"]

    if direction == "up":
        kinds_expected = ["low", "high", "low", "high",
                          "low", "high", "low", "high"]
    else:
        kinds_expected = ["high", "low", "high", "low",
                          "high", "low", "high", "low"]
    if [s.kind for s in swings] != kinds_expected:
        return WaveCandidate(
            pattern=f"complex_correction_{direction}", swings_used=swings,
            labels=labels, score=0.0,
            rule_violations=["swing alternation does not match WXY direction"],
            current_position="structural mismatch",
        )

    w_length = abs(sc1.price - s0.price)
    x_length = abs(sx.price - sc1.price)
    y_length = abs(sc2.price - sx.price)

    violations: list[str] = []
    notes: list[str] = []
    score = 0.0

    # WX0 — direction sanity (net displacement matches label)
    if direction == "up" and sc2.price <= s0.price:
        violations.append("WX0 violated: up WXY ends below start")
    elif direction == "down" and sc2.price >= s0.price:
        violations.append("WX0 violated: down WXY ends above start")
    else:
        score += 15

    # WX1 — W is a valid ABC. Cheap structural check: W has 3 sub-waves with
    # B retracing some of A.
    wave_a_w = _abs_len(s0, sa1)
    wave_b_w = _abs_len(sa1, sb1)
    if wave_a_w > 0 and 0.2 <= wave_b_w / wave_a_w <= 0.95:
        score += 12
    else:
        violations.append(f"WX1 violated: W's B/A = {wave_b_w/max(wave_a_w,1e-9):.2f}")

    # WX2 — X is a shallow retracement of W
    if w_length > 0:
        x_ratio = x_length / w_length
        if 0.236 <= x_ratio <= 0.786:
            score += 18
            notes.append(f"X retraced {x_ratio:.1%} of W (within 23.6-78.6%)")
        else:
            violations.append(f"WX2 violated: X = {x_ratio:.1%} of W "
                              f"(outside 23.6-78.6%)")

    # WX3 — Y ≈ W (or 1.618× W for stretched Y)
    if w_length > 0:
        y_ratio = y_length / w_length
        y_fib_score, _ = _nearest_fib_score(y_ratio, (FIB_618, FIB_1000, FIB_1618))
        score += 20 * y_fib_score
        notes.append(f"Y/W = {y_ratio:.2f}× (target 1.0 or 1.618)")

    # Travel bonus
    if s0.price > 0:
        travel_pct = abs(sc2.price - s0.price) / s0.price
        score += min(4.0, travel_pct * 12.0)

    # Calibration boost: WXY needs 8 swings — most-rare pattern. Almost never
    # won in earlier report (5+2 out of 6399). Increased baseline to compete.
    score += 25  # baseline

    fibs = {
        "100%": s0.price,
        "W end": sc1.price,
        "X end": sx.price,
        "Y end (now)": sc2.price,
    }
    # After WXY, expect strong impulse opposite to correction direction
    if direction == "up":
        next_targets = {
            "post-WXY thrust 1.618× W":
                sc2.price - FIB_1618 * w_length,
        }
    else:
        next_targets = {
            "post-WXY thrust 1.618× W":
                sc2.price + FIB_1618 * w_length,
        }

    return WaveCandidate(
        pattern=f"complex_correction_{direction}",
        swings_used=list(swings),
        labels=labels,
        rule_violations=violations,
        guideline_notes=notes,
        score=float(score),
        current_position=(
            f"WXY double-three {direction} correction complete — "
            f"strong impulse in opposite direction expected"
            if not violations else
            f"{direction} WXY hypothesis (with violations)"
        ),
        fibs=fibs,
        next_targets=next_targets,
    )


def _best_wxy(
    swings: list[Swing], direction: Literal["up", "down"]
) -> WaveCandidate:
    if len(swings) < 8:
        return WaveCandidate(
            pattern=f"complex_correction_{direction}", swings_used=swings,
            labels=[], score=0.0,
        )
    best: WaveCandidate | None = None
    n = len(swings)
    e_min = max(8, n - WINDOW_SEARCH_DEPTH)
    for e in range(n, e_min - 1, -1):
        window = swings[e - 8:e]
        if len(window) != 8:
            continue
        cand = _score_wxy_window(window, direction)
        if best is None or cand.score > best.score:
            best = cand
            bars_back = n - e
            if best.score > 0 and bars_back > 0:
                best.current_position += f" — formed {bars_back} swing(s) ago"
    return best or WaveCandidate(
        pattern=f"complex_correction_{direction}",
        swings_used=[], labels=[], score=0.0,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def classify(swings: list[Swing], top_k: int = 3) -> list[WaveCandidate]:
    """Return ranked Elliott hypotheses for the current swing structure.

    We evaluate four canonical patterns by searching multiple recent
    windows for each:
      - impulse_up   (6 swings, sliding over last few endpoints)
      - impulse_down
      - correction_up   (4 swings)
      - correction_down

    The candidates are ranked by score descending. Caller picks the top
    one or shows multiple alongside each other.
    """
    candidates: list[WaveCandidate] = []
    if len(swings) >= 6:
        candidates.append(_best_impulse(swings, "up"))
        candidates.append(_best_impulse(swings, "down"))
        # Phase 1.2 — Triangles share the 6-swing window with impulses
        candidates.append(_best_triangle(swings, "up"))
        candidates.append(_best_triangle(swings, "down"))
        # Phase 2.1 + 2.2 — Diagonals (leading + ending)
        candidates.append(_best_diagonal(swings, "up",   "ending"))
        candidates.append(_best_diagonal(swings, "down", "ending"))
        candidates.append(_best_diagonal(swings, "up",   "leading"))
        candidates.append(_best_diagonal(swings, "down", "leading"))
    if len(swings) >= 4:
        candidates.append(_best_correction(swings, "up"))
        candidates.append(_best_correction(swings, "down"))
        # Phase 1.1 — Flats share the 4-swing window with zigzags
        candidates.append(_best_flat(swings, "up"))
        candidates.append(_best_flat(swings, "down"))
    if len(swings) >= 8:
        # Phase 2.4 — WXY complex correction (needs more swings)
        candidates.append(_best_wxy(swings, "up"))
        candidates.append(_best_wxy(swings, "down"))

    # Drop pure structural-mismatch zeroes if there are real candidates.
    real = [c for c in candidates if c.score > 0]
    candidates = real if real else candidates
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Multi-pattern labeling — annotate the WHOLE swing series, not just the tail
# ---------------------------------------------------------------------------
@dataclass
class LabeledSwing:
    """One annotated swing point inside a multi-pattern labeling.

    Unlike ``WaveCandidate.swings_used`` (which only labels the last detected
    pattern), this represents one swing inside a longer segmentation of the
    whole price history — letting the chart show wave labels at every
    structurally meaningful point, like a professional analyst would.
    """
    swing: Swing
    label: str            # e.g. "1", "3", "5" or "A", "B", "C"
    pattern: str          # which pattern this swing belongs to
    pattern_index: int    # 0-based ordinal of the pattern in left-to-right order
    is_up: bool           # direction of the parent pattern
    score: float          # parent pattern's score


def multi_label_swings(swings: list[Swing],
                       min_score: float = 40.0) -> list[LabeledSwing]:
    """Segment the entire swing series into back-to-back patterns and label
    every swing point.

    Greedy left-to-right algorithm:
      1. Start at swing[0].
      2. Try every pattern type starting from current position (impulse +
         zigzag + flat + triangle + diagonal + wxy). Pick the highest-scoring
         one whose score ≥ ``min_score``.
      3. Emit one ``LabeledSwing`` per swing in that pattern. Advance position
         to the END of the pattern's swings (so neighboring patterns share
         the boundary point).
      4. If no pattern scores high enough at this position, skip the swing
         (unlabeled) and try the next one.

    Returns a flat list of ``LabeledSwing`` covering as much of ``swings``
    as the detector could explain.
    """
    if len(swings) < 4:
        return []

    out: list[LabeledSwing] = []
    used_ids: set[int] = set()
    pos = 0
    pattern_index = 0
    n = len(swings)

    while pos < n - 3:
        slice_ = swings[pos:]
        if len(slice_) < 4:
            break

        # Run classifier on the remaining slice; it searches several windows
        # internally and returns the best match. We pick top-1.
        top_cands = classify(slice_, top_k=1)
        if not top_cands or top_cands[0].score < min_score \
                or not top_cands[0].swings_used or not top_cands[0].labels:
            pos += 1
            continue

        best = top_cands[0]
        # The pattern may start LATER than `pos` (classify slides windows).
        # Anchor `pos` to the first swing of the returned pattern.
        anchor = best.swings_used[0]
        try:
            anchor_pos = swings.index(anchor, pos)
        except ValueError:
            pos += 1
            continue

        # Skip if all swings of this pattern were already labeled
        # (avoids infinite re-emission of overlapping patterns)
        if all(id(s) in used_ids for s in best.swings_used):
            pos = anchor_pos + 1
            continue

        is_up = "up" in best.pattern
        for s, lab in zip(best.swings_used, best.labels):
            if id(s) in used_ids:
                continue
            out.append(LabeledSwing(
                swing=s, label=lab, pattern=best.pattern,
                pattern_index=pattern_index, is_up=is_up,
                score=best.score,
            ))
            used_ids.add(id(s))

        # Advance past this pattern (share the boundary swing with the next)
        new_pos = anchor_pos + len(best.swings_used) - 1
        if new_pos <= pos:
            pos += 1
        else:
            pos = new_pos
        pattern_index += 1

    # Sort by chronological position (bar_index) for stable rendering
    out.sort(key=lambda ls: ls.swing.bar_index)
    return out
