"""Модель v1: на случайном блуждании гейт закрыт, на плацебо мета-модель не «помогает», на тренде сигнал находится."""
from signal_lab.tests.test_lab import synth
from signal_lab.v1 import evaluate_v1, report_v1


def test_random_walk_no_gate():
    res = evaluate_v1(synth(momentum=0.0, seed=7), "4h", 48)
    assert not res["gate"] and not res["m2_improves"]
    assert "Гейт НЕ пройден" in report_v1(res, "rw")


def test_placebo_m2_does_not_improve():
    res = evaluate_v1(synth(momentum=0.35, seed=2), "4h", 48, placebo=True)
    assert not res["m2_improves"]


def test_trend_found_and_beats_always_long():
    res = evaluate_v1(synth(momentum=0.35, seed=2), "4h", 48)
    assert res["gate"] and res["beats"]
    assert res["signals_per_month"] >= 2
