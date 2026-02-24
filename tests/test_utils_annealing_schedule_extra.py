"""Extra tests for puzzle_reconstruction/utils/annealing_schedule.py."""
from __future__ import annotations

import math
import pytest
import numpy as np

from puzzle_reconstruction.utils.annealing_schedule import (
    ScheduleConfig,
    TemperatureRecord,
    linear_schedule,
    geometric_schedule,
    exponential_schedule,
    cosine_schedule,
    stepped_schedule,
    get_temperature,
    estimate_steps,
    batch_temperatures,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _cfg(**kw):
    defaults = dict(t_start=1.0, t_end=1e-2, n_steps=10, kind="geometric")
    defaults.update(kw)
    return ScheduleConfig(**defaults)


def _temps(schedule):
    return [r.temperature for r in schedule]


# ─── ScheduleConfig (extra) ───────────────────────────────────────────────────

class TestScheduleConfigExtra:
    def test_large_t_start_ok(self):
        cfg = ScheduleConfig(t_start=1000.0, t_end=1.0)
        assert cfg.t_start == pytest.approx(1000.0)

    def test_large_n_steps_ok(self):
        cfg = _cfg(n_steps=100000)
        assert cfg.n_steps == 100000

    def test_step_size_1_default(self):
        cfg = _cfg()
        assert cfg.step_size >= 1

    def test_step_size_large_ok(self):
        cfg = _cfg(n_steps=100, step_size=10)
        assert cfg.step_size == 10

    def test_cooling_rate_less_than_one(self):
        cfg = _cfg(t_start=1.0, t_end=0.01, n_steps=10)
        assert cfg.cooling_rate < 1.0

    def test_cooling_rate_positive(self):
        cfg = _cfg()
        assert cfg.cooling_rate > 0.0

    def test_all_five_kinds(self):
        for k in ("linear", "geometric", "exponential", "cosine", "stepped"):
            cfg = _cfg(kind=k)
            assert cfg.kind == k

    def test_n_steps_one_cooling_rate(self):
        cfg = _cfg(t_start=4.0, t_end=0.04, n_steps=1)
        assert cfg.cooling_rate == pytest.approx(0.01)

    def test_t_end_close_to_t_start_high_cooling_rate(self):
        cfg = _cfg(t_start=1.0, t_end=0.999, n_steps=10)
        assert cfg.cooling_rate > 0.9


# ─── TemperatureRecord (extra) ────────────────────────────────────────────────

class TestTemperatureRecordExtra:
    def test_large_step_ok(self):
        r = TemperatureRecord(step=999, temperature=0.001, progress=1.0)
        assert r.step == 999

    def test_high_temperature_ok(self):
        r = TemperatureRecord(step=0, temperature=1000.0, progress=0.0)
        assert r.temperature == pytest.approx(1000.0)

    def test_progress_mid_ok(self):
        r = TemperatureRecord(step=5, temperature=0.5, progress=0.5)
        assert r.progress == pytest.approx(0.5)

    def test_is_cooling_property_true(self):
        r = TemperatureRecord(step=5, temperature=0.5, progress=0.5)
        assert r.temperature > 0.0

    def test_step_progress_consistency(self):
        cfg = _cfg(n_steps=10, kind="linear")
        records = linear_schedule(cfg)
        for r in records:
            assert 0.0 <= r.progress <= 1.0

    def test_temperature_always_positive(self):
        cfg = _cfg(n_steps=10, kind="geometric")
        for r in geometric_schedule(cfg):
            assert r.temperature > 0.0


# ─── linear_schedule (extra) ──────────────────────────────────────────────────

class TestLinearScheduleExtra:
    def test_equidistant_steps(self):
        cfg = _cfg(kind="linear", t_start=1.0, t_end=0.0 + 1e-9, n_steps=11)
        temps = _temps(linear_schedule(cfg))
        diffs = [temps[i] - temps[i + 1] for i in range(len(temps) - 1)]
        assert all(abs(d - diffs[0]) < 1e-8 for d in diffs)

    def test_progress_midpoint(self):
        cfg = _cfg(kind="linear", n_steps=11)
        records = linear_schedule(cfg)
        assert records[5].progress == pytest.approx(0.5)

    def test_step_indices_sequential(self):
        cfg = _cfg(kind="linear", n_steps=5)
        records = linear_schedule(cfg)
        for i, r in enumerate(records):
            assert r.step == i

    def test_large_n_steps_length(self):
        cfg = _cfg(kind="linear", n_steps=1000)
        assert len(linear_schedule(cfg)) == 1000

    def test_temperatures_between_t_end_and_t_start(self):
        cfg = _cfg(kind="linear", t_start=10.0, t_end=0.1, n_steps=20)
        for r in linear_schedule(cfg):
            assert 0.1 <= r.temperature <= 10.0


# ─── geometric_schedule (extra) ───────────────────────────────────────────────

class TestGeometricScheduleExtra:
    def test_ratio_consistent_across_all_steps(self):
        cfg = _cfg(kind="geometric", t_start=1.0, t_end=0.01, n_steps=11)
        temps = _temps(geometric_schedule(cfg))
        ratios = [temps[i + 1] / temps[i] for i in range(len(temps) - 1)]
        first_ratio = ratios[0]
        for r in ratios:
            assert r == pytest.approx(first_ratio, rel=1e-6)

    def test_temperatures_above_t_end(self):
        cfg = _cfg(kind="geometric", t_start=2.0, t_end=0.01, n_steps=50)
        for r in geometric_schedule(cfg):
            assert r.temperature >= 0.01 - 1e-10

    def test_step_indices_sequential(self):
        cfg = _cfg(kind="geometric", n_steps=8)
        for i, r in enumerate(geometric_schedule(cfg)):
            assert r.step == i

    def test_large_temperature_range(self):
        cfg = _cfg(kind="geometric", t_start=1000.0, t_end=0.001, n_steps=100)
        records = geometric_schedule(cfg)
        assert len(records) == 100
        assert records[0].temperature == pytest.approx(1000.0)


# ─── exponential_schedule (extra) ─────────────────────────────────────────────

class TestExponentialScheduleExtra:
    def test_first_equals_t_start(self):
        cfg = _cfg(kind="exponential", t_start=5.0, t_end=0.05, n_steps=20)
        records = exponential_schedule(cfg)
        assert records[0].temperature == pytest.approx(5.0)

    def test_temperatures_positive(self):
        cfg = _cfg(kind="exponential", n_steps=20)
        for r in exponential_schedule(cfg):
            assert r.temperature > 0.0

    def test_monotone_decreasing(self):
        cfg = _cfg(kind="exponential", n_steps=15)
        temps = _temps(exponential_schedule(cfg))
        for a, b in zip(temps, temps[1:]):
            assert a >= b

    def test_large_step_count(self):
        cfg = _cfg(kind="exponential", n_steps=500)
        assert len(exponential_schedule(cfg)) == 500

    def test_step_indices_sequential(self):
        cfg = _cfg(kind="exponential", n_steps=6)
        for i, r in enumerate(exponential_schedule(cfg)):
            assert r.step == i


# ─── cosine_schedule (extra) ──────────────────────────────────────────────────

class TestCosineScheduleExtra:
    def test_temperatures_between_t_end_and_t_start(self):
        cfg = _cfg(kind="cosine", t_start=4.0, t_end=0.01, n_steps=30)
        for r in cosine_schedule(cfg):
            assert 0.01 - 1e-9 <= r.temperature <= 4.0 + 1e-9

    def test_progress_last_equals_one(self):
        cfg = _cfg(kind="cosine", n_steps=20)
        records = cosine_schedule(cfg)
        assert records[-1].progress == pytest.approx(1.0)

    def test_step_indices_sequential(self):
        cfg = _cfg(kind="cosine", n_steps=7)
        for i, r in enumerate(cosine_schedule(cfg)):
            assert r.step == i

    def test_large_n_steps(self):
        cfg = _cfg(kind="cosine", n_steps=200)
        assert len(cosine_schedule(cfg)) == 200


# ─── stepped_schedule (extra) ─────────────────────────────────────────────────

class TestSteppedScheduleExtra:
    def test_plateaus_of_correct_size(self):
        cfg = _cfg(kind="stepped", t_start=1.0, t_end=0.01,
                   n_steps=9, step_size=3)
        temps = _temps(stepped_schedule(cfg))
        # First three should be equal
        assert temps[0] == pytest.approx(temps[1])
        assert temps[1] == pytest.approx(temps[2])

    def test_step_size_1_returns_correct_count(self):
        cfg = _cfg(kind="stepped", n_steps=10, step_size=1)
        temps = _temps(stepped_schedule(cfg))
        assert len(temps) == 10

    def test_step_indices_sequential(self):
        cfg = _cfg(kind="stepped", n_steps=6, step_size=2)
        for i, r in enumerate(stepped_schedule(cfg)):
            assert r.step == i

    def test_large_step_size_constant(self):
        cfg = _cfg(kind="stepped", n_steps=6, step_size=10)
        temps = _temps(stepped_schedule(cfg))
        # All within one plateau → all equal
        assert all(t == pytest.approx(temps[0]) for t in temps)


# ─── get_temperature (extra) ──────────────────────────────────────────────────

class TestGetTemperatureExtra:
    def test_middle_step_positive(self):
        cfg = _cfg(kind="linear", n_steps=10)
        t = get_temperature(5, cfg)
        assert t > 0.0

    def test_consistent_with_linear_schedule(self):
        cfg = _cfg(kind="linear", n_steps=10)
        records = linear_schedule(cfg)
        for i in range(10):
            assert get_temperature(i, cfg) == pytest.approx(records[i].temperature)

    def test_consistent_with_geometric_schedule(self):
        cfg = _cfg(kind="geometric", n_steps=10)
        records = geometric_schedule(cfg)
        for i in range(10):
            assert get_temperature(i, cfg) == pytest.approx(records[i].temperature)

    def test_consistent_with_exponential(self):
        cfg = _cfg(kind="exponential", n_steps=8)
        records = exponential_schedule(cfg)
        for i in range(8):
            assert get_temperature(i, cfg) == pytest.approx(records[i].temperature)

    def test_consistent_with_stepped(self):
        cfg = _cfg(kind="stepped", n_steps=6, step_size=2)
        records = stepped_schedule(cfg)
        for i in range(6):
            assert get_temperature(i, cfg) == pytest.approx(records[i].temperature)

    def test_monotone_for_linear(self):
        cfg = _cfg(kind="linear", n_steps=5)
        temps = [get_temperature(i, cfg) for i in range(5)]
        for a, b in zip(temps, temps[1:]):
            assert a >= b


# ─── estimate_steps (extra) ───────────────────────────────────────────────────

class TestEstimateStepsExtra:
    def test_result_at_least_one(self):
        assert estimate_steps(10.0, 9.0, 0.5) >= 1

    def test_smaller_target_more_steps(self):
        s1 = estimate_steps(1.0, 0.1, 0.9)
        s2 = estimate_steps(1.0, 0.01, 0.9)
        assert s2 > s1

    def test_low_alpha_fewer_steps(self):
        s_slow = estimate_steps(1.0, 0.01, 0.99)
        s_fast = estimate_steps(1.0, 0.01, 0.5)
        assert s_fast < s_slow

    def test_positive_result(self):
        result = estimate_steps(5.0, 0.5, 0.8)
        assert result > 0

    def test_alpha_close_to_one_many_steps(self):
        result = estimate_steps(1.0, 0.001, 0.999)
        assert result > 100

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError):
            estimate_steps(1.0, 0.01, -0.1)


# ─── batch_temperatures (extra) ───────────────────────────────────────────────

class TestBatchTemperaturesExtra:
    def test_single_step_zero(self):
        cfg = _cfg(kind="linear", n_steps=10)
        result = batch_temperatures(np.array([0]), cfg)
        assert result[0] == pytest.approx(cfg.t_start)

    def test_single_step_last(self):
        cfg = _cfg(kind="linear", n_steps=10)
        result = batch_temperatures(np.array([9]), cfg)
        assert result[0] == pytest.approx(cfg.t_end)

    def test_all_steps_match_schedule(self):
        cfg = _cfg(kind="geometric", n_steps=8)
        records = geometric_schedule(cfg)
        steps = np.arange(8)
        result = batch_temperatures(steps, cfg)
        for i in range(8):
            assert result[i] == pytest.approx(records[i].temperature)

    def test_all_positive(self):
        cfg = _cfg(n_steps=20)
        result = batch_temperatures(np.arange(20), cfg)
        assert (result > 0).all()

    def test_cosine_batch(self):
        cfg = _cfg(kind="cosine", n_steps=10)
        result = batch_temperatures(np.array([0, 5, 9]), cfg)
        assert len(result) == 3
        assert result[0] == pytest.approx(cfg.t_start)

    def test_negative_step_raises(self):
        cfg = _cfg(n_steps=10)
        with pytest.raises(ValueError):
            batch_temperatures(np.array([-1]), cfg)
