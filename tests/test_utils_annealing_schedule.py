"""Tests for puzzle_reconstruction/utils/annealing_schedule.py"""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _cfg(**kw):
    defaults = dict(t_start=1.0, t_end=1e-2, n_steps=10, kind="geometric")
    defaults.update(kw)
    return ScheduleConfig(**defaults)


def _temps(schedule):
    return [r.temperature for r in schedule]


# ─── TestScheduleConfig ───────────────────────────────────────────────────────

class TestScheduleConfig:
    def test_defaults(self):
        cfg = ScheduleConfig()
        assert cfg.t_start == pytest.approx(1.0)
        assert cfg.t_end == pytest.approx(1e-3)
        assert cfg.n_steps == 1000
        assert cfg.kind == "geometric"
        assert cfg.step_size == 1

    def test_t_start_zero_raises(self):
        with pytest.raises(ValueError):
            ScheduleConfig(t_start=0.0, t_end=1e-3)

    def test_t_start_negative_raises(self):
        with pytest.raises(ValueError):
            ScheduleConfig(t_start=-1.0, t_end=1e-3)

    def test_t_end_zero_raises(self):
        with pytest.raises(ValueError):
            ScheduleConfig(t_end=0.0)

    def test_t_end_gte_t_start_raises(self):
        with pytest.raises(ValueError):
            ScheduleConfig(t_start=1.0, t_end=2.0)

    def test_t_end_equal_t_start_raises(self):
        with pytest.raises(ValueError):
            ScheduleConfig(t_start=1.0, t_end=1.0)

    def test_n_steps_zero_raises(self):
        with pytest.raises(ValueError):
            _cfg(n_steps=0)

    def test_n_steps_negative_raises(self):
        with pytest.raises(ValueError):
            _cfg(n_steps=-5)

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            _cfg(kind="quadratic")

    def test_step_size_zero_raises(self):
        with pytest.raises(ValueError):
            _cfg(step_size=0)

    def test_all_valid_kinds(self):
        for k in ("linear", "geometric", "exponential", "cosine", "stepped"):
            cfg = _cfg(kind=k)
            assert cfg.kind == k

    def test_cooling_rate_formula(self):
        cfg = _cfg(t_start=1.0, t_end=0.01, n_steps=11)
        expected = (0.01 / 1.0) ** (1.0 / 10)
        assert cfg.cooling_rate == pytest.approx(expected)

    def test_cooling_rate_n_steps_1(self):
        cfg = _cfg(t_start=2.0, t_end=1.0, n_steps=1)
        # n_steps=1: t_end/t_start
        assert cfg.cooling_rate == pytest.approx(0.5)


# ─── TestTemperatureRecord ────────────────────────────────────────────────────

class TestTemperatureRecord:
    def test_construction(self):
        r = TemperatureRecord(step=5, temperature=0.5, progress=0.5)
        assert r.step == 5
        assert r.temperature == pytest.approx(0.5)
        assert r.progress == pytest.approx(0.5)

    def test_step_negative_raises(self):
        with pytest.raises(ValueError):
            TemperatureRecord(step=-1, temperature=1.0, progress=0.0)

    def test_temperature_zero_raises(self):
        with pytest.raises(ValueError):
            TemperatureRecord(step=0, temperature=0.0, progress=0.0)

    def test_temperature_negative_raises(self):
        with pytest.raises(ValueError):
            TemperatureRecord(step=0, temperature=-0.1, progress=0.0)

    def test_progress_negative_raises(self):
        with pytest.raises(ValueError):
            TemperatureRecord(step=0, temperature=1.0, progress=-0.1)

    def test_progress_above_one_raises(self):
        with pytest.raises(ValueError):
            TemperatureRecord(step=0, temperature=1.0, progress=1.1)

    def test_progress_zero_ok(self):
        r = TemperatureRecord(step=0, temperature=1.0, progress=0.0)
        assert r.progress == 0.0

    def test_progress_one_ok(self):
        r = TemperatureRecord(step=9, temperature=0.01, progress=1.0)
        assert r.progress == pytest.approx(1.0)


# ─── TestLinearSchedule ───────────────────────────────────────────────────────

class TestLinearSchedule:
    def test_length_matches_n_steps(self):
        cfg = _cfg(kind="linear", n_steps=20)
        assert len(linear_schedule(cfg)) == 20

    def test_first_is_t_start(self):
        cfg = _cfg(kind="linear", t_start=5.0, t_end=0.1)
        assert linear_schedule(cfg)[0].temperature == pytest.approx(5.0)

    def test_last_is_t_end(self):
        cfg = _cfg(kind="linear", t_start=5.0, t_end=0.1)
        assert linear_schedule(cfg)[-1].temperature == pytest.approx(0.1)

    def test_monotone_decreasing(self):
        cfg = _cfg(kind="linear", n_steps=20)
        temps = _temps(linear_schedule(cfg))
        for a, b in zip(temps, temps[1:]):
            assert a >= b

    def test_single_step(self):
        cfg = _cfg(kind="linear", n_steps=1)
        records = linear_schedule(cfg)
        assert len(records) == 1
        assert records[0].step == 0

    def test_progress_first_zero(self):
        cfg = _cfg(kind="linear")
        assert linear_schedule(cfg)[0].progress == pytest.approx(0.0)

    def test_progress_last_one(self):
        cfg = _cfg(kind="linear")
        assert linear_schedule(cfg)[-1].progress == pytest.approx(1.0)

    def test_all_temperatures_positive(self):
        cfg = _cfg(kind="linear", n_steps=10)
        for r in linear_schedule(cfg):
            assert r.temperature > 0


# ─── TestGeometricSchedule ────────────────────────────────────────────────────

class TestGeometricSchedule:
    def test_length_matches_n_steps(self):
        cfg = _cfg(kind="geometric", n_steps=15)
        assert len(geometric_schedule(cfg)) == 15

    def test_first_is_t_start(self):
        cfg = _cfg(kind="geometric", t_start=2.0, t_end=0.01)
        assert geometric_schedule(cfg)[0].temperature == pytest.approx(2.0)

    def test_last_approx_t_end(self):
        cfg = _cfg(kind="geometric", t_start=2.0, t_end=0.01, n_steps=100)
        assert geometric_schedule(cfg)[-1].temperature == pytest.approx(0.01, rel=0.01)

    def test_monotone_decreasing(self):
        cfg = _cfg(kind="geometric", n_steps=20)
        temps = _temps(geometric_schedule(cfg))
        for a, b in zip(temps, temps[1:]):
            assert a >= b

    def test_geometric_ratio(self):
        cfg = _cfg(kind="geometric", t_start=1.0, t_end=0.01, n_steps=11)
        records = geometric_schedule(cfg)
        ratio = records[1].temperature / records[0].temperature
        assert ratio == pytest.approx(cfg.cooling_rate, rel=1e-6)

    def test_all_temperatures_positive(self):
        cfg = _cfg(kind="geometric", n_steps=10)
        for r in geometric_schedule(cfg):
            assert r.temperature > 0


# ─── TestExponentialSchedule ──────────────────────────────────────────────────

class TestExponentialSchedule:
    def test_length_matches_n_steps(self):
        cfg = _cfg(kind="exponential", n_steps=12)
        assert len(exponential_schedule(cfg)) == 12

    def test_first_is_t_start(self):
        cfg = _cfg(kind="exponential", t_start=3.0, t_end=0.01)
        assert exponential_schedule(cfg)[0].temperature == pytest.approx(3.0)

    def test_last_approx_t_end(self):
        cfg = _cfg(kind="exponential", t_start=3.0, t_end=0.01, n_steps=100)
        assert exponential_schedule(cfg)[-1].temperature == pytest.approx(0.01, rel=0.01)

    def test_monotone_decreasing(self):
        cfg = _cfg(kind="exponential", n_steps=20)
        temps = _temps(exponential_schedule(cfg))
        for a, b in zip(temps, temps[1:]):
            assert a >= b

    def test_single_step_ok(self):
        cfg = _cfg(kind="exponential", n_steps=1)
        records = exponential_schedule(cfg)
        assert len(records) == 1

    def test_all_temperatures_positive(self):
        cfg = _cfg(kind="exponential", n_steps=10)
        for r in exponential_schedule(cfg):
            assert r.temperature > 0


# ─── TestCosineSchedule ───────────────────────────────────────────────────────

class TestCosineSchedule:
    def test_length_matches_n_steps(self):
        cfg = _cfg(kind="cosine", n_steps=16)
        assert len(cosine_schedule(cfg)) == 16

    def test_first_approx_t_start(self):
        cfg = _cfg(kind="cosine", t_start=4.0, t_end=0.01)
        # cos(0) = 1 → t_end + 0.5*(t_start-t_end)*2 = t_start
        assert cosine_schedule(cfg)[0].temperature == pytest.approx(4.0)

    def test_last_approx_t_end(self):
        cfg = _cfg(kind="cosine", t_start=4.0, t_end=0.01, n_steps=100)
        # cos(pi) = -1 → t_end + 0 = t_end
        assert cosine_schedule(cfg)[-1].temperature == pytest.approx(0.01, rel=0.01)

    def test_all_temperatures_nonneg(self):
        cfg = _cfg(kind="cosine", n_steps=20)
        for r in cosine_schedule(cfg):
            assert r.temperature > 0

    def test_monotone_decreasing(self):
        cfg = _cfg(kind="cosine", n_steps=30)
        temps = _temps(cosine_schedule(cfg))
        for a, b in zip(temps, temps[1:]):
            assert a >= b - 1e-12  # non-strict


# ─── TestSteppedSchedule ──────────────────────────────────────────────────────

class TestSteppedSchedule:
    def test_length_matches_n_steps(self):
        cfg = _cfg(kind="stepped", n_steps=15, step_size=3)
        assert len(stepped_schedule(cfg)) == 15

    def test_first_is_t_start(self):
        cfg = _cfg(kind="stepped", t_start=2.0, t_end=0.01, step_size=2)
        assert stepped_schedule(cfg)[0].temperature == pytest.approx(2.0)

    def test_constant_within_step(self):
        cfg = _cfg(kind="stepped", t_start=1.0, t_end=0.01,
                   n_steps=6, step_size=2)
        records = stepped_schedule(cfg)
        # Steps 0,1 are plateau 0; same temperature
        assert records[0].temperature == pytest.approx(records[1].temperature)

    def test_all_temperatures_positive(self):
        cfg = _cfg(kind="stepped", n_steps=15, step_size=3)
        for r in stepped_schedule(cfg):
            assert r.temperature > 0

    def test_monotone_non_increasing(self):
        cfg = _cfg(kind="stepped", n_steps=10, step_size=2)
        temps = _temps(stepped_schedule(cfg))
        for a, b in zip(temps, temps[1:]):
            assert a >= b - 1e-12


# ─── TestGetTemperature ───────────────────────────────────────────────────────

class TestGetTemperature:
    def test_step_zero_returns_t_start(self):
        cfg = _cfg(kind="linear", t_start=5.0, t_end=0.1)
        assert get_temperature(0, cfg) == pytest.approx(5.0)

    def test_step_last_returns_t_end(self):
        cfg = _cfg(kind="linear", t_start=5.0, t_end=0.1, n_steps=10)
        assert get_temperature(9, cfg) == pytest.approx(0.1)

    def test_out_of_bounds_negative_raises(self):
        cfg = _cfg()
        with pytest.raises(ValueError):
            get_temperature(-1, cfg)

    def test_out_of_bounds_too_large_raises(self):
        cfg = _cfg(n_steps=10)
        with pytest.raises(ValueError):
            get_temperature(10, cfg)

    def test_matches_schedule(self):
        cfg = _cfg(kind="cosine", n_steps=20)
        records = cosine_schedule(cfg)
        for i in range(20):
            assert get_temperature(i, cfg) == pytest.approx(records[i].temperature)

    def test_works_for_all_kinds(self):
        for kind in ("linear", "geometric", "exponential", "cosine", "stepped"):
            cfg = _cfg(kind=kind, n_steps=5)
            t = get_temperature(2, cfg)
            assert t > 0


# ─── TestEstimateSteps ────────────────────────────────────────────────────────

class TestEstimateSteps:
    def test_returns_int(self):
        result = estimate_steps(1.0, 0.01, 0.9)
        assert isinstance(result, int)

    def test_at_least_one(self):
        assert estimate_steps(1.0, 0.999, 0.5) >= 1

    def test_basic_formula(self):
        t_start, t_target, alpha = 1.0, 0.01, 0.9
        k = math.log(t_target / t_start) / math.log(alpha)
        expected = max(1, math.ceil(k))
        assert estimate_steps(t_start, t_target, alpha) == expected

    def test_t_start_zero_raises(self):
        with pytest.raises(ValueError):
            estimate_steps(0.0, 0.01, 0.9)

    def test_t_target_gte_t_start_raises(self):
        with pytest.raises(ValueError):
            estimate_steps(1.0, 1.0, 0.9)

    def test_t_target_zero_raises(self):
        with pytest.raises(ValueError):
            estimate_steps(1.0, 0.0, 0.9)

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            estimate_steps(1.0, 0.01, 0.0)

    def test_alpha_one_raises(self):
        with pytest.raises(ValueError):
            estimate_steps(1.0, 0.01, 1.0)

    def test_faster_cooling_fewer_steps(self):
        slow = estimate_steps(1.0, 0.01, 0.99)
        fast = estimate_steps(1.0, 0.01, 0.5)
        assert fast < slow


# ─── TestBatchTemperatures ────────────────────────────────────────────────────

class TestBatchTemperatures:
    def test_returns_float64(self):
        cfg = _cfg(n_steps=10)
        result = batch_temperatures(np.array([0, 5, 9]), cfg)
        assert result.dtype == np.float64

    def test_length_matches(self):
        cfg = _cfg(n_steps=10)
        result = batch_temperatures(np.array([0, 3, 7]), cfg)
        assert len(result) == 3

    def test_empty_steps(self):
        cfg = _cfg(n_steps=10)
        result = batch_temperatures(np.array([], dtype=int), cfg)
        assert len(result) == 0

    def test_2d_raises(self):
        cfg = _cfg(n_steps=10)
        with pytest.raises(ValueError):
            batch_temperatures(np.array([[0, 1]]), cfg)

    def test_out_of_bounds_raises(self):
        cfg = _cfg(n_steps=10)
        with pytest.raises(ValueError):
            batch_temperatures(np.array([0, 10]), cfg)

    def test_values_match_get_temperature(self):
        cfg = _cfg(kind="linear", n_steps=10)
        steps = np.array([0, 4, 9])
        result = batch_temperatures(steps, cfg)
        for i, s in enumerate(steps):
            assert result[i] == pytest.approx(get_temperature(int(s), cfg))

    def test_all_positive(self):
        cfg = _cfg(n_steps=10)
        result = batch_temperatures(np.arange(10), cfg)
        assert (result > 0).all()
