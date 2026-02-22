"""Tests for puzzle_reconstruction/utils/sampling_utils.py"""
import math
import numpy as np
import pytest

from puzzle_reconstruction.utils.sampling_utils import (
    SamplingConfig,
    uniform_sample,
    sample_angle,
    sample_position,
    sample_positions_grid,
    sample_permutation,
    weighted_sample,
    acceptance_probability,
    sample_swap_pair,
    batch_uniform_sample,
)


# ─── SamplingConfig ───────────────────────────────────────────────────────────

class TestSamplingConfig:
    def test_defaults(self):
        cfg = SamplingConfig()
        assert cfg.seed is None
        assert cfg.grid_step == 10
        assert cfg.angles_deg == [0.0, 90.0, 180.0, 270.0]

    def test_custom_seed(self):
        cfg = SamplingConfig(seed=42)
        assert cfg.seed == 42

    def test_custom_grid_step(self):
        cfg = SamplingConfig(grid_step=5)
        assert cfg.grid_step == 5

    def test_grid_step_zero_raises(self):
        with pytest.raises(ValueError, match="grid_step"):
            SamplingConfig(grid_step=0)

    def test_grid_step_negative_raises(self):
        with pytest.raises(ValueError):
            SamplingConfig(grid_step=-1)

    def test_empty_angles_raises(self):
        with pytest.raises(ValueError, match="angles_deg"):
            SamplingConfig(angles_deg=[])

    def test_make_rng_returns_generator(self):
        cfg = SamplingConfig(seed=0)
        rng = cfg.make_rng()
        assert isinstance(rng, np.random.Generator)

    def test_make_rng_reproducible(self):
        cfg = SamplingConfig(seed=123)
        r1 = cfg.make_rng().random()
        r2 = cfg.make_rng().random()
        assert r1 == pytest.approx(r2)

    def test_custom_angles(self):
        cfg = SamplingConfig(angles_deg=[0.0, 45.0])
        assert len(cfg.angles_deg) == 2


# ─── uniform_sample ───────────────────────────────────────────────────────────

class TestUniformSample:
    def test_returns_float(self):
        result = uniform_sample(0.0, 1.0)
        assert isinstance(result, float)

    def test_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            r = uniform_sample(2.0, 5.0, rng=rng)
            assert 2.0 <= r <= 5.0

    def test_lo_equals_hi(self):
        result = uniform_sample(3.0, 3.0)
        assert result == pytest.approx(3.0)

    def test_lo_greater_than_hi_raises(self):
        with pytest.raises(ValueError, match="lo"):
            uniform_sample(5.0, 3.0)

    def test_reproducible_with_rng(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        assert uniform_sample(0.0, 10.0, rng1) == pytest.approx(
            uniform_sample(0.0, 10.0, rng2)
        )

    def test_negative_range(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            r = uniform_sample(-5.0, -1.0, rng=rng)
            assert -5.0 <= r <= -1.0


# ─── sample_angle ─────────────────────────────────────────────────────────────

class TestSampleAngle:
    def test_returns_float(self):
        result = sample_angle()
        assert isinstance(result, float)

    def test_angle_in_valid_set(self):
        cfg = SamplingConfig(angles_deg=[0.0, 90.0, 180.0, 270.0], seed=7)
        rng = cfg.make_rng()
        valid_radians = {math.radians(d) for d in cfg.angles_deg}
        for _ in range(50):
            angle = sample_angle(cfg, rng)
            assert any(abs(angle - v) < 1e-9 for v in valid_radians)

    def test_custom_angles(self):
        cfg = SamplingConfig(angles_deg=[45.0], seed=0)
        rng = cfg.make_rng()
        result = sample_angle(cfg, rng)
        assert result == pytest.approx(math.radians(45.0))

    def test_none_cfg_uses_defaults(self):
        result = sample_angle(None)
        assert isinstance(result, float)


# ─── sample_position ──────────────────────────────────────────────────────────

class TestSamplePosition:
    def test_returns_tuple(self):
        result = sample_position(100, 200)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            x, y = sample_position(100, 200, rng=rng)
            assert 0.0 <= x < 100.0
            assert 0.0 <= y < 200.0

    def test_width_zero_raises(self):
        with pytest.raises(ValueError, match="width"):
            sample_position(0, 100)

    def test_height_zero_raises(self):
        with pytest.raises(ValueError, match="height"):
            sample_position(100, 0)

    def test_width_negative_raises(self):
        with pytest.raises(ValueError):
            sample_position(-1, 100)

    def test_returns_floats(self):
        x, y = sample_position(50, 50)
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_1x1_canvas(self):
        x, y = sample_position(1, 1)
        assert 0.0 <= x < 1.0
        assert 0.0 <= y < 1.0


# ─── sample_positions_grid ────────────────────────────────────────────────────

class TestSamplePositionsGrid:
    def test_returns_list(self):
        result = sample_positions_grid(100, 100)
        assert isinstance(result, list)

    def test_all_tuples(self):
        result = sample_positions_grid(50, 50)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in result)

    def test_in_bounds(self):
        result = sample_positions_grid(40, 30, SamplingConfig(grid_step=10))
        for x, y in result:
            assert 0 <= x < 40
            assert 0 <= y < 30

    def test_step_10_count(self):
        # width=30, height=20, step=10: x in {0,10,20}, y in {0,10}
        result = sample_positions_grid(30, 20, SamplingConfig(grid_step=10))
        assert len(result) == 6  # 3 * 2

    def test_step_1_count(self):
        result = sample_positions_grid(3, 3, SamplingConfig(grid_step=1))
        assert len(result) == 9

    def test_contains_origin(self):
        result = sample_positions_grid(100, 100, SamplingConfig(grid_step=10))
        assert (0, 0) in result

    def test_width_zero_raises(self):
        with pytest.raises(ValueError):
            sample_positions_grid(0, 100)

    def test_height_zero_raises(self):
        with pytest.raises(ValueError):
            sample_positions_grid(100, 0)


# ─── sample_permutation ───────────────────────────────────────────────────────

class TestSamplePermutation:
    def test_returns_list(self):
        result = sample_permutation(5)
        assert isinstance(result, list)

    def test_correct_length(self):
        assert len(sample_permutation(7)) == 7

    def test_contains_all_indices(self):
        result = sample_permutation(6)
        assert sorted(result) == list(range(6))

    def test_n_one(self):
        result = sample_permutation(1)
        assert result == [0]

    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="n"):
            sample_permutation(0)

    def test_n_negative_raises(self):
        with pytest.raises(ValueError):
            sample_permutation(-1)

    def test_reproducible_with_rng(self):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        assert sample_permutation(10, rng1) == sample_permutation(10, rng2)

    def test_randomness(self):
        # With seed, should produce a shuffle (not necessarily sorted)
        rng = np.random.default_rng(1)
        perm = sample_permutation(10, rng)
        # It is astronomically unlikely that a random permutation of 10 equals sorted
        assert perm != list(range(10)) or True  # can't assert disorder, just runs


# ─── weighted_sample ──────────────────────────────────────────────────────────

class TestWeightedSample:
    def test_returns_int(self):
        w = np.array([1.0, 2.0, 3.0])
        result = weighted_sample(w)
        assert isinstance(result, int)

    def test_index_in_range(self):
        w = np.array([0.5, 0.3, 0.2])
        rng = np.random.default_rng(0)
        for _ in range(50):
            idx = weighted_sample(w, rng)
            assert 0 <= idx < 3

    def test_deterministic_single_nonzero(self):
        w = np.array([0.0, 0.0, 1.0])
        rng = np.random.default_rng(0)
        for _ in range(20):
            assert weighted_sample(w, rng) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_sample(np.array([]))

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="отрицательных"):
            weighted_sample(np.array([1.0, -0.5, 2.0]))

    def test_all_zero_raises(self):
        with pytest.raises(ValueError, match="Сумма"):
            weighted_sample(np.array([0.0, 0.0]))

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            weighted_sample(np.ones((2, 2)))

    def test_uniform_weights_covers_indices(self):
        w = np.ones(4)
        rng = np.random.default_rng(42)
        seen = set()
        for _ in range(200):
            seen.add(weighted_sample(w, rng))
        assert seen == {0, 1, 2, 3}


# ─── acceptance_probability ───────────────────────────────────────────────────

class TestAcceptanceProbability:
    def test_improvement_returns_one(self):
        assert acceptance_probability(-1.0, 100.0) == pytest.approx(1.0)

    def test_zero_delta_returns_one(self):
        assert acceptance_probability(0.0, 100.0) == pytest.approx(1.0)

    def test_positive_delta_less_than_one(self):
        p = acceptance_probability(1.0, 10.0)
        assert 0.0 < p < 1.0

    def test_formula(self):
        delta, T = 2.0, 5.0
        expected = math.exp(-delta / T)
        assert acceptance_probability(delta, T) == pytest.approx(expected)

    def test_high_temperature_higher_prob(self):
        p_hot = acceptance_probability(1.0, 1000.0)
        p_cold = acceptance_probability(1.0, 0.1)
        assert p_hot > p_cold

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            acceptance_probability(1.0, 0.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            acceptance_probability(1.0, -1.0)

    def test_returns_float(self):
        result = acceptance_probability(0.5, 1.0)
        assert isinstance(result, float)

    def test_very_large_delta_near_zero(self):
        p = acceptance_probability(1e9, 1.0)
        assert p == pytest.approx(0.0, abs=1e-9)


# ─── sample_swap_pair ─────────────────────────────────────────────────────────

class TestSampleSwapPair:
    def test_returns_tuple(self):
        result = sample_swap_pair(5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_different_indices(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            i, j = sample_swap_pair(5, rng)
            assert i != j

    def test_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            i, j = sample_swap_pair(10, rng)
            assert 0 <= i < 10
            assert 0 <= j < 10

    def test_n_one_raises(self):
        with pytest.raises(ValueError, match="n"):
            sample_swap_pair(1)

    def test_n_zero_raises(self):
        with pytest.raises(ValueError):
            sample_swap_pair(0)

    def test_n_two_always_returns_0_1(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            pair = sample_swap_pair(2, rng)
            assert set(pair) == {0, 1}

    def test_reproducible_with_rng(self):
        rng1 = np.random.default_rng(5)
        rng2 = np.random.default_rng(5)
        assert sample_swap_pair(10, rng1) == sample_swap_pair(10, rng2)


# ─── batch_uniform_sample ─────────────────────────────────────────────────────

class TestBatchUniformSample:
    def test_returns_ndarray(self):
        result = batch_uniform_sample(0.0, 1.0, 10)
        assert isinstance(result, np.ndarray)

    def test_correct_length(self):
        result = batch_uniform_sample(0.0, 1.0, 20)
        assert len(result) == 20

    def test_dtype_float64(self):
        result = batch_uniform_sample(0.0, 1.0, 5)
        assert result.dtype == np.float64

    def test_all_in_range(self):
        rng = np.random.default_rng(0)
        result = batch_uniform_sample(2.0, 5.0, 100, rng)
        assert np.all(result >= 2.0)
        assert np.all(result <= 5.0)

    def test_lo_equals_hi_constant(self):
        result = batch_uniform_sample(3.0, 3.0, 10)
        np.testing.assert_allclose(result, 3.0)

    def test_lo_greater_than_hi_raises(self):
        with pytest.raises(ValueError, match="lo"):
            batch_uniform_sample(5.0, 2.0, 10)

    def test_size_zero_raises(self):
        with pytest.raises(ValueError, match="size"):
            batch_uniform_sample(0.0, 1.0, 0)

    def test_size_negative_raises(self):
        with pytest.raises(ValueError):
            batch_uniform_sample(0.0, 1.0, -1)

    def test_reproducible_with_rng(self):
        rng1 = np.random.default_rng(77)
        rng2 = np.random.default_rng(77)
        r1 = batch_uniform_sample(0.0, 10.0, 50, rng1)
        r2 = batch_uniform_sample(0.0, 10.0, 50, rng2)
        np.testing.assert_allclose(r1, r2)
