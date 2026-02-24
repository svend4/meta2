"""Extra tests for puzzle_reconstruction/utils/sampling_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

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

class TestSamplingConfigExtra:
    def test_default_seed_none(self):
        assert SamplingConfig().seed is None

    def test_default_grid_step(self):
        assert SamplingConfig().grid_step == 10

    def test_default_angles(self):
        assert SamplingConfig().angles_deg == [0.0, 90.0, 180.0, 270.0]

    def test_grid_step_zero_raises(self):
        with pytest.raises(ValueError):
            SamplingConfig(grid_step=0)

    def test_grid_step_negative_raises(self):
        with pytest.raises(ValueError):
            SamplingConfig(grid_step=-1)

    def test_empty_angles_raises(self):
        with pytest.raises(ValueError):
            SamplingConfig(angles_deg=[])

    def test_make_rng_returns_generator(self):
        rng = SamplingConfig(seed=0).make_rng()
        assert isinstance(rng, np.random.Generator)

    def test_make_rng_reproducible(self):
        cfg = SamplingConfig(seed=7)
        r1 = cfg.make_rng().random()
        r2 = cfg.make_rng().random()
        assert r1 == pytest.approx(r2)

    def test_custom_seed(self):
        cfg = SamplingConfig(seed=42)
        assert cfg.seed == 42

    def test_custom_grid_step(self):
        cfg = SamplingConfig(grid_step=5)
        assert cfg.grid_step == 5


# ─── uniform_sample ───────────────────────────────────────────────────────────

class TestUniformSampleExtra:
    def test_returns_float(self):
        assert isinstance(uniform_sample(0.0, 1.0), float)

    def test_lo_gt_hi_raises(self):
        with pytest.raises(ValueError):
            uniform_sample(1.0, 0.0)

    def test_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            v = uniform_sample(2.0, 5.0, rng=rng)
            assert 2.0 <= v <= 5.0

    def test_lo_equals_hi(self):
        v = uniform_sample(3.0, 3.0)
        assert v == pytest.approx(3.0)

    def test_reproducible_with_rng(self):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        assert uniform_sample(0.0, 1.0, rng=rng1) == pytest.approx(
            uniform_sample(0.0, 1.0, rng=rng2)
        )


# ─── sample_angle ─────────────────────────────────────────────────────────────

class TestSampleAngleExtra:
    def test_returns_float(self):
        assert isinstance(sample_angle(), float)

    def test_from_default_angles(self):
        import math
        rng = np.random.default_rng(0)
        cfg = SamplingConfig(angles_deg=[0.0, 90.0])
        for _ in range(20):
            a = sample_angle(cfg=cfg, rng=rng)
            assert a in [0.0, math.radians(90.0)]

    def test_none_cfg_uses_defaults(self):
        a = sample_angle(cfg=None)
        assert isinstance(a, float)

    def test_custom_single_angle(self):
        import math
        cfg = SamplingConfig(angles_deg=[45.0])
        a = sample_angle(cfg=cfg)
        assert a == pytest.approx(math.radians(45.0))


# ─── sample_position ──────────────────────────────────────────────────────────

class TestSamplePositionExtra:
    def test_returns_tuple(self):
        result = sample_position(100, 80)
        assert isinstance(result, tuple) and len(result) == 2

    def test_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            x, y = sample_position(100, 80, rng=rng)
            assert 0.0 <= x < 100
            assert 0.0 <= y < 80

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            sample_position(0, 80)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            sample_position(100, 0)

    def test_reproducible(self):
        rng1 = np.random.default_rng(5)
        rng2 = np.random.default_rng(5)
        p1 = sample_position(100, 100, rng=rng1)
        p2 = sample_position(100, 100, rng=rng2)
        assert p1 == pytest.approx(p2)


# ─── sample_positions_grid ────────────────────────────────────────────────────

class TestSamplePositionsGridExtra:
    def test_returns_list(self):
        result = sample_positions_grid(100, 100)
        assert isinstance(result, list)

    def test_elements_are_tuples(self):
        for pos in sample_positions_grid(50, 50):
            assert isinstance(pos, tuple) and len(pos) == 2

    def test_positions_in_bounds(self):
        for x, y in sample_positions_grid(80, 60):
            assert 0 <= x < 80
            assert 0 <= y < 60

    def test_custom_step(self):
        cfg = SamplingConfig(grid_step=20)
        positions = sample_positions_grid(100, 100, cfg=cfg)
        assert len(positions) > 0

    def test_none_cfg(self):
        result = sample_positions_grid(50, 50, cfg=None)
        assert isinstance(result, list)


# ─── sample_permutation ───────────────────────────────────────────────────────

class TestSamplePermutationExtra:
    def test_returns_list(self):
        assert isinstance(sample_permutation(5), list)

    def test_length_n(self):
        assert len(sample_permutation(8)) == 8

    def test_contains_all_elements(self):
        perm = sample_permutation(6)
        assert sorted(perm) == list(range(6))

    def test_n_lt_1_raises(self):
        with pytest.raises(ValueError):
            sample_permutation(0)

    def test_reproducible_with_rng(self):
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(1)
        assert sample_permutation(5, rng=rng1) == sample_permutation(5, rng=rng2)

    def test_n_equals_one(self):
        perm = sample_permutation(1)
        assert perm == [0]


# ─── weighted_sample ──────────────────────────────────────────────────────────

class TestWeightedSampleExtra:
    def test_returns_int(self):
        w = np.array([1.0, 2.0, 3.0])
        assert isinstance(weighted_sample(w), int)

    def test_index_in_range(self):
        w = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(0)
        for _ in range(20):
            idx = weighted_sample(w, rng=rng)
            assert 0 <= idx < 3

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_sample(np.array([]))

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            weighted_sample(np.array([-1.0, 1.0]))

    def test_zero_sum_raises(self):
        with pytest.raises(ValueError):
            weighted_sample(np.array([0.0, 0.0]))

    def test_single_element(self):
        w = np.array([1.0])
        assert weighted_sample(w) == 0

    def test_deterministic_weight(self):
        # Only last element has weight → always returns index 2
        w = np.array([0.0, 0.0, 1.0])
        rng = np.random.default_rng(0)
        for _ in range(10):
            assert weighted_sample(w, rng=rng) == 2


# ─── acceptance_probability ───────────────────────────────────────────────────

class TestAcceptanceProbabilityExtra:
    def test_returns_float(self):
        assert isinstance(acceptance_probability(1.0, 1.0), float)

    def test_negative_delta_returns_one(self):
        assert acceptance_probability(-1.0, 1.0) == pytest.approx(1.0)

    def test_zero_delta_returns_one(self):
        assert acceptance_probability(0.0, 1.0) == pytest.approx(1.0)

    def test_positive_delta_less_than_one(self):
        p = acceptance_probability(1.0, 1.0)
        assert 0.0 < p < 1.0

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError):
            acceptance_probability(1.0, 0.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            acceptance_probability(1.0, -1.0)

    def test_higher_temperature_higher_prob(self):
        p_low = acceptance_probability(1.0, 0.1)
        p_high = acceptance_probability(1.0, 10.0)
        assert p_high > p_low

    def test_value_is_exp(self):
        import math
        delta, T = 2.0, 3.0
        expected = math.exp(-delta / T)
        assert acceptance_probability(delta, T) == pytest.approx(expected)


# ─── sample_swap_pair ─────────────────────────────────────────────────────────

class TestSampleSwapPairExtra:
    def test_returns_tuple(self):
        result = sample_swap_pair(5)
        assert isinstance(result, tuple) and len(result) == 2

    def test_indices_differ(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            i, j = sample_swap_pair(5, rng=rng)
            assert i != j

    def test_indices_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            i, j = sample_swap_pair(5, rng=rng)
            assert 0 <= i < 5 and 0 <= j < 5

    def test_n_lt_2_raises(self):
        with pytest.raises(ValueError):
            sample_swap_pair(1)

    def test_reproducible(self):
        rng1 = np.random.default_rng(3)
        rng2 = np.random.default_rng(3)
        assert sample_swap_pair(10, rng=rng1) == sample_swap_pair(10, rng=rng2)


# ─── batch_uniform_sample ─────────────────────────────────────────────────────

class TestBatchUniformSampleExtra:
    def test_returns_ndarray(self):
        out = batch_uniform_sample(0.0, 1.0, 10)
        assert isinstance(out, np.ndarray)

    def test_dtype_float64(self):
        assert batch_uniform_sample(0.0, 1.0, 5).dtype == np.float64

    def test_length_equals_size(self):
        assert len(batch_uniform_sample(0.0, 1.0, 7)) == 7

    def test_values_in_range(self):
        rng = np.random.default_rng(0)
        out = batch_uniform_sample(-5.0, 5.0, 50, rng=rng)
        assert np.all(out >= -5.0) and np.all(out <= 5.0)

    def test_lo_gt_hi_raises(self):
        with pytest.raises(ValueError):
            batch_uniform_sample(1.0, 0.0, 5)

    def test_size_lt_1_raises(self):
        with pytest.raises(ValueError):
            batch_uniform_sample(0.0, 1.0, 0)

    def test_reproducible(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        np.testing.assert_array_equal(
            batch_uniform_sample(0.0, 1.0, 10, rng=rng1),
            batch_uniform_sample(0.0, 1.0, 10, rng=rng2),
        )
