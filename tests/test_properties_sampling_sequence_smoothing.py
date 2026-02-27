"""
Property-based tests for:
  1. puzzle_reconstruction.utils.sampling_utils
  2. puzzle_reconstruction.utils.sequence_utils
  3. puzzle_reconstruction.utils.smoothing_utils

Verifies mathematical invariants:

sampling_utils:
- SamplingConfig:          grid_step >= 1; empty angles_deg raises
- uniform_sample:          ∈ [lo, hi]; lo==hi → lo; lo>hi raises
- sample_angle:            result in radians of cfg.angles_deg
- sample_position:         x ∈ [0, width), y ∈ [0, height)
- sample_positions_grid:   all positions in bounds; count = rows × cols
- sample_permutation:      sorted = sorted(range(n)); length = n
- weighted_sample:         ∈ [0, len(weights)); single weight → 0;
                           zero-weight element never chosen
- acceptance_probability:  ∈ [0, 1]; delta<=0 → 1; T→∞ → 1; T<=0 raises
- sample_swap_pair:        i!=j; both in [0, n); n<2 raises
- batch_uniform_sample:    length=size; all ∈ [lo, hi]; lo==hi → all lo

sequence_utils:
- SequenceConfig:          window>=1; invalid agg raises; threshold∈[0,1]
- rank_sequence:           sorted ranks = [1..n]; min=1; max=n; same length
                           tied elements get equal ranks
- normalize_sequence:      ∈ [0, 1]; min→0; max→1; constant→all 0
- invert_sequence:         a + invert(a) = 1.0 element-wise
- sliding_scores:          same length as input; constant seq → all = const
- align_sequences:         both length target_len; both dtype float64;
                           identical sequences stay identical
- kendall_tau_distance:    ∈ [0, n*(n-1)//2]; self=0; symmetric;
                           reversed perm → max distance
- longest_increasing:      ∈ [0, n]; sorted → n; constant → 1; empty → 0;
                           single → 1; sorted desc → 1
- segment_by_threshold:    all indices within [0, n-1]; no overlap;
                           threshold=0 → single segment (all)

smoothing_utils:
- SmoothingParams:         invalid method raises; window_size odd >=3;
                           polyorder < window_size; alpha ∈ (0, 1]
- moving_average:          same length; constant → identity; mean preserved
- gaussian_smooth:         same length; constant → identity
- median_smooth:           same length; constant → identity
- exponential_smooth:      same length; first element preserved; alpha=1 → identity
- savgol_smooth:           same length; constant → identity
- smooth_contour:          shape=(N,2); constant → identity
- apply_smoothing:         same length as input for each method
- batch_smooth:            same count as input
"""
from __future__ import annotations

import math
from typing import List

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
from puzzle_reconstruction.utils.sequence_utils import (
    SequenceConfig,
    rank_sequence,
    normalize_sequence,
    invert_sequence,
    sliding_scores,
    align_sequences,
    kendall_tau_distance,
    longest_increasing,
    segment_by_threshold,
    batch_rank,
)
from puzzle_reconstruction.utils.smoothing_utils import (
    SmoothingParams,
    moving_average,
    gaussian_smooth,
    median_smooth,
    exponential_smooth,
    savgol_smooth,
    smooth_contour,
    apply_smoothing,
    batch_smooth,
)

RNG = np.random.default_rng(99)
FIXED_RNG = np.random.default_rng(7)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _seq(n: int = 20, lo: float = -5.0, hi: float = 5.0) -> np.ndarray:
    return RNG.uniform(lo, hi, size=n)


def _pos_seq(n: int = 20) -> np.ndarray:
    return RNG.uniform(0.1, 5.0, size=n)


# ═══════════════════════════════════════════════════════════════════════════════
# SamplingConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestSamplingConfig:

    def test_default_valid(self):
        cfg = SamplingConfig()
        assert cfg.grid_step >= 1
        assert len(cfg.angles_deg) > 0

    def test_custom_grid_step(self):
        cfg = SamplingConfig(grid_step=5)
        assert cfg.grid_step == 5

    def test_raises_on_zero_grid_step(self):
        with pytest.raises(ValueError):
            SamplingConfig(grid_step=0)

    def test_raises_on_empty_angles(self):
        with pytest.raises(ValueError):
            SamplingConfig(angles_deg=[])

    def test_make_rng_returns_generator(self):
        cfg = SamplingConfig(seed=42)
        rng = cfg.make_rng()
        assert isinstance(rng, np.random.Generator)

    def test_make_rng_deterministic_with_seed(self):
        cfg = SamplingConfig(seed=42)
        v1 = cfg.make_rng().random()
        v2 = cfg.make_rng().random()
        assert abs(v1 - v2) < 1e-15


# ═══════════════════════════════════════════════════════════════════════════════
# uniform_sample
# ═══════════════════════════════════════════════════════════════════════════════

class TestUniformSample:

    def test_in_range(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            v = uniform_sample(-3.0, 7.0, rng=rng)
            assert -3.0 <= v <= 7.0

    def test_lo_equals_hi(self):
        v = uniform_sample(5.0, 5.0)
        assert abs(v - 5.0) < 1e-12

    def test_lo_greater_than_hi_raises(self):
        with pytest.raises(ValueError):
            uniform_sample(10.0, 5.0)

    def test_returns_float(self):
        v = uniform_sample(0.0, 1.0)
        assert isinstance(v, float)

    def test_deterministic_with_fixed_rng(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        v1 = uniform_sample(0.0, 1.0, rng=rng1)
        v2 = uniform_sample(0.0, 1.0, rng=rng2)
        assert abs(v1 - v2) < 1e-15

    def test_zero_to_one_range(self):
        rng = np.random.default_rng(3)
        for _ in range(30):
            v = uniform_sample(0.0, 1.0, rng=rng)
            assert 0.0 <= v <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# sample_angle
# ═══════════════════════════════════════════════════════════════════════════════

class TestSampleAngle:

    def test_result_in_angles_deg(self):
        cfg = SamplingConfig(angles_deg=[0.0, 90.0, 180.0, 270.0], seed=10)
        rng = np.random.default_rng(10)
        for _ in range(40):
            angle = sample_angle(cfg, rng=rng)
            deg = math.degrees(angle)
            # angle should correspond to one of the allowed degrees
            assert any(abs(deg - d) < 1e-8 for d in cfg.angles_deg)

    def test_single_angle_always_returns_it(self):
        cfg = SamplingConfig(angles_deg=[45.0])
        rng = np.random.default_rng(0)
        for _ in range(10):
            angle = sample_angle(cfg, rng=rng)
            assert abs(angle - math.radians(45.0)) < 1e-10

    def test_default_config(self):
        angle = sample_angle(rng=np.random.default_rng(5))
        allowed_rads = {math.radians(d) for d in [0.0, 90.0, 180.0, 270.0]}
        assert any(abs(angle - r) < 1e-8 for r in allowed_rads)

    def test_returns_float(self):
        angle = sample_angle(rng=np.random.default_rng(1))
        assert isinstance(angle, float)


# ═══════════════════════════════════════════════════════════════════════════════
# sample_position
# ═══════════════════════════════════════════════════════════════════════════════

class TestSamplePosition:

    def test_x_in_range(self):
        rng = np.random.default_rng(2)
        for _ in range(30):
            x, y = sample_position(width=100, height=200, rng=rng)
            assert 0.0 <= x < 100.0
            assert 0.0 <= y < 200.0

    def test_raises_zero_width(self):
        with pytest.raises(ValueError):
            sample_position(0, 10)

    def test_raises_zero_height(self):
        with pytest.raises(ValueError):
            sample_position(10, 0)

    def test_returns_tuple_of_floats(self):
        pos = sample_position(50, 50, rng=np.random.default_rng(1))
        assert len(pos) == 2
        assert isinstance(pos[0], float)
        assert isinstance(pos[1], float)


# ═══════════════════════════════════════════════════════════════════════════════
# sample_positions_grid
# ═══════════════════════════════════════════════════════════════════════════════

class TestSamplePositionsGrid:

    def test_all_in_bounds(self):
        cfg = SamplingConfig(grid_step=10)
        positions = sample_positions_grid(100, 80, cfg=cfg)
        for x, y in positions:
            assert 0 <= x < 100
            assert 0 <= y < 80

    def test_count(self):
        cfg = SamplingConfig(grid_step=10)
        positions = sample_positions_grid(100, 80, cfg=cfg)
        expected_rows = len(range(0, 80, 10))
        expected_cols = len(range(0, 100, 10))
        assert len(positions) == expected_rows * expected_cols

    def test_step_one(self):
        cfg = SamplingConfig(grid_step=1)
        positions = sample_positions_grid(3, 3, cfg=cfg)
        assert len(positions) == 9

    def test_raises_zero_width(self):
        with pytest.raises(ValueError):
            sample_positions_grid(0, 10)

    def test_raises_zero_height(self):
        with pytest.raises(ValueError):
            sample_positions_grid(10, 0)

    def test_first_position_is_origin(self):
        cfg = SamplingConfig(grid_step=5)
        positions = sample_positions_grid(20, 20, cfg=cfg)
        assert positions[0] == (0, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# sample_permutation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSamplePermutation:

    def test_length_equals_n(self):
        rng = np.random.default_rng(5)
        for n in [1, 5, 10, 20]:
            perm = sample_permutation(n, rng=rng)
            assert len(perm) == n

    def test_sorted_equals_range(self):
        rng = np.random.default_rng(6)
        for n in [3, 7, 15]:
            perm = sample_permutation(n, rng=rng)
            assert sorted(perm) == list(range(n))

    def test_raises_n_zero(self):
        with pytest.raises(ValueError):
            sample_permutation(0)

    def test_single_element(self):
        perm = sample_permutation(1, rng=np.random.default_rng(1))
        assert perm == [0]

    def test_different_orders(self):
        rng = np.random.default_rng(99)
        results = {tuple(sample_permutation(5, rng=rng)) for _ in range(100)}
        # With 5 elements, we should see multiple different orderings
        assert len(results) > 1


# ═══════════════════════════════════════════════════════════════════════════════
# weighted_sample
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeightedSample:

    def test_result_in_range(self):
        rng = np.random.default_rng(7)
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        for _ in range(30):
            idx = weighted_sample(weights, rng=rng)
            assert 0 <= idx < len(weights)

    def test_single_weight_returns_zero(self):
        rng = np.random.default_rng(8)
        weights = np.array([5.0])
        for _ in range(10):
            assert weighted_sample(weights, rng=rng) == 0

    def test_zero_weight_never_selected(self):
        rng = np.random.default_rng(9)
        weights = np.array([0.0, 1.0, 0.0])
        for _ in range(50):
            idx = weighted_sample(weights, rng=rng)
            assert idx == 1

    def test_raises_empty_weights(self):
        with pytest.raises(ValueError):
            weighted_sample(np.array([]))

    def test_raises_negative_weights(self):
        with pytest.raises(ValueError):
            weighted_sample(np.array([1.0, -0.5, 2.0]))

    def test_raises_zero_sum(self):
        with pytest.raises(ValueError):
            weighted_sample(np.array([0.0, 0.0, 0.0]))


# ═══════════════════════════════════════════════════════════════════════════════
# acceptance_probability
# ═══════════════════════════════════════════════════════════════════════════════

class TestAcceptanceProbability:

    def test_range(self):
        for delta in [-5.0, 0.0, 1.0, 10.0, 100.0]:
            for T in [0.01, 1.0, 100.0]:
                p = acceptance_probability(delta, T)
                assert 0.0 <= p <= 1.0

    def test_negative_delta_is_one(self):
        for delta in [-10.0, -1.0, -0.001]:
            p = acceptance_probability(delta, temperature=1.0)
            assert p == 1.0

    def test_zero_delta_is_one(self):
        p = acceptance_probability(0.0, temperature=1.0)
        assert p == 1.0

    def test_high_temperature_approaches_one(self):
        # exp(-delta/T) → 1 as T → ∞
        p = acceptance_probability(1.0, temperature=1e8)
        assert p > 0.99

    def test_low_temperature_approaches_zero(self):
        # exp(-delta/T) → 0 as T → 0+ for delta > 0
        p = acceptance_probability(100.0, temperature=0.001)
        assert p < 1e-10

    def test_raises_zero_temperature(self):
        with pytest.raises(ValueError):
            acceptance_probability(1.0, temperature=0.0)

    def test_raises_negative_temperature(self):
        with pytest.raises(ValueError):
            acceptance_probability(1.0, temperature=-1.0)

    def test_formula(self):
        delta, T = 2.0, 3.0
        expected = math.exp(-2.0 / 3.0)
        p = acceptance_probability(delta, T)
        assert abs(p - expected) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# sample_swap_pair
# ═══════════════════════════════════════════════════════════════════════════════

class TestSampleSwapPair:

    def test_i_not_equal_j(self):
        rng = np.random.default_rng(11)
        for _ in range(50):
            i, j = sample_swap_pair(5, rng=rng)
            assert i != j

    def test_both_in_range(self):
        rng = np.random.default_rng(12)
        for _ in range(50):
            i, j = sample_swap_pair(6, rng=rng)
            assert 0 <= i < 6
            assert 0 <= j < 6

    def test_raises_n_less_than_2(self):
        with pytest.raises(ValueError):
            sample_swap_pair(1)

    def test_n_equals_2(self):
        rng = np.random.default_rng(13)
        for _ in range(10):
            i, j = sample_swap_pair(2, rng=rng)
            assert {i, j} == {0, 1}


# ═══════════════════════════════════════════════════════════════════════════════
# batch_uniform_sample
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchUniformSample:

    def test_length(self):
        arr = batch_uniform_sample(0.0, 1.0, size=30, rng=np.random.default_rng(1))
        assert len(arr) == 30

    def test_all_in_range(self):
        arr = batch_uniform_sample(-2.0, 3.0, size=50, rng=np.random.default_rng(2))
        assert float(arr.min()) >= -2.0 - 1e-12
        assert float(arr.max()) <= 3.0 + 1e-12

    def test_lo_equals_hi_all_same(self):
        arr = batch_uniform_sample(5.0, 5.0, size=20)
        assert np.allclose(arr, 5.0)

    def test_raises_lo_gt_hi(self):
        with pytest.raises(ValueError):
            batch_uniform_sample(10.0, 5.0, size=10)

    def test_raises_size_zero(self):
        with pytest.raises(ValueError):
            batch_uniform_sample(0.0, 1.0, size=0)

    def test_dtype_float64(self):
        arr = batch_uniform_sample(0.0, 1.0, size=5)
        assert arr.dtype == np.float64


# ═══════════════════════════════════════════════════════════════════════════════
# SequenceConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestSequenceConfig:

    def test_default_valid(self):
        cfg = SequenceConfig()
        assert cfg.window >= 1
        assert cfg.agg in ("mean", "max", "min", "sum")
        assert 0.0 <= cfg.threshold <= 1.0

    def test_raises_zero_window(self):
        with pytest.raises(ValueError):
            SequenceConfig(window=0)

    def test_raises_invalid_agg(self):
        with pytest.raises(ValueError):
            SequenceConfig(agg="median")  # type: ignore

    def test_raises_threshold_out_of_range(self):
        with pytest.raises(ValueError):
            SequenceConfig(threshold=1.5)
        with pytest.raises(ValueError):
            SequenceConfig(threshold=-0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# rank_sequence
# ═══════════════════════════════════════════════════════════════════════════════

class TestRankSequence:

    def test_same_length(self):
        seq = _seq(15)
        ranks = rank_sequence(seq)
        assert len(ranks) == len(seq)

    def test_min_rank_is_one(self):
        seq = _seq(20)
        ranks = rank_sequence(seq)
        assert abs(float(ranks.min()) - 1.0) < 1e-10

    def test_max_rank_is_n(self):
        seq = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        ranks = rank_sequence(seq)
        assert float(ranks.max()) == pytest.approx(5.0)

    def test_sorted_input_gives_ascending_ranks(self):
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ranks = rank_sequence(seq)
        assert np.allclose(ranks, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_tied_elements_get_equal_ranks(self):
        seq = np.array([1.0, 2.0, 2.0, 3.0])
        ranks = rank_sequence(seq)
        # Both 2.0 values should get the same rank (2.5 = (2+3)/2)
        assert abs(ranks[1] - ranks[2]) < 1e-10

    def test_sum_of_ranks_equals_n_times_n_plus_1_over_2(self):
        seq = _seq(10)
        ranks = rank_sequence(seq)
        n = len(seq)
        assert abs(float(ranks.sum()) - n * (n + 1) / 2.0) < 1e-9

    def test_raises_empty(self):
        with pytest.raises(ValueError):
            rank_sequence(np.array([]))

    def test_raises_2d(self):
        with pytest.raises(ValueError):
            rank_sequence(np.array([[1.0, 2.0]]))

    def test_single_element(self):
        ranks = rank_sequence(np.array([42.0]))
        assert float(ranks[0]) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# normalize_sequence
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeSequence:

    def test_output_in_zero_one(self):
        seq = _seq(20)
        out = normalize_sequence(seq)
        assert float(out.min()) >= 0.0 - 1e-12
        assert float(out.max()) <= 1.0 + 1e-12

    def test_min_maps_to_zero(self):
        seq = np.array([1.0, 3.0, 5.0, 2.0])
        out = normalize_sequence(seq)
        assert abs(float(out[np.argmin(seq)]) - 0.0) < 1e-10

    def test_max_maps_to_one(self):
        seq = np.array([1.0, 3.0, 5.0, 2.0])
        out = normalize_sequence(seq)
        assert abs(float(out[np.argmax(seq)]) - 1.0) < 1e-10

    def test_constant_seq_maps_to_zeros(self):
        seq = np.full(10, 7.0)
        out = normalize_sequence(seq)
        assert np.allclose(out, 0.0)

    def test_same_length(self):
        seq = _seq(15)
        out = normalize_sequence(seq)
        assert len(out) == len(seq)

    def test_raises_empty(self):
        with pytest.raises(ValueError):
            normalize_sequence(np.array([]))

    def test_single_element(self):
        out = normalize_sequence(np.array([5.0]))
        assert float(out[0]) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# invert_sequence
# ═══════════════════════════════════════════════════════════════════════════════

class TestInvertSequence:

    def test_sum_with_original_is_one(self):
        seq = normalize_sequence(_seq(15))
        inv = invert_sequence(seq)
        assert np.allclose(seq + inv, 1.0, atol=1e-12)

    def test_same_length(self):
        seq = _seq(10)
        inv = invert_sequence(seq)
        assert len(inv) == len(seq)

    def test_double_invert_restores_original(self):
        seq = normalize_sequence(_seq(10))
        inv = invert_sequence(invert_sequence(seq))
        assert np.allclose(inv, seq, atol=1e-12)

    def test_zero_maps_to_one(self):
        seq = np.array([0.0, 0.5, 1.0])
        inv = invert_sequence(seq)
        assert abs(inv[0] - 1.0) < 1e-12
        assert abs(inv[2] - 0.0) < 1e-12

    def test_raises_empty(self):
        with pytest.raises(ValueError):
            invert_sequence(np.array([]))

    def test_raises_2d(self):
        with pytest.raises(ValueError):
            invert_sequence(np.array([[1.0, 2.0]]))


# ═══════════════════════════════════════════════════════════════════════════════
# sliding_scores
# ═══════════════════════════════════════════════════════════════════════════════

class TestSlidingScores:

    def test_same_length(self):
        seq = _seq(20)
        for agg in ("mean", "max", "min", "sum"):
            out = sliding_scores(seq, SequenceConfig(window=3, agg=agg))
            assert len(out) == len(seq)

    def test_constant_seq_preserved(self):
        seq = np.full(15, 3.5)
        out = sliding_scores(seq, SequenceConfig(window=5, agg="mean"))
        assert np.allclose(out, 3.5, atol=1e-10)

    def test_window_one_is_identity(self):
        seq = _seq(10)
        out = sliding_scores(seq, SequenceConfig(window=1, agg="mean"))
        assert np.allclose(out, seq, atol=1e-10)

    def test_max_agg_ge_mean_agg(self):
        seq = _pos_seq(15)
        out_max = sliding_scores(seq, SequenceConfig(window=3, agg="max"))
        out_mean = sliding_scores(seq, SequenceConfig(window=3, agg="mean"))
        assert np.all(out_max >= out_mean - 1e-12)

    def test_min_agg_le_mean_agg(self):
        seq = _pos_seq(15)
        out_min = sliding_scores(seq, SequenceConfig(window=3, agg="min"))
        out_mean = sliding_scores(seq, SequenceConfig(window=3, agg="mean"))
        assert np.all(out_min <= out_mean + 1e-12)

    def test_raises_empty(self):
        with pytest.raises(ValueError):
            sliding_scores(np.array([]))

    def test_raises_2d(self):
        with pytest.raises(ValueError):
            sliding_scores(np.array([[1.0, 2.0]]))


# ═══════════════════════════════════════════════════════════════════════════════
# align_sequences
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlignSequences:

    def test_both_have_target_length(self):
        a = _seq(7)
        b = _seq(12)
        a2, b2 = align_sequences(a, b, target_len=20)
        assert len(a2) == 20
        assert len(b2) == 20

    def test_default_target_is_max(self):
        a = _seq(7)
        b = _seq(12)
        a2, b2 = align_sequences(a, b)
        assert len(a2) == 12
        assert len(b2) == 12

    def test_both_dtype_float64(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        a2, b2 = align_sequences(a, b)
        assert a2.dtype == np.float64
        assert b2.dtype == np.float64

    def test_same_length_no_change(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        a2, b2 = align_sequences(a, b)
        assert np.allclose(a2, a)
        assert np.allclose(b2, b)

    def test_constant_seq_stays_constant(self):
        a = np.full(5, 3.0)
        b = np.ones(8) * 2.0
        a2, b2 = align_sequences(a, b, target_len=10)
        assert np.allclose(a2, 3.0, atol=1e-10)
        assert np.allclose(b2, 2.0, atol=1e-10)

    def test_target_len_one(self):
        a = _seq(5)
        b = _seq(7)
        a2, b2 = align_sequences(a, b, target_len=1)
        assert len(a2) == 1 and len(b2) == 1

    def test_raises_if_target_len_zero(self):
        with pytest.raises(ValueError):
            align_sequences(_seq(5), _seq(5), target_len=0)


# ═══════════════════════════════════════════════════════════════════════════════
# kendall_tau_distance
# ═══════════════════════════════════════════════════════════════════════════════

class TestKendallTauDistance:

    def test_self_distance_is_zero(self):
        perm = np.array([2, 0, 3, 1, 4])
        assert kendall_tau_distance(perm, perm) == 0

    def test_symmetric(self):
        a = np.array([0, 2, 1, 3])
        b = np.array([3, 1, 2, 0])
        assert kendall_tau_distance(a, b) == kendall_tau_distance(b, a)

    def test_range(self):
        n = 5
        perm = np.arange(n)
        rev = perm[::-1].copy()
        dist = kendall_tau_distance(perm, rev)
        max_dist = n * (n - 1) // 2
        assert 0 <= dist <= max_dist

    def test_reversed_perm_is_max_distance(self):
        n = 5
        perm = np.arange(n)
        rev = perm[::-1].copy()
        dist = kendall_tau_distance(perm, rev)
        assert dist == n * (n - 1) // 2

    def test_known_one_swap(self):
        a = np.array([0, 1, 2, 3])
        b = np.array([0, 2, 1, 3])  # swap 1 and 2 → 1 inversion
        assert kendall_tau_distance(a, b) == 1

    def test_raises_different_lengths(self):
        with pytest.raises(ValueError):
            kendall_tau_distance(np.array([0, 1, 2]), np.array([0, 1]))

    def test_raises_2d_input(self):
        with pytest.raises(ValueError):
            kendall_tau_distance(np.array([[0, 1]]), np.array([[1, 0]]))

    def test_single_element(self):
        assert kendall_tau_distance(np.array([0]), np.array([0])) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# longest_increasing
# ═══════════════════════════════════════════════════════════════════════════════

class TestLongestIncreasing:

    def test_empty_returns_zero(self):
        assert longest_increasing(np.array([])) == 0

    def test_single_returns_one(self):
        assert longest_increasing(np.array([5.0])) == 1

    def test_sorted_returns_n(self):
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert longest_increasing(seq) == 5

    def test_sorted_desc_returns_one(self):
        seq = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert longest_increasing(seq) == 1

    def test_constant_returns_one(self):
        seq = np.full(10, 3.0)
        assert longest_increasing(seq) == 1

    def test_in_range(self):
        seq = _seq(20)
        n = longest_increasing(seq)
        assert 0 <= n <= 20

    def test_known_sequence(self):
        # [3, 1, 4, 1, 5, 9, 2, 6] → LIS = [1, 4, 5, 9] or [1, 5, 9] etc. = length 4
        seq = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        assert longest_increasing(seq) == 4

    def test_raises_2d(self):
        with pytest.raises(ValueError):
            longest_increasing(np.array([[1.0, 2.0]]))


# ═══════════════════════════════════════════════════════════════════════════════
# segment_by_threshold
# ═══════════════════════════════════════════════════════════════════════════════

class TestSegmentByThreshold:

    def test_all_indices_within_range(self):
        seq = normalize_sequence(_seq(20))
        cfg = SequenceConfig(threshold=0.5)
        segs = segment_by_threshold(seq, cfg)
        for start, end in segs:
            assert 0 <= start <= end < len(seq)

    def test_no_overlap(self):
        seq = normalize_sequence(_seq(20))
        segs = segment_by_threshold(seq)
        for i in range(len(segs) - 1):
            assert segs[i][1] < segs[i + 1][0]

    def test_threshold_zero_covers_all(self):
        seq = np.ones(10) * 0.5  # all >= 0.0
        cfg = SequenceConfig(threshold=0.0)
        segs = segment_by_threshold(seq, cfg)
        assert len(segs) == 1
        assert segs[0] == (0, 9)

    def test_threshold_one_only_ones(self):
        seq = np.array([0.5, 1.0, 0.5, 1.0, 1.0])
        cfg = SequenceConfig(threshold=1.0)
        segs = segment_by_threshold(seq, cfg)
        # Only indices 1, 3, 4 qualify
        all_covered = set()
        for s, e in segs:
            all_covered.update(range(s, e + 1))
        assert 0 not in all_covered
        assert 2 not in all_covered

    def test_empty_seq_returns_empty(self):
        segs = segment_by_threshold(np.array([]))
        assert segs == []

    def test_raises_2d(self):
        with pytest.raises(ValueError):
            segment_by_threshold(np.array([[1.0, 0.5]]))

    def test_all_below_threshold_empty(self):
        seq = np.zeros(10)
        cfg = SequenceConfig(threshold=0.5)
        segs = segment_by_threshold(seq, cfg)
        assert segs == []


# ═══════════════════════════════════════════════════════════════════════════════
# batch_rank
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchRank:

    def test_same_count_as_input(self):
        seqs = [_seq(5), _seq(8), _seq(3)]
        results = batch_rank(seqs)
        assert len(results) == 3

    def test_each_has_same_length(self):
        seqs = [_seq(5), _seq(8)]
        results = batch_rank(seqs)
        assert len(results[0]) == 5
        assert len(results[1]) == 8

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            batch_rank([])


# ═══════════════════════════════════════════════════════════════════════════════
# SmoothingParams
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmoothingParams:

    def test_default_valid(self):
        p = SmoothingParams()
        assert p.method in {"moving_average", "gaussian", "median", "savgol", "exponential"}
        assert p.window_size >= 3
        assert p.window_size % 2 == 1
        assert p.sigma > 0

    def test_raises_invalid_method(self):
        with pytest.raises(ValueError):
            SmoothingParams(method="lowess")

    def test_raises_window_too_small(self):
        with pytest.raises(ValueError):
            SmoothingParams(window_size=2)

    def test_raises_even_window(self):
        with pytest.raises(ValueError):
            SmoothingParams(window_size=4)

    def test_raises_polyorder_ge_window(self):
        with pytest.raises(ValueError):
            SmoothingParams(method="savgol", window_size=5, polyorder=5)

    def test_raises_alpha_zero(self):
        with pytest.raises(ValueError):
            SmoothingParams(method="exponential", alpha=0.0)

    def test_raises_alpha_gt_one(self):
        with pytest.raises(ValueError):
            SmoothingParams(method="exponential", alpha=1.1)


# ═══════════════════════════════════════════════════════════════════════════════
# moving_average
# ═══════════════════════════════════════════════════════════════════════════════

class TestMovingAverage:

    def test_same_length(self):
        sig = _seq(20)
        out = moving_average(sig, window_size=5)
        assert len(out) == len(sig)

    def test_constant_signal_preserved(self):
        sig = np.full(20, 3.7)
        out = moving_average(sig, window_size=5)
        assert np.allclose(out, 3.7, atol=1e-10)

    def test_dtype_float64(self):
        out = moving_average(_seq(10), window_size=3)
        assert out.dtype == np.float64

    def test_raises_even_window(self):
        with pytest.raises(ValueError):
            moving_average(_seq(10), window_size=4)

    def test_raises_small_window(self):
        with pytest.raises(ValueError):
            moving_average(_seq(10), window_size=2)

    def test_raises_2d(self):
        with pytest.raises(ValueError):
            moving_average(np.ones((5, 5)), window_size=3)

    def test_empty_signal_returns_empty(self):
        out = moving_average(np.array([]), window_size=3)
        assert len(out) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# gaussian_smooth
# ═══════════════════════════════════════════════════════════════════════════════

class TestGaussianSmooth:

    def test_same_length(self):
        sig = _seq(20)
        out = gaussian_smooth(sig, sigma=1.5)
        assert len(out) == len(sig)

    def test_constant_signal_preserved(self):
        sig = np.full(30, 4.2)
        out = gaussian_smooth(sig, sigma=2.0)
        assert np.allclose(out, 4.2, atol=1e-9)

    def test_dtype_float64(self):
        out = gaussian_smooth(_seq(10), sigma=1.0)
        assert out.dtype == np.float64

    def test_raises_zero_sigma(self):
        with pytest.raises(ValueError):
            gaussian_smooth(_seq(10), sigma=0.0)

    def test_raises_negative_sigma(self):
        with pytest.raises(ValueError):
            gaussian_smooth(_seq(10), sigma=-1.0)

    def test_raises_2d(self):
        with pytest.raises(ValueError):
            gaussian_smooth(np.ones((5, 5)), sigma=1.0)

    def test_empty_signal_returns_empty(self):
        out = gaussian_smooth(np.array([]))
        assert len(out) == 0

    def test_large_sigma_flattens_signal(self):
        sig = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
        out = gaussian_smooth(sig, sigma=5.0)
        # After heavy smoothing, max should be reduced
        assert float(out.max()) < 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# median_smooth
# ═══════════════════════════════════════════════════════════════════════════════

class TestMedianSmooth:

    def test_same_length(self):
        sig = _seq(20)
        out = median_smooth(sig, window_size=5)
        assert len(out) == len(sig)

    def test_constant_signal_preserved(self):
        sig = np.full(15, 2.3)
        out = median_smooth(sig, window_size=5)
        assert np.allclose(out, 2.3, atol=1e-10)

    def test_single_outlier_removed(self):
        sig = np.array([1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0])
        out = median_smooth(sig, window_size=3)
        # The outlier at index 3 should be reduced
        assert float(out[3]) < 100.0

    def test_raises_even_window(self):
        with pytest.raises(ValueError):
            median_smooth(_seq(10), window_size=4)

    def test_raises_small_window(self):
        with pytest.raises(ValueError):
            median_smooth(_seq(10), window_size=2)

    def test_empty_returns_empty(self):
        out = median_smooth(np.array([]), window_size=3)
        assert len(out) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# exponential_smooth
# ═══════════════════════════════════════════════════════════════════════════════

class TestExponentialSmooth:

    def test_same_length(self):
        sig = _seq(20)
        out = exponential_smooth(sig, alpha=0.3)
        assert len(out) == len(sig)

    def test_first_element_preserved(self):
        sig = _seq(15)
        out = exponential_smooth(sig, alpha=0.5)
        assert abs(float(out[0]) - float(sig[0])) < 1e-12

    def test_alpha_one_is_identity(self):
        sig = _seq(15)
        out = exponential_smooth(sig, alpha=1.0)
        assert np.allclose(out, sig, atol=1e-12)

    def test_constant_signal_preserved(self):
        sig = np.full(10, 5.5)
        out = exponential_smooth(sig, alpha=0.3)
        assert np.allclose(out, 5.5, atol=1e-10)

    def test_raises_zero_alpha(self):
        with pytest.raises(ValueError):
            exponential_smooth(_seq(10), alpha=0.0)

    def test_raises_alpha_gt_one(self):
        with pytest.raises(ValueError):
            exponential_smooth(_seq(10), alpha=1.1)

    def test_empty_returns_empty(self):
        out = exponential_smooth(np.array([]), alpha=0.5)
        assert len(out) == 0

    def test_dtype_float64(self):
        out = exponential_smooth(_seq(10))
        assert out.dtype == np.float64


# ═══════════════════════════════════════════════════════════════════════════════
# savgol_smooth
# ═══════════════════════════════════════════════════════════════════════════════

class TestSavgolSmooth:

    def test_same_length(self):
        sig = _seq(20)
        out = savgol_smooth(sig, window_size=5, polyorder=2)
        assert len(out) == len(sig)

    def test_constant_signal_preserved(self):
        sig = np.full(20, 3.3)
        out = savgol_smooth(sig, window_size=5, polyorder=2)
        assert np.allclose(out, 3.3, atol=1e-8)

    def test_raises_even_window(self):
        with pytest.raises(ValueError):
            savgol_smooth(_seq(10), window_size=4)

    def test_raises_polyorder_ge_window(self):
        with pytest.raises(ValueError):
            savgol_smooth(_seq(10), window_size=5, polyorder=5)

    def test_dtype_float64(self):
        out = savgol_smooth(_seq(20))
        assert out.dtype == np.float64

    def test_empty_returns_empty(self):
        out = savgol_smooth(np.array([]), window_size=5, polyorder=2)
        assert len(out) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# smooth_contour
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmoothContour:

    def _circle_contour(self, n: int = 20) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)

    def test_output_shape(self):
        c = self._circle_contour(30)
        out = smooth_contour(c, sigma=1.0)
        assert out.shape == (30, 2)

    def test_constant_contour_preserved(self):
        c = np.full((10, 2), 5.0, dtype=np.float32)
        out = smooth_contour(c, sigma=1.0)
        assert np.allclose(out, 5.0, atol=1e-6)

    def test_dtype_float32(self):
        c = self._circle_contour(20)
        out = smooth_contour(c, sigma=1.0)
        assert out.dtype == np.float32

    def test_raises_wrong_shape(self):
        with pytest.raises(ValueError):
            smooth_contour(np.ones((10,)), sigma=1.0)
        with pytest.raises(ValueError):
            smooth_contour(np.ones((10, 3)), sigma=1.0)

    def test_raises_zero_sigma(self):
        c = self._circle_contour(10)
        with pytest.raises(ValueError):
            smooth_contour(c, sigma=0.0)

    def test_raises_empty_contour(self):
        with pytest.raises(ValueError):
            smooth_contour(np.zeros((0, 2)), sigma=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# apply_smoothing
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplySmoothing:

    @pytest.mark.parametrize("method", [
        "moving_average", "gaussian", "median", "savgol", "exponential"
    ])
    def test_same_length_for_each_method(self, method: str):
        params = SmoothingParams(method=method, window_size=5, polyorder=2)
        sig = _seq(20)
        out = apply_smoothing(sig, params)
        assert len(out) == len(sig)

    @pytest.mark.parametrize("method", [
        "moving_average", "gaussian", "median", "savgol", "exponential"
    ])
    def test_constant_signal_preserved(self, method: str):
        params = SmoothingParams(method=method, window_size=5, polyorder=2)
        sig = np.full(20, 4.4)
        out = apply_smoothing(sig, params)
        assert np.allclose(out, 4.4, atol=1e-8)

    @pytest.mark.parametrize("method", [
        "moving_average", "gaussian", "median", "savgol", "exponential"
    ])
    def test_dtype_float64(self, method: str):
        params = SmoothingParams(method=method, window_size=5, polyorder=2)
        out = apply_smoothing(_seq(20), params)
        assert out.dtype == np.float64


# ═══════════════════════════════════════════════════════════════════════════════
# batch_smooth
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchSmooth:

    def test_same_count_as_input(self):
        signals = [_seq(10), _seq(15), _seq(8)]
        params = SmoothingParams()
        results = batch_smooth(signals, params)
        assert len(results) == 3

    def test_each_has_same_length_as_input(self):
        signals = [_seq(10), _seq(15)]
        params = SmoothingParams()
        results = batch_smooth(signals, params)
        assert len(results[0]) == 10
        assert len(results[1]) == 15

    @pytest.mark.parametrize("method", [
        "moving_average", "gaussian", "median", "savgol", "exponential"
    ])
    def test_all_methods_work(self, method: str):
        params = SmoothingParams(method=method, window_size=5, polyorder=2)
        signals = [_seq(12) for _ in range(4)]
        results = batch_smooth(signals, params)
        assert len(results) == 4
        for s, r in zip(signals, results):
            assert len(r) == len(s)
