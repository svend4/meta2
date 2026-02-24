"""Extra tests for puzzle_reconstruction/algorithms/region_scorer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.region_scorer import (
    RegionScorerConfig,
    RegionScore,
    color_similarity,
    texture_similarity,
    shape_similarity,
    boundary_proximity,
    score_region_pair,
    batch_score_regions,
    rank_region_pairs,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _patch(h=20, w=20, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=20, w=20, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=20, w=20, val=(100, 150, 200)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = val
    return img


def _score(v=0.5):
    return RegionScore(score=v, color_score=v, texture_score=v,
                       shape_score=v, boundary_score=v)


# ─── RegionScorerConfig (extra) ──────────────────────────────────────────────

class TestRegionScorerConfigExtra:
    def test_large_weights_ok(self):
        cfg = RegionScorerConfig(w_color=10.0, w_texture=5.0,
                                  w_shape=3.0, w_boundary=2.0)
        assert cfg.total_weight == pytest.approx(20.0)

    def test_single_nonzero_weight(self):
        cfg = RegionScorerConfig(w_color=1.0, w_texture=0.0,
                                  w_shape=0.0, w_boundary=0.0)
        assert cfg.total_weight == pytest.approx(1.0)

    def test_large_max_distance(self):
        cfg = RegionScorerConfig(max_distance=10000.0)
        assert cfg.max_distance == pytest.approx(10000.0)

    def test_independent_instances(self):
        c1 = RegionScorerConfig(w_color=0.1)
        c2 = RegionScorerConfig(w_color=0.9)
        assert c1.w_color != c2.w_color

    def test_small_max_distance_ok(self):
        cfg = RegionScorerConfig(max_distance=0.1)
        assert cfg.max_distance == pytest.approx(0.1)

    def test_equal_weights(self):
        cfg = RegionScorerConfig(w_color=0.25, w_texture=0.25,
                                  w_shape=0.25, w_boundary=0.25)
        assert cfg.total_weight == pytest.approx(1.0)


# ─── RegionScore (extra) ─────────────────────────────────────────────────────

class TestRegionScoreExtra:
    def test_score_zero_ok(self):
        rs = _score(0.0)
        assert rs.score == pytest.approx(0.0)

    def test_score_one_ok(self):
        rs = _score(1.0)
        assert rs.score == pytest.approx(1.0)

    def test_all_channels_zero(self):
        rs = RegionScore(score=0.0, color_score=0.0, texture_score=0.0,
                         shape_score=0.0, boundary_score=0.0)
        assert rs.color_score == pytest.approx(0.0)
        assert rs.texture_score == pytest.approx(0.0)

    def test_params_default_empty(self):
        rs = _score()
        assert rs.params == {}

    def test_params_custom(self):
        rs = RegionScore(score=0.5, color_score=0.5, texture_score=0.5,
                         shape_score=0.5, boundary_score=0.5,
                         params={"method": "hist"})
        assert rs.params["method"] == "hist"

    def test_all_channels_one(self):
        rs = RegionScore(score=1.0, color_score=1.0, texture_score=1.0,
                         shape_score=1.0, boundary_score=1.0)
        assert rs.shape_score == pytest.approx(1.0)


# ─── color_similarity (extra) ────────────────────────────────────────────────

class TestColorSimilarityExtra:
    def test_different_sizes_different_values(self):
        a = _patch(10, 10, val=50)
        b = _patch(30, 30, val=200)
        result = color_similarity(a, b)
        assert 0.0 <= result <= 1.0

    def test_result_float(self):
        result = color_similarity(_patch(), _patch())
        assert isinstance(result, float)

    def test_consistent_calls(self):
        a, b = _patch(val=100), _patch(val=200)
        assert color_similarity(a, b) == pytest.approx(color_similarity(a, b))

    def test_slightly_different_high_score(self):
        a = _patch(val=127)
        b = _patch(val=129)
        assert color_similarity(a, b) > 0.95

    def test_bgr_identical_one(self):
        img = _bgr(val=(50, 100, 150))
        assert color_similarity(img, img) == pytest.approx(1.0)


# ─── texture_similarity (extra) ──────────────────────────────────────────────

class TestTextureSimilarityExtra:
    def test_result_float(self):
        result = texture_similarity(_noisy(seed=1), _noisy(seed=2))
        assert isinstance(result, float)

    def test_consistent_calls(self):
        a = _noisy(seed=5)
        b = _noisy(seed=6)
        assert texture_similarity(a, b) == pytest.approx(texture_similarity(a, b))

    def test_both_noisy_higher_than_noisy_flat(self):
        n1 = _noisy(seed=0)
        n2 = _noisy(seed=0)
        flat = _patch(val=128)
        assert texture_similarity(n1, n2) >= texture_similarity(n1, flat)

    def test_range_always_valid(self):
        for s in range(5):
            result = texture_similarity(_noisy(seed=s), _noisy(seed=s + 10))
            assert 0.0 <= result <= 1.0

    def test_same_image_gives_one(self):
        img = _noisy(seed=42)
        assert texture_similarity(img, img) == pytest.approx(1.0)


# ─── shape_similarity (extra) ────────────────────────────────────────────────

class TestShapeSimilarityExtra:
    def test_result_float(self):
        result = shape_similarity((0, 0, 10, 10), (0, 0, 10, 10))
        assert isinstance(result, float)

    def test_tall_vs_wide_same_aspect(self):
        tall = (0, 0, 5, 50)
        wide = (0, 0, 50, 5)
        # Same aspect ratio (just rotated) — function treats them as equal
        result = shape_similarity(tall, wide)
        assert 0.0 <= result <= 1.0

    def test_same_ratio_different_sizes(self):
        a = (0, 0, 10, 20)
        b = (0, 0, 20, 40)
        assert shape_similarity(a, b) == pytest.approx(1.0, abs=0.01)

    def test_consistent_calls(self):
        a = (0, 0, 10, 30)
        b = (0, 0, 20, 10)
        assert shape_similarity(a, b) == pytest.approx(shape_similarity(a, b))

    def test_all_same_position(self):
        bbox = (5, 10, 15, 15)
        assert shape_similarity(bbox, bbox) == pytest.approx(1.0)


# ─── boundary_proximity (extra) ──────────────────────────────────────────────

class TestBoundaryProximityExtra:
    def test_result_float(self):
        result = boundary_proximity((0.0, 0.0), (10.0, 0.0))
        assert isinstance(result, float)

    def test_diagonal_distance(self):
        a = (0.0, 0.0)
        b = (30.0, 40.0)
        # distance = 50
        result = boundary_proximity(a, b, max_distance=100.0)
        assert result == pytest.approx(0.5)

    def test_at_max_distance_zero(self):
        a = (0.0, 0.0)
        b = (100.0, 0.0)
        assert boundary_proximity(a, b, max_distance=100.0) == pytest.approx(0.0)

    def test_beyond_max_clipped(self):
        a = (0.0, 0.0)
        b = (500.0, 0.0)
        assert boundary_proximity(a, b, max_distance=100.0) == pytest.approx(0.0)

    def test_close_points_high(self):
        a = (50.0, 50.0)
        b = (51.0, 50.0)
        assert boundary_proximity(a, b, max_distance=100.0) > 0.98

    def test_same_point_gives_one(self):
        p = (25.5, 30.0)
        assert boundary_proximity(p, p) == pytest.approx(1.0)


# ─── score_region_pair (extra) ───────────────────────────────────────────────

class TestScoreRegionPairExtra:
    def test_all_channels_in_range(self):
        pa = _noisy(seed=1)
        pb = _noisy(seed=2)
        ba = (0, 0, 20, 20)
        bb = (10, 10, 20, 20)
        r = score_region_pair(pa, ba, pb, bb)
        assert 0.0 <= r.color_score <= 1.0
        assert 0.0 <= r.texture_score <= 1.0
        assert 0.0 <= r.shape_score <= 1.0
        assert 0.0 <= r.boundary_score <= 1.0

    def test_identical_all_high(self):
        pa = _patch(val=100)
        ba = (0, 0, 20, 20)
        r = score_region_pair(pa, ba, pa.copy(), ba)
        assert r.color_score > 0.9
        assert r.shape_score == pytest.approx(1.0)
        assert r.boundary_score == pytest.approx(1.0)

    def test_result_params_present(self):
        pa = _patch(val=128)
        ba = (0, 0, 20, 20)
        r = score_region_pair(pa, ba, pa.copy(), ba)
        assert isinstance(r.params, dict)

    def test_different_sizes_ok(self):
        pa = _patch(10, 10, val=100)
        pb = _patch(30, 30, val=200)
        ba = (0, 0, 10, 10)
        bb = (0, 0, 30, 30)
        r = score_region_pair(pa, ba, pb, bb)
        assert 0.0 <= r.score <= 1.0

    def test_far_apart_low_boundary(self):
        pa = _patch(val=128)
        pb = _patch(val=128)
        ba = (0, 0, 20, 20)
        bb = (500, 500, 20, 20)
        r = score_region_pair(pa, ba, pb, bb)
        assert r.boundary_score < 0.5


# ─── batch_score_regions (extra) ─────────────────────────────────────────────

class TestBatchScoreRegionsExtra:
    def test_single_pair(self):
        pa = _patch(val=100)
        ba = (0, 0, 20, 20)
        results = batch_score_regions([(pa, ba, pa, ba)])
        assert len(results) == 1

    def test_large_batch(self):
        pa = _patch(val=100)
        ba = (0, 0, 20, 20)
        pairs = [(pa, ba, pa, ba) for _ in range(10)]
        results = batch_score_regions(pairs)
        assert len(results) == 10

    def test_all_scores_in_range(self):
        pairs = [(_noisy(seed=i), (i * 10, 0, 20, 20),
                  _noisy(seed=i + 10), (i * 10 + 5, 0, 20, 20))
                 for i in range(5)]
        for r in batch_score_regions(pairs):
            assert 0.0 <= r.score <= 1.0

    def test_consistent_with_individual(self):
        pa = _patch(val=100)
        ba = (0, 0, 20, 20)
        batch_result = batch_score_regions([(pa, ba, pa, ba)])[0]
        individual = score_region_pair(pa, ba, pa, ba)
        assert batch_result.score == pytest.approx(individual.score, abs=1e-6)

    def test_bgr_batch(self):
        pa = _bgr()
        ba = (0, 0, 20, 20)
        results = batch_score_regions([(pa, ba, pa, ba)])
        assert len(results) == 1


# ─── rank_region_pairs (extra) ───────────────────────────────────────────────

class TestRankRegionPairsExtra:
    def test_single_score(self):
        ranked = rank_region_pairs([_score(0.5)])
        assert len(ranked) == 1

    def test_first_has_highest(self):
        scores = [_score(0.3), _score(0.9), _score(0.6)]
        ranked = rank_region_pairs(scores)
        assert ranked[0][1] == pytest.approx(0.9)

    def test_last_has_lowest(self):
        scores = [_score(0.3), _score(0.9), _score(0.6)]
        ranked = rank_region_pairs(scores)
        assert ranked[-1][1] == pytest.approx(0.3)

    def test_all_same_score_all_present(self):
        scores = [_score(0.5) for _ in range(4)]
        ranked = rank_region_pairs(scores)
        assert len(ranked) == 4

    def test_tuple_format(self):
        scores = [_score(0.7)]
        idx, val = rank_region_pairs(scores)[0]
        assert isinstance(idx, int)
        assert isinstance(val, float)

    def test_custom_indices_applied(self):
        scores = [_score(0.5), _score(0.8)]
        ranked = rank_region_pairs(scores, indices=[42, 99])
        ids = {i for i, _ in ranked}
        assert ids == {42, 99}
