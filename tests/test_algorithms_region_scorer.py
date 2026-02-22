"""Tests for puzzle_reconstruction/algorithms/region_scorer.py"""
import pytest
import numpy as np

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


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_patch(h=20, w=20, value=128, dtype=np.uint8):
    return np.full((h, w), value, dtype=dtype)


def make_noisy_patch(h=20, w=20, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, 255, (h, w))).astype(np.uint8)


def make_bgr_patch(h=20, w=20, value=(100, 150, 200)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = value
    return img


# ─── RegionScorerConfig ───────────────────────────────────────────────────────

class TestRegionScorerConfig:
    def test_defaults(self):
        cfg = RegionScorerConfig()
        assert cfg.w_color == pytest.approx(0.35)
        assert cfg.w_texture == pytest.approx(0.25)
        assert cfg.w_shape == pytest.approx(0.20)
        assert cfg.w_boundary == pytest.approx(0.20)
        assert cfg.max_distance == pytest.approx(100.0)

    def test_total_weight(self):
        cfg = RegionScorerConfig(w_color=0.4, w_texture=0.3,
                                  w_shape=0.2, w_boundary=0.1)
        assert cfg.total_weight == pytest.approx(1.0)

    def test_negative_w_color_raises(self):
        with pytest.raises(ValueError):
            RegionScorerConfig(w_color=-0.1)

    def test_negative_w_texture_raises(self):
        with pytest.raises(ValueError):
            RegionScorerConfig(w_texture=-0.5)

    def test_negative_w_shape_raises(self):
        with pytest.raises(ValueError):
            RegionScorerConfig(w_shape=-1.0)

    def test_negative_w_boundary_raises(self):
        with pytest.raises(ValueError):
            RegionScorerConfig(w_boundary=-0.01)

    def test_max_distance_zero_raises(self):
        with pytest.raises(ValueError):
            RegionScorerConfig(max_distance=0.0)

    def test_max_distance_negative_raises(self):
        with pytest.raises(ValueError):
            RegionScorerConfig(max_distance=-10.0)

    def test_custom_max_distance(self):
        cfg = RegionScorerConfig(max_distance=500.0)
        assert cfg.max_distance == pytest.approx(500.0)

    def test_zero_weights_ok(self):
        cfg = RegionScorerConfig(w_color=0.0, w_texture=0.0,
                                  w_shape=0.0, w_boundary=0.0)
        assert cfg.total_weight == pytest.approx(0.0)


# ─── RegionScore ──────────────────────────────────────────────────────────────

class TestRegionScore:
    def test_basic(self):
        rs = RegionScore(score=0.7, color_score=0.8, texture_score=0.6,
                         shape_score=0.9, boundary_score=0.5)
        assert rs.score == pytest.approx(0.7)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RegionScore(score=1.5, color_score=0.5, texture_score=0.5,
                        shape_score=0.5, boundary_score=0.5)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            RegionScore(score=-0.1, color_score=0.5, texture_score=0.5,
                        shape_score=0.5, boundary_score=0.5)

    def test_color_score_invalid_raises(self):
        with pytest.raises(ValueError):
            RegionScore(score=0.5, color_score=-0.1, texture_score=0.5,
                        shape_score=0.5, boundary_score=0.5)

    def test_texture_score_invalid_raises(self):
        with pytest.raises(ValueError):
            RegionScore(score=0.5, color_score=0.5, texture_score=2.0,
                        shape_score=0.5, boundary_score=0.5)

    def test_shape_score_invalid_raises(self):
        with pytest.raises(ValueError):
            RegionScore(score=0.5, color_score=0.5, texture_score=0.5,
                        shape_score=1.1, boundary_score=0.5)

    def test_boundary_score_invalid_raises(self):
        with pytest.raises(ValueError):
            RegionScore(score=0.5, color_score=0.5, texture_score=0.5,
                        shape_score=0.5, boundary_score=-0.5)

    def test_boundary_values_ok(self):
        rs = RegionScore(score=0.0, color_score=1.0, texture_score=0.0,
                         shape_score=1.0, boundary_score=0.0)
        assert rs.score == 0.0

    def test_params_stored(self):
        rs = RegionScore(score=0.5, color_score=0.5, texture_score=0.5,
                         shape_score=0.5, boundary_score=0.5,
                         params={"tag": "test"})
        assert rs.params["tag"] == "test"


# ─── color_similarity ─────────────────────────────────────────────────────────

class TestColorSimilarity:
    def test_identical_patches(self):
        patch = make_patch(value=128)
        assert color_similarity(patch, patch) == pytest.approx(1.0)

    def test_zero_vs_255(self):
        a = make_patch(value=0)
        b = make_patch(value=255)
        assert color_similarity(a, b) == pytest.approx(0.0, abs=0.01)

    def test_range(self):
        a = make_noisy_patch(seed=1)
        b = make_noisy_patch(seed=2)
        result = color_similarity(a, b)
        assert 0.0 <= result <= 1.0

    def test_empty_a_returns_one(self):
        a = np.zeros((0, 10), dtype=np.uint8)
        b = make_patch(value=100)
        assert color_similarity(a, b) == 1.0

    def test_empty_b_returns_one(self):
        a = make_patch(value=100)
        b = np.zeros((0, 10), dtype=np.uint8)
        assert color_similarity(a, b) == 1.0

    def test_close_values_high_score(self):
        a = make_patch(value=100)
        b = make_patch(value=102)
        assert color_similarity(a, b) > 0.99

    def test_bgr_patches(self):
        a = make_bgr_patch(value=(100, 100, 100))
        b = make_bgr_patch(value=(100, 100, 100))
        result = color_similarity(a, b)
        assert result == pytest.approx(1.0)

    def test_symmetric(self):
        a = make_patch(value=80)
        b = make_patch(value=200)
        assert color_similarity(a, b) == pytest.approx(color_similarity(b, a))

    def test_non_negative(self):
        a = make_patch(value=0)
        b = make_patch(value=255)
        assert color_similarity(a, b) >= 0.0


# ─── texture_similarity ───────────────────────────────────────────────────────

class TestTextureSimilarity:
    def test_identical_patches(self):
        patch = make_noisy_patch(seed=5)
        assert texture_similarity(patch, patch) == pytest.approx(1.0)

    def test_both_constant_returns_one(self):
        a = make_patch(value=100)
        b = make_patch(value=200)
        # both have std=0 → returns 1.0
        assert texture_similarity(a, b) == pytest.approx(1.0)

    def test_range(self):
        a = make_noisy_patch(seed=1)
        b = make_patch(value=128)
        result = texture_similarity(a, b)
        assert 0.0 <= result <= 1.0

    def test_symmetric(self):
        a = make_noisy_patch(seed=3)
        b = make_noisy_patch(seed=7)
        assert texture_similarity(a, b) == pytest.approx(texture_similarity(b, a))

    def test_bgr_patches(self):
        a = make_bgr_patch(value=(50, 100, 150))
        b = make_bgr_patch(value=(200, 210, 220))
        result = texture_similarity(a, b)
        assert 0.0 <= result <= 1.0

    def test_noisy_vs_flat_lower_than_two_noisy(self):
        """Noisy vs flat should score lower than two identical noisy patches."""
        noisy = make_noisy_patch(seed=0)
        flat = make_patch(value=128)
        sim_nn = texture_similarity(noisy, noisy)
        sim_nf = texture_similarity(noisy, flat)
        assert sim_nn >= sim_nf

    def test_non_negative(self):
        a = make_noisy_patch(seed=0)
        b = make_patch(value=255)
        assert texture_similarity(a, b) >= 0.0


# ─── shape_similarity ─────────────────────────────────────────────────────────

class TestShapeSimilarity:
    def test_identical_square_bboxes(self):
        bbox = (0, 0, 10, 10)
        assert shape_similarity(bbox, bbox) == pytest.approx(1.0)

    def test_identical_rect_bboxes(self):
        bbox = (5, 5, 20, 10)
        assert shape_similarity(bbox, bbox) == pytest.approx(1.0)

    def test_range(self):
        a = (0, 0, 10, 40)  # tall
        b = (0, 0, 40, 10)  # wide — same aspect ratio
        result = shape_similarity(a, b)
        assert 0.0 <= result <= 1.0

    def test_square_vs_square_equal_score(self):
        """Two squares of different sizes should have same aspect → 1.0."""
        a = (0, 0, 5, 5)
        b = (0, 0, 15, 15)
        assert shape_similarity(a, b) == pytest.approx(1.0)

    def test_different_aspect_ratios_lower(self):
        """Square vs very elongated rectangle → lower score."""
        sq = (0, 0, 10, 10)
        rect = (0, 0, 1, 100)
        score = shape_similarity(sq, rect)
        assert score < 1.0

    def test_symmetric(self):
        a = (0, 0, 10, 30)
        b = (0, 0, 20, 10)
        assert shape_similarity(a, b) == pytest.approx(shape_similarity(b, a))

    def test_non_negative(self):
        a = (0, 0, 1, 100)
        b = (0, 0, 100, 1)
        assert shape_similarity(a, b) >= 0.0

    def test_position_ignored(self):
        """x, y position should not affect aspect ratio similarity."""
        a = (0, 0, 10, 20)
        b = (100, 200, 10, 20)
        assert shape_similarity(a, b) == pytest.approx(1.0)


# ─── boundary_proximity ───────────────────────────────────────────────────────

class TestBoundaryProximity:
    def test_same_centroid_returns_one(self):
        c = (50.0, 50.0)
        assert boundary_proximity(c, c) == pytest.approx(1.0)

    def test_beyond_max_distance_returns_zero(self):
        a = (0.0, 0.0)
        b = (200.0, 0.0)
        result = boundary_proximity(a, b, max_distance=100.0)
        assert result == pytest.approx(0.0)

    def test_half_max_distance(self):
        a = (0.0, 0.0)
        b = (50.0, 0.0)
        result = boundary_proximity(a, b, max_distance=100.0)
        assert result == pytest.approx(0.5)

    def test_range(self):
        a = (10.0, 20.0)
        b = (30.0, 40.0)
        result = boundary_proximity(a, b)
        assert 0.0 <= result <= 1.0

    def test_max_distance_zero_raises(self):
        with pytest.raises(ValueError):
            boundary_proximity((0.0, 0.0), (1.0, 1.0), max_distance=0.0)

    def test_max_distance_negative_raises(self):
        with pytest.raises(ValueError):
            boundary_proximity((0.0, 0.0), (1.0, 1.0), max_distance=-5.0)

    def test_symmetric(self):
        a = (10.0, 20.0)
        b = (50.0, 80.0)
        assert boundary_proximity(a, b) == pytest.approx(boundary_proximity(b, a))

    def test_custom_max_distance(self):
        a = (0.0, 0.0)
        b = (10.0, 0.0)
        result = boundary_proximity(a, b, max_distance=20.0)
        assert result == pytest.approx(0.5)

    def test_non_negative(self):
        a = (0.0, 0.0)
        b = (1000.0, 1000.0)
        result = boundary_proximity(a, b, max_distance=100.0)
        assert result >= 0.0


# ─── score_region_pair ────────────────────────────────────────────────────────

class TestScoreRegionPair:
    def test_returns_region_score(self):
        pa = make_patch(value=100)
        pb = make_patch(value=100)
        ba = (0, 0, 20, 20)
        bb = (0, 0, 20, 20)
        result = score_region_pair(pa, ba, pb, bb)
        assert isinstance(result, RegionScore)

    def test_score_in_range(self):
        pa = make_noisy_patch(seed=1)
        pb = make_noisy_patch(seed=2)
        ba = (0, 0, 20, 20)
        bb = (50, 50, 20, 20)
        result = score_region_pair(pa, ba, pb, bb)
        assert 0.0 <= result.score <= 1.0

    def test_identical_patches_high_score(self):
        pa = make_patch(value=128)
        ba = (0, 0, 20, 20)
        result = score_region_pair(pa, ba, pa.copy(), ba)
        assert result.score > 0.5

    def test_channels_stored_in_result(self):
        pa = make_patch(value=100)
        pb = make_patch(value=200)
        ba = (0, 0, 20, 20)
        bb = (0, 0, 20, 20)
        result = score_region_pair(pa, ba, pb, bb)
        assert hasattr(result, "color_score")
        assert hasattr(result, "texture_score")
        assert hasattr(result, "shape_score")
        assert hasattr(result, "boundary_score")

    def test_custom_config(self):
        pa = make_patch(value=128)
        ba = (0, 0, 20, 20)
        cfg = RegionScorerConfig(max_distance=50.0)
        result = score_region_pair(pa, ba, pa.copy(), ba, cfg=cfg)
        assert isinstance(result, RegionScore)

    def test_default_config(self):
        pa = make_patch(value=128)
        ba = (0, 0, 20, 20)
        result = score_region_pair(pa, ba, pa.copy(), ba)
        assert "w_color" in result.params

    def test_bgr_patches(self):
        pa = make_bgr_patch()
        pb = make_bgr_patch(value=(150, 180, 200))
        ba = (0, 0, 20, 20)
        bb = (10, 10, 20, 20)
        result = score_region_pair(pa, ba, pb, bb)
        assert 0.0 <= result.score <= 1.0

    def test_close_centroids_higher_boundary_score(self):
        """Regions at same position should have high boundary score."""
        pa = make_patch(value=100)
        pb = make_patch(value=100)
        ba = (0, 0, 20, 20)   # centroid at (10, 10)
        bb = (0, 0, 20, 20)   # same centroid
        result = score_region_pair(pa, ba, pb, bb)
        assert result.boundary_score == pytest.approx(1.0)


# ─── batch_score_regions ──────────────────────────────────────────────────────

class TestBatchScoreRegions:
    def test_returns_list(self):
        pa = make_patch(value=100)
        ba = (0, 0, 20, 20)
        pairs = [(pa, ba, pa, ba) for _ in range(3)]
        results = batch_score_regions(pairs)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_region_scores(self):
        pa = make_patch(value=100)
        ba = (0, 0, 20, 20)
        pairs = [(pa, ba, pa, ba) for _ in range(4)]
        results = batch_score_regions(pairs)
        assert all(isinstance(r, RegionScore) for r in results)

    def test_empty_list(self):
        results = batch_score_regions([])
        assert results == []

    def test_custom_config(self):
        pa = make_patch(value=100)
        ba = (0, 0, 20, 20)
        cfg = RegionScorerConfig(max_distance=200.0)
        results = batch_score_regions([(pa, ba, pa, ba)], cfg=cfg)
        assert isinstance(results[0], RegionScore)

    def test_scores_in_range(self):
        patches = [make_noisy_patch(seed=i) for i in range(4)]
        bboxes = [(i * 30, 0, 20, 20) for i in range(4)]
        pairs = [(patches[i], bboxes[i], patches[j], bboxes[j])
                 for i in range(2) for j in range(2, 4)]
        results = batch_score_regions(pairs)
        for r in results:
            assert 0.0 <= r.score <= 1.0


# ─── rank_region_pairs ────────────────────────────────────────────────────────

class TestRankRegionPairs:
    def _make_scores(self, values):
        return [RegionScore(score=v, color_score=v, texture_score=v,
                            shape_score=v, boundary_score=v)
                for v in values]

    def test_returns_list_of_tuples(self):
        scores = self._make_scores([0.5, 0.8, 0.3])
        ranked = rank_region_pairs(scores)
        assert isinstance(ranked, list)
        for item in ranked:
            assert len(item) == 2

    def test_sorted_descending(self):
        scores = self._make_scores([0.3, 0.9, 0.1, 0.7])
        ranked = rank_region_pairs(scores)
        vals = [s for _, s in ranked]
        assert vals == sorted(vals, reverse=True)

    def test_default_indices(self):
        scores = self._make_scores([0.4, 0.6, 0.2])
        ranked = rank_region_pairs(scores)
        indices = [i for i, _ in ranked]
        assert set(indices) == {0, 1, 2}

    def test_custom_indices(self):
        scores = self._make_scores([0.3, 0.7, 0.5])
        ranked = rank_region_pairs(scores, indices=[100, 200, 300])
        indices = [i for i, _ in ranked]
        assert set(indices) == {100, 200, 300}

    def test_empty_scores(self):
        ranked = rank_region_pairs([])
        assert ranked == []

    def test_highest_first(self):
        scores = self._make_scores([0.2, 0.9, 0.5])
        ranked = rank_region_pairs(scores)
        assert ranked[0][1] == pytest.approx(0.9)
        assert ranked[0][0] == 1

    def test_single_score(self):
        scores = self._make_scores([0.75])
        ranked = rank_region_pairs(scores)
        assert len(ranked) == 1
        assert ranked[0] == (0, pytest.approx(0.75))

    def test_all_scores_equal(self):
        scores = self._make_scores([0.5, 0.5, 0.5])
        ranked = rank_region_pairs(scores)
        vals = [s for _, s in ranked]
        assert all(v == pytest.approx(0.5) for v in vals)
