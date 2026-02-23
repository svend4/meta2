"""Extra tests for puzzle_reconstruction.matching.edge_comparator."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.edge_comparator import (
    EdgeCompConfig,
    EdgeCompResult,
    EdgeSample,
    batch_compare_edges,
    compare_edge_gradient,
    compare_edge_intensity,
    compare_edge_pair,
    compare_edge_texture,
    extract_edge_sample,
    score_edge_comparison,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sample(fid=0, n=16, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.random(n).astype(float)
    return EdgeSample(fragment_id=fid, intensity=arr,
                      gradient=arr.copy(), texture=arr.copy())


def _image(h=16, w=16, seed=0):
    return (np.random.default_rng(seed).random((h, w)) * 255).astype(np.uint8)


# ─── TestEdgeCompConfigExtra ─────────────────────────────────────────────────

class TestEdgeCompConfigExtra:
    def test_default_strip_width(self):
        assert EdgeCompConfig().strip_width == 4

    def test_default_n_samples(self):
        assert EdgeCompConfig().n_samples == 32

    def test_default_use_gradient_true(self):
        assert EdgeCompConfig().use_gradient is True

    def test_default_use_texture_true(self):
        assert EdgeCompConfig().use_texture is True

    def test_default_normalize_true(self):
        assert EdgeCompConfig().normalize is True

    def test_strip_width_2_ok(self):
        assert EdgeCompConfig(strip_width=2).strip_width == 2

    def test_strip_width_0_raises(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(strip_width=0)

    def test_strip_width_neg_raises(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(strip_width=-1)

    def test_n_samples_16_ok(self):
        assert EdgeCompConfig(n_samples=16).n_samples == 16

    def test_n_samples_0_raises(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(n_samples=0)

    def test_n_samples_neg_raises(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(n_samples=-5)

    def test_use_gradient_false(self):
        assert EdgeCompConfig(use_gradient=False).use_gradient is False

    def test_use_texture_false(self):
        assert EdgeCompConfig(use_texture=False).use_texture is False


# ─── TestEdgeSampleExtra ─────────────────────────────────────────────────────

class TestEdgeSampleExtra:
    def test_fragment_id_stored(self):
        assert _sample(fid=5).fragment_id == 5

    def test_n_samples_property(self):
        assert _sample(n=24).n_samples == 24

    def test_mean_intensity(self):
        arr = np.array([0.0, 0.5, 1.0])
        s = EdgeSample(fragment_id=0, intensity=arr, gradient=arr.copy(), texture=arr.copy())
        assert abs(s.mean_intensity - 0.5) < 1e-9

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=-1, intensity=np.zeros(4),
                       gradient=np.zeros(4), texture=np.zeros(4))

    def test_2d_intensity_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=0, intensity=np.zeros((4, 4)),
                       gradient=np.zeros(4), texture=np.zeros(4))

    def test_mismatched_gradient_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=0, intensity=np.zeros(4),
                       gradient=np.zeros(5), texture=np.zeros(4))

    def test_mismatched_texture_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=0, intensity=np.zeros(4),
                       gradient=np.zeros(4), texture=np.zeros(6))


# ─── TestEdgeCompResultExtra ─────────────────────────────────────────────────

class TestEdgeCompResultExtra:
    def _make(self, total=0.7):
        return EdgeCompResult(pair=(0, 1), intensity_score=0.8,
                              gradient_score=0.7, texture_score=0.6,
                              total_score=total)

    def test_fragment_a(self):
        assert self._make().fragment_a == 0

    def test_fragment_b(self):
        assert self._make().fragment_b == 1

    def test_is_good_match_above_threshold(self):
        assert self._make(total=0.75).is_good_match is True

    def test_is_good_match_below_threshold(self):
        assert self._make(total=0.5).is_good_match is False

    def test_intensity_above_1_raises(self):
        with pytest.raises(ValueError):
            EdgeCompResult(pair=(0, 1), intensity_score=1.5,
                           gradient_score=0.5, texture_score=0.5, total_score=0.5)

    def test_total_neg_raises(self):
        with pytest.raises(ValueError):
            EdgeCompResult(pair=(0, 1), intensity_score=0.5,
                           gradient_score=0.5, texture_score=0.5, total_score=-0.1)


# ─── TestExtractEdgeSampleExtra ───────────────────────────────────────────────

class TestExtractEdgeSampleExtra:
    def test_2d_image_ok(self):
        s = extract_edge_sample(_image(16, 16), fragment_id=0,
                                cfg=EdgeCompConfig(n_samples=8, strip_width=2))
        assert s.n_samples == 8

    def test_3d_image_ok(self):
        img = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
        s = extract_edge_sample(img, cfg=EdgeCompConfig(n_samples=8))
        assert s.n_samples == 8

    def test_fragment_id_stored(self):
        s = extract_edge_sample(_image(), fragment_id=7)
        assert s.fragment_id == 7

    def test_normalized_range(self):
        s = extract_edge_sample(_image(32, 32), cfg=EdgeCompConfig(n_samples=16, normalize=True))
        assert float(s.intensity.min()) >= -1e-9
        assert float(s.intensity.max()) <= 1.0 + 1e-9

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            extract_edge_sample(np.zeros((8,)), cfg=EdgeCompConfig())


# ─── TestCompareEdgeIntensityExtra ───────────────────────────────────────────

class TestCompareEdgeIntensityExtra:
    def test_identical_is_1(self):
        s = _sample(seed=42)
        assert compare_edge_intensity(s, s) == pytest.approx(1.0, abs=1e-6)

    def test_in_0_1(self):
        a = _sample(seed=1)
        b = _sample(fid=1, seed=2)
        assert 0.0 <= compare_edge_intensity(a, b) <= 1.0

    def test_returns_float(self):
        a = _sample(seed=3)
        b = _sample(fid=1, seed=4)
        assert isinstance(compare_edge_intensity(a, b), float)


# ─── TestCompareEdgeGradientExtra ─────────────────────────────────────────────

class TestCompareEdgeGradientExtra:
    def test_identical_is_1(self):
        s = _sample(seed=10)
        assert compare_edge_gradient(s, s) == pytest.approx(1.0, abs=1e-6)

    def test_in_0_1(self):
        a = _sample(seed=5)
        b = _sample(fid=1, seed=6)
        assert 0.0 <= compare_edge_gradient(a, b) <= 1.0


# ─── TestCompareEdgeTextureExtra ─────────────────────────────────────────────

class TestCompareEdgeTextureExtra:
    def test_identical_is_1(self):
        s = _sample(seed=20)
        assert compare_edge_texture(s, s) == pytest.approx(1.0, abs=1e-6)

    def test_in_0_1(self):
        a = _sample(seed=7)
        b = _sample(fid=1, seed=8)
        assert 0.0 <= compare_edge_texture(a, b) <= 1.0


# ─── TestScoreEdgeComparisonExtra ─────────────────────────────────────────────

class TestScoreEdgeComparisonExtra:
    def test_all_components_averaged(self):
        cfg = EdgeCompConfig(use_gradient=True, use_texture=True)
        score = score_edge_comparison(0.9, 0.8, 0.7, cfg)
        assert abs(score - (0.9 + 0.8 + 0.7) / 3) < 1e-6

    def test_intensity_only(self):
        cfg = EdgeCompConfig(use_gradient=False, use_texture=False)
        assert abs(score_edge_comparison(0.6, 0.9, 0.9, cfg) - 0.6) < 1e-6

    def test_intensity_and_gradient_only(self):
        cfg = EdgeCompConfig(use_gradient=True, use_texture=False)
        assert abs(score_edge_comparison(0.4, 0.8, 0.0, cfg) - 0.6) < 1e-6

    def test_in_0_1(self):
        assert 0.0 <= score_edge_comparison(1.0, 1.0, 1.0) <= 1.0

    def test_default_config_ok(self):
        assert 0.0 <= score_edge_comparison(0.5, 0.5, 0.5) <= 1.0


# ─── TestCompareEdgePairExtra ─────────────────────────────────────────────────

class TestCompareEdgePairExtra:
    def test_pair_ids(self):
        a = _sample(fid=0, seed=0)
        b = _sample(fid=1, seed=1)
        r = compare_edge_pair(a, b)
        assert r.pair == (0, 1)

    def test_total_score_in_0_1(self):
        a = _sample(seed=2)
        b = _sample(fid=1, seed=3)
        assert 0.0 <= compare_edge_pair(a, b).total_score <= 1.0

    def test_identical_high_score(self):
        s = _sample(seed=42)
        assert compare_edge_pair(s, s).total_score > 0.9

    def test_scores_keys(self):
        a = _sample(seed=0)
        b = _sample(fid=1, seed=1)
        r = compare_edge_pair(a, b)
        assert "intensity" in r.scores
        assert "gradient" in r.scores
        assert "texture" in r.scores


# ─── TestBatchCompareEdgesExtra ───────────────────────────────────────────────

class TestBatchCompareEdgesExtra:
    def test_empty_returns_empty(self):
        assert batch_compare_edges([]) == []

    def test_single_returns_empty(self):
        assert batch_compare_edges([_sample()]) == []

    def test_two_samples_one_pair(self):
        results = batch_compare_edges([_sample(fid=0), _sample(fid=1, seed=1)])
        assert len(results) == 1

    def test_three_samples_three_pairs(self):
        samples = [_sample(fid=i, seed=i) for i in range(3)]
        assert len(batch_compare_edges(samples)) == 3

    def test_four_samples_six_pairs(self):
        samples = [_sample(fid=i, seed=i) for i in range(4)]
        assert len(batch_compare_edges(samples)) == 6

    def test_all_scores_valid(self):
        samples = [_sample(fid=i, seed=i) for i in range(4)]
        for r in batch_compare_edges(samples):
            assert 0.0 <= r.total_score <= 1.0

    def test_all_pairs_present(self):
        samples = [_sample(fid=i, seed=i) for i in range(3)]
        pairs = {r.pair for r in batch_compare_edges(samples)}
        assert (0, 1) in pairs
        assert (0, 2) in pairs
        assert (1, 2) in pairs

    def test_custom_cfg(self):
        cfg = EdgeCompConfig(n_samples=16, use_gradient=False, use_texture=False)
        samples = [_sample(fid=i, n=16, seed=i) for i in range(3)]
        results = batch_compare_edges(samples, cfg)
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.total_score <= 1.0
