"""Extra tests for puzzle_reconstruction/matching/edge_comparator.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.edge_comparator import (
    EdgeCompConfig,
    EdgeSample,
    EdgeCompResult,
    extract_edge_sample,
    compare_edge_intensity,
    compare_edge_gradient,
    compare_edge_texture,
    score_edge_comparison,
    compare_edge_pair,
    batch_compare_edges,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _sample(fid=0, n=32):
    return EdgeSample(
        fragment_id=fid,
        intensity=np.random.rand(n),
        gradient=np.random.rand(n),
        texture=np.random.rand(n),
    )


def _const_sample(fid=0, n=32, val=0.5):
    return EdgeSample(
        fragment_id=fid,
        intensity=np.full(n, val),
        gradient=np.full(n, val),
        texture=np.full(n, val),
    )


# ─── EdgeCompConfig ─────────────────────────────────────────────────────────

class TestEdgeCompConfigExtra:
    def test_defaults(self):
        cfg = EdgeCompConfig()
        assert cfg.strip_width == 4
        assert cfg.n_samples == 32
        assert cfg.use_gradient is True
        assert cfg.use_texture is True
        assert cfg.normalize is True

    def test_zero_strip_width_raises(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(strip_width=0)

    def test_zero_n_samples_raises(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(n_samples=0)

    def test_custom(self):
        cfg = EdgeCompConfig(strip_width=8, n_samples=64,
                             use_gradient=False)
        assert cfg.strip_width == 8
        assert cfg.use_gradient is False


# ─── EdgeSample ─────────────────────────────────────────────────────────────

class TestEdgeSampleExtra:
    def test_negative_fid_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=-1,
                       intensity=np.zeros(10),
                       gradient=np.zeros(10),
                       texture=np.zeros(10))

    def test_2d_intensity_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=0,
                       intensity=np.zeros((5, 2)),
                       gradient=np.zeros(10),
                       texture=np.zeros(10))

    def test_shape_mismatch_gradient_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=0,
                       intensity=np.zeros(10),
                       gradient=np.zeros(5),
                       texture=np.zeros(10))

    def test_shape_mismatch_texture_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=0,
                       intensity=np.zeros(10),
                       gradient=np.zeros(10),
                       texture=np.zeros(5))

    def test_n_samples(self):
        s = _sample(n=16)
        assert s.n_samples == 16

    def test_mean_intensity(self):
        s = _const_sample(val=0.5)
        assert s.mean_intensity == pytest.approx(0.5)


# ─── EdgeCompResult ─────────────────────────────────────────────────────────

class TestEdgeCompResultExtra:
    def test_fields_stored(self):
        r = EdgeCompResult(pair=(0, 1), intensity_score=0.8,
                           gradient_score=0.7, texture_score=0.6,
                           total_score=0.7)
        assert r.fragment_a == 0
        assert r.fragment_b == 1

    def test_is_good_match_true(self):
        r = EdgeCompResult(pair=(0, 1), intensity_score=0.8,
                           gradient_score=0.7, texture_score=0.7,
                           total_score=0.75)
        assert r.is_good_match is True

    def test_is_good_match_false(self):
        r = EdgeCompResult(pair=(0, 1), intensity_score=0.5,
                           gradient_score=0.4, texture_score=0.3,
                           total_score=0.4)
        assert r.is_good_match is False

    def test_out_of_range_intensity_raises(self):
        with pytest.raises(ValueError):
            EdgeCompResult(pair=(0, 1), intensity_score=1.5,
                           gradient_score=0.5, texture_score=0.5,
                           total_score=0.5)

    def test_negative_total_raises(self):
        with pytest.raises(ValueError):
            EdgeCompResult(pair=(0, 1), intensity_score=0.5,
                           gradient_score=0.5, texture_score=0.5,
                           total_score=-0.1)


# ─── extract_edge_sample ────────────────────────────────────────────────────

class TestExtractEdgeSampleExtra:
    def test_grayscale_image(self):
        img = np.random.randint(0, 256, (50, 40), dtype=np.uint8)
        s = extract_edge_sample(img, fragment_id=0)
        assert isinstance(s, EdgeSample)
        assert s.n_samples == 32

    def test_color_image(self):
        img = np.random.randint(0, 256, (50, 40, 3), dtype=np.uint8)
        s = extract_edge_sample(img)
        assert s.n_samples == 32

    def test_custom_config(self):
        img = np.random.randint(0, 256, (50, 40), dtype=np.uint8)
        cfg = EdgeCompConfig(n_samples=16, strip_width=2)
        s = extract_edge_sample(img, cfg=cfg)
        assert s.n_samples == 16

    def test_no_normalize(self):
        img = np.random.randint(0, 256, (50, 40), dtype=np.uint8)
        cfg = EdgeCompConfig(normalize=False)
        s = extract_edge_sample(img, cfg=cfg)
        assert s.n_samples == 32

    def test_1d_image_raises(self):
        with pytest.raises(ValueError):
            extract_edge_sample(np.zeros(10))


# ─── compare functions ──────────────────────────────────────────────────────

class TestCompareEdgeFuncsExtra:
    def test_intensity_identical(self):
        s = _const_sample(n=32, val=0.5)
        score = compare_edge_intensity(s, s)
        assert 0.0 <= score <= 1.0

    def test_gradient_identical(self):
        s = _const_sample(n=32, val=0.5)
        score = compare_edge_gradient(s, s)
        assert 0.0 <= score <= 1.0

    def test_texture_identical(self):
        s = _const_sample(n=32, val=0.5)
        score = compare_edge_texture(s, s)
        assert 0.0 <= score <= 1.0

    def test_intensity_range(self):
        a = _sample(n=32)
        b = _sample(fid=1, n=32)
        score = compare_edge_intensity(a, b)
        assert 0.0 <= score <= 1.0


# ─── score_edge_comparison ──────────────────────────────────────────────────

class TestScoreEdgeComparisonExtra:
    def test_all_enabled(self):
        s = score_edge_comparison(0.8, 0.6, 0.7)
        assert s == pytest.approx(0.7, abs=0.01)

    def test_gradient_disabled(self):
        cfg = EdgeCompConfig(use_gradient=False)
        s = score_edge_comparison(0.8, 0.0, 0.6, cfg)
        # (0.8 + 0.6) / 2 = 0.7
        assert s == pytest.approx(0.7, abs=0.01)

    def test_both_disabled(self):
        cfg = EdgeCompConfig(use_gradient=False, use_texture=False)
        s = score_edge_comparison(0.8, 0.0, 0.0, cfg)
        assert s == pytest.approx(0.8, abs=0.01)


# ─── compare_edge_pair ──────────────────────────────────────────────────────

class TestCompareEdgePairExtra:
    def test_returns_result(self):
        a = _sample(fid=0, n=32)
        b = _sample(fid=1, n=32)
        r = compare_edge_pair(a, b)
        assert isinstance(r, EdgeCompResult)
        assert r.pair == (0, 1)
        assert 0.0 <= r.total_score <= 1.0

    def test_scores_dict_populated(self):
        a = _sample(fid=0, n=32)
        b = _sample(fid=1, n=32)
        r = compare_edge_pair(a, b)
        assert "intensity" in r.scores
        assert "gradient" in r.scores
        assert "texture" in r.scores


# ─── batch_compare_edges ────────────────────────────────────────────────────

class TestBatchCompareEdgesExtra:
    def test_empty(self):
        assert batch_compare_edges([]) == []

    def test_single_no_pairs(self):
        assert batch_compare_edges([_sample()]) == []

    def test_two_samples(self):
        results = batch_compare_edges([_sample(fid=0), _sample(fid=1)])
        assert len(results) == 1

    def test_three_samples(self):
        samples = [_sample(fid=i) for i in range(3)]
        results = batch_compare_edges(samples)
        assert len(results) == 3  # C(3,2) = 3
