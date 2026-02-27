"""Тесты для puzzle_reconstruction/matching/edge_comparator.py."""
import pytest
import numpy as np

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


def _make_sample(fid=0, n=32, val=0.5):
    arr = np.full(n, val)
    return EdgeSample(fragment_id=fid, intensity=arr, gradient=arr * 0.1, texture=arr * 0.05)


def _make_img(h=40, w=40, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


class TestEdgeCompConfig:
    def test_defaults(self):
        c = EdgeCompConfig()
        assert c.strip_width == 4
        assert c.n_samples == 32
        assert c.use_gradient

    def test_strip_width_zero_raises(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(strip_width=0)

    def test_n_samples_zero_raises(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(n_samples=0)


class TestEdgeSample:
    def test_n_samples_property(self):
        s = _make_sample(n=16)
        assert s.n_samples == 16

    def test_mean_intensity(self):
        s = _make_sample(val=0.6)
        assert s.mean_intensity == pytest.approx(0.6)

    def test_negative_fid_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=-1, intensity=np.zeros(8),
                       gradient=np.zeros(8), texture=np.zeros(8))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=0, intensity=np.zeros(8),
                       gradient=np.zeros(4), texture=np.zeros(8))


class TestEdgeCompResult:
    def test_is_good_match(self):
        r = EdgeCompResult(pair=(0, 1), intensity_score=0.8, gradient_score=0.8,
                           texture_score=0.8, total_score=0.8)
        assert r.is_good_match

    def test_not_good_match_below_threshold(self):
        r = EdgeCompResult(pair=(0, 1), intensity_score=0.5, gradient_score=0.5,
                           texture_score=0.5, total_score=0.5)
        assert not r.is_good_match

    def test_fragment_ab_properties(self):
        r = EdgeCompResult(pair=(3, 7), intensity_score=0.5, gradient_score=0.5,
                           texture_score=0.5, total_score=0.5)
        assert r.fragment_a == 3
        assert r.fragment_b == 7


class TestExtractEdgeSample:
    def test_returns_edge_sample(self):
        img = _make_img()
        s = extract_edge_sample(img, fragment_id=0)
        assert isinstance(s, EdgeSample)

    def test_correct_n_samples(self):
        img = _make_img()
        cfg = EdgeCompConfig(n_samples=16)
        s = extract_edge_sample(img, fragment_id=0, cfg=cfg)
        assert s.n_samples == 16

    def test_grayscale_input(self):
        img = np.zeros((40, 40), dtype=np.uint8)
        s = extract_edge_sample(img, fragment_id=0)
        assert isinstance(s, EdgeSample)

    def test_invalid_ndim_raises(self):
        img = np.zeros((4,), dtype=np.uint8)
        with pytest.raises(ValueError):
            extract_edge_sample(img)


class TestCompareEdges:
    def test_identical_intensity_score_high(self):
        # Use non-constant profile so NCC is well-defined
        arr = np.sin(np.linspace(0, np.pi, 32))
        s = EdgeSample(fragment_id=0, intensity=arr, gradient=arr * 0.1, texture=arr * 0.05)
        score = compare_edge_intensity(s, s)
        assert score > 0.9

    def test_intensity_score_in_range(self):
        s1 = _make_sample(val=0.2)
        s2 = _make_sample(val=0.8)
        score = compare_edge_intensity(s1, s2)
        assert 0.0 <= score <= 1.0

    def test_gradient_score_in_range(self):
        s1 = _make_sample(val=0.3)
        s2 = _make_sample(val=0.7)
        score = compare_edge_gradient(s1, s2)
        assert 0.0 <= score <= 1.0

    def test_texture_score_identical(self):
        s = _make_sample(val=0.5)
        score = compare_edge_texture(s, s)
        assert score > 0.9


class TestCompareEdgePair:
    def test_returns_edge_comp_result(self):
        s1 = _make_sample(fid=0)
        s2 = _make_sample(fid=1)
        r = compare_edge_pair(s1, s2)
        assert isinstance(r, EdgeCompResult)

    def test_identical_samples_reasonable_score(self):
        # Constant profiles have zero variance → NCC returns 0.5; total ≥ 0.5
        s = _make_sample(val=0.5)
        s2 = _make_sample(fid=1, val=0.5)
        r = compare_edge_pair(s, s2)
        assert r.total_score >= 0.5

    def test_pair_ids_correct(self):
        s1 = _make_sample(fid=3)
        s2 = _make_sample(fid=5)
        r = compare_edge_pair(s1, s2)
        assert r.pair == (3, 5)


class TestBatchCompareEdges:
    def test_n_pairs(self):
        samples = [_make_sample(fid=i) for i in range(4)]
        results = batch_compare_edges(samples)
        assert len(results) == 6  # C(4,2)

    def test_empty_returns_empty(self):
        assert batch_compare_edges([]) == []

    def test_single_returns_empty(self):
        assert batch_compare_edges([_make_sample()]) == []
