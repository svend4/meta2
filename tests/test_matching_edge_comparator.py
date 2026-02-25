"""Тесты для puzzle_reconstruction.matching.edge_comparator."""
import pytest
import numpy as np
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


def _image(h=32, w=32) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def _sample(fid=0, n=16) -> EdgeSample:
    rng = np.random.default_rng(fid + 1)
    return EdgeSample(
        fragment_id=fid,
        intensity=rng.random(n).astype(np.float64),
        gradient=rng.random(n).astype(np.float64),
        texture=rng.random(n).astype(np.float64),
    )


# ─── TestEdgeCompConfig ───────────────────────────────────────────────────────

class TestEdgeCompConfig:
    def test_defaults(self):
        cfg = EdgeCompConfig()
        assert cfg.strip_width == 4
        assert cfg.n_samples == 32
        assert cfg.use_gradient is True
        assert cfg.use_texture is True
        assert cfg.normalize is True

    def test_custom_values(self):
        cfg = EdgeCompConfig(strip_width=8, n_samples=64, normalize=False)
        assert cfg.strip_width == 8
        assert cfg.n_samples == 64

    def test_strip_width_zero_raises(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(strip_width=0)

    def test_n_samples_zero_raises(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(n_samples=0)


# ─── TestEdgeSample ───────────────────────────────────────────────────────────

class TestEdgeSample:
    def test_basic_construction(self):
        s = _sample(fid=3, n=16)
        assert s.fragment_id == 3
        assert s.n_samples == 16

    def test_mean_intensity(self):
        arr = np.array([0.2, 0.4, 0.6, 0.8])
        s = EdgeSample(fragment_id=0,
                       intensity=arr,
                       gradient=arr.copy(),
                       texture=arr.copy())
        assert s.mean_intensity == pytest.approx(0.5)

    def test_fragment_id_neg_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=-1,
                       intensity=np.zeros(4),
                       gradient=np.zeros(4),
                       texture=np.zeros(4))

    def test_non_1d_intensity_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=0,
                       intensity=np.zeros((4, 4)),
                       gradient=np.zeros(4),
                       texture=np.zeros(4))

    def test_shape_mismatch_gradient_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=0,
                       intensity=np.zeros(4),
                       gradient=np.zeros(8),
                       texture=np.zeros(4))

    def test_shape_mismatch_texture_raises(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=0,
                       intensity=np.zeros(4),
                       gradient=np.zeros(4),
                       texture=np.zeros(8))


# ─── TestEdgeCompResult ───────────────────────────────────────────────────────

class TestEdgeCompResult:
    def _make(self, total=0.7) -> EdgeCompResult:
        return EdgeCompResult(
            pair=(0, 1),
            intensity_score=0.8,
            gradient_score=0.7,
            texture_score=0.6,
            total_score=total,
        )

    def test_fragment_a(self):
        r = self._make()
        assert r.fragment_a == 0

    def test_fragment_b(self):
        r = self._make()
        assert r.fragment_b == 1

    def test_is_good_match_true(self):
        r = self._make(total=0.75)
        assert r.is_good_match is True

    def test_is_good_match_false(self):
        r = self._make(total=0.65)
        assert r.is_good_match is False

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            EdgeCompResult(pair=(0, 1),
                           intensity_score=1.1,
                           gradient_score=0.5,
                           texture_score=0.5,
                           total_score=0.5)


# ─── TestExtractEdgeSample ────────────────────────────────────────────────────

class TestExtractEdgeSample:
    def test_returns_edge_sample(self):
        img = _image()
        s = extract_edge_sample(img)
        assert isinstance(s, EdgeSample)

    def test_n_samples_correct(self):
        img = _image()
        cfg = EdgeCompConfig(n_samples=16)
        s = extract_edge_sample(img, cfg=cfg)
        assert s.n_samples == 16

    def test_fragment_id_stored(self):
        img = _image()
        s = extract_edge_sample(img, fragment_id=5)
        assert s.fragment_id == 5

    def test_bgr_image(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        s = extract_edge_sample(img)
        assert isinstance(s, EdgeSample)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError):
            extract_edge_sample(np.zeros((4,)))


# ─── TestCompareEdgeIntensity ─────────────────────────────────────────────────

class TestCompareEdgeIntensity:
    def test_identical_samples_high_score(self):
        s = _sample(n=16)
        score = compare_edge_intensity(s, s)
        assert score > 0.9

    def test_returns_float_in_range(self):
        a = _sample(fid=0, n=16)
        b = _sample(fid=1, n=16)
        score = compare_edge_intensity(a, b)
        assert 0.0 <= score <= 1.0


# ─── TestCompareEdgeGradient ──────────────────────────────────────────────────

class TestCompareEdgeGradient:
    def test_identical_samples(self):
        s = _sample(n=16)
        score = compare_edge_gradient(s, s)
        assert score > 0.9

    def test_in_range(self):
        a = _sample(fid=0, n=16)
        b = _sample(fid=2, n=16)
        score = compare_edge_gradient(a, b)
        assert 0.0 <= score <= 1.0


# ─── TestCompareEdgeTexture ───────────────────────────────────────────────────

class TestCompareEdgeTexture:
    def test_identical_samples(self):
        s = _sample(n=16)
        score = compare_edge_texture(s, s)
        assert score >= 0.9

    def test_in_range(self):
        a = _sample(fid=0, n=16)
        b = _sample(fid=3, n=16)
        score = compare_edge_texture(a, b)
        assert 0.0 <= score <= 1.0


# ─── TestScoreEdgeComparison ──────────────────────────────────────────────────

class TestScoreEdgeComparison:
    def test_perfect_scores(self):
        score = score_edge_comparison(1.0, 1.0, 1.0)
        assert score == pytest.approx(1.0)

    def test_result_in_range(self):
        score = score_edge_comparison(0.7, 0.5, 0.3)
        assert 0.0 <= score <= 1.0

    def test_no_gradient_no_texture(self):
        cfg = EdgeCompConfig(use_gradient=False, use_texture=False)
        score = score_edge_comparison(0.8, 0.0, 0.0, cfg)
        assert score == pytest.approx(0.8)


# ─── TestCompareEdgePair ──────────────────────────────────────────────────────

class TestCompareEdgePair:
    def test_returns_edge_comp_result(self):
        a = _sample(fid=0, n=16)
        b = _sample(fid=1, n=16)
        r = compare_edge_pair(a, b)
        assert isinstance(r, EdgeCompResult)

    def test_pair_ids_correct(self):
        a = _sample(fid=2, n=16)
        b = _sample(fid=5, n=16)
        r = compare_edge_pair(a, b)
        assert r.pair == (2, 5)

    def test_total_score_in_range(self):
        a = _sample(fid=0, n=16)
        b = _sample(fid=1, n=16)
        r = compare_edge_pair(a, b)
        assert 0.0 <= r.total_score <= 1.0

    def test_scores_dict_has_keys(self):
        a = _sample(fid=0, n=16)
        b = _sample(fid=1, n=16)
        r = compare_edge_pair(a, b)
        assert "intensity" in r.scores
        assert "gradient" in r.scores
        assert "texture" in r.scores


# ─── TestBatchCompareEdges ────────────────────────────────────────────────────

class TestBatchCompareEdges:
    def test_returns_list(self):
        samples = [_sample(i, n=16) for i in range(4)]
        results = batch_compare_edges(samples)
        assert isinstance(results, list)

    def test_n_pairs_correct(self):
        samples = [_sample(i, n=16) for i in range(4)]
        results = batch_compare_edges(samples)
        # C(4,2) = 6
        assert len(results) == 6

    def test_empty_list(self):
        assert batch_compare_edges([]) == []

    def test_single_sample(self):
        samples = [_sample(0, n=16)]
        assert batch_compare_edges(samples) == []
