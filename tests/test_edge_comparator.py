"""Тесты для puzzle_reconstruction.matching.edge_comparator."""
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

def _sample(fragment_id: int = 0, n: int = 8, seed: int = 0) -> EdgeSample:
    rng = np.random.default_rng(seed)
    return EdgeSample(
        fragment_id=fragment_id,
        intensity=rng.random(n).astype(float),
        gradient=rng.random(n).astype(float),
        texture=rng.random(n).astype(float),
    )


def _image(h: int = 16, w: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) * 255).astype(np.uint8)


# ─── TestEdgeCompConfig ───────────────────────────────────────────────────────

class TestEdgeCompConfig:
    def test_defaults(self):
        cfg = EdgeCompConfig()
        assert cfg.strip_width == 4
        assert cfg.n_samples == 32
        assert cfg.use_gradient is True
        assert cfg.use_texture is True
        assert cfg.normalize is True

    def test_valid_custom(self):
        cfg = EdgeCompConfig(strip_width=2, n_samples=16, use_gradient=False)
        assert cfg.strip_width == 2
        assert cfg.n_samples == 16
        assert cfg.use_gradient is False

    def test_invalid_strip_width_zero(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(strip_width=0)

    def test_invalid_strip_width_neg(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(strip_width=-1)

    def test_invalid_n_samples_zero(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(n_samples=0)

    def test_invalid_n_samples_neg(self):
        with pytest.raises(ValueError):
            EdgeCompConfig(n_samples=-5)


# ─── TestEdgeSample ───────────────────────────────────────────────────────────

class TestEdgeSample:
    def test_basic(self):
        s = _sample(fragment_id=3, n=8)
        assert s.fragment_id == 3
        assert s.n_samples == 8

    def test_n_samples_property(self):
        s = _sample(n=16)
        assert s.n_samples == 16

    def test_mean_intensity(self):
        arr = np.array([0.0, 0.5, 1.0])
        s = EdgeSample(fragment_id=0, intensity=arr,
                       gradient=arr.copy(), texture=arr.copy())
        assert abs(s.mean_intensity - 0.5) < 1e-9

    def test_invalid_fragment_id(self):
        arr = np.zeros(4)
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=-1, intensity=arr,
                       gradient=arr.copy(), texture=arr.copy())

    def test_invalid_intensity_2d(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=0,
                       intensity=np.zeros((4, 4)),
                       gradient=np.zeros(4),
                       texture=np.zeros(4))

    def test_invalid_gradient_shape(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=0,
                       intensity=np.zeros(4),
                       gradient=np.zeros(5),
                       texture=np.zeros(4))

    def test_invalid_texture_shape(self):
        with pytest.raises(ValueError):
            EdgeSample(fragment_id=0,
                       intensity=np.zeros(4),
                       gradient=np.zeros(4),
                       texture=np.zeros(6))


# ─── TestEdgeCompResult ───────────────────────────────────────────────────────

class TestEdgeCompResult:
    def _make(self, i=0.8, g=0.7, t=0.6, total=0.7):
        return EdgeCompResult(
            pair=(0, 1),
            intensity_score=i,
            gradient_score=g,
            texture_score=t,
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

    def test_is_good_match_exact_threshold(self):
        r = self._make(total=0.7)
        assert r.is_good_match is True

    def test_is_good_match_false(self):
        r = self._make(total=0.5)
        assert r.is_good_match is False

    def test_invalid_intensity_score(self):
        with pytest.raises(ValueError):
            EdgeCompResult(pair=(0, 1), intensity_score=1.5,
                           gradient_score=0.5, texture_score=0.5, total_score=0.5)

    def test_invalid_total_score_neg(self):
        with pytest.raises(ValueError):
            EdgeCompResult(pair=(0, 1), intensity_score=0.5,
                           gradient_score=0.5, texture_score=0.5, total_score=-0.1)

    def test_scores_dict_populated(self):
        s_a = _sample(0, n=8, seed=0)
        s_b = _sample(1, n=8, seed=1)
        r = compare_edge_pair(s_a, s_b)
        assert "intensity" in r.scores
        assert "gradient" in r.scores
        assert "texture" in r.scores


# ─── TestExtractEdgeSample ────────────────────────────────────────────────────

class TestExtractEdgeSample:
    def test_basic_2d(self):
        img = _image(16, 16)
        cfg = EdgeCompConfig(n_samples=8, strip_width=2)
        s = extract_edge_sample(img, fragment_id=0, cfg=cfg)
        assert s.fragment_id == 0
        assert s.n_samples == 8

    def test_basic_3d(self):
        img = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
        cfg = EdgeCompConfig(n_samples=8)
        s = extract_edge_sample(img, cfg=cfg)
        assert s.n_samples == 8

    def test_normalized_range(self):
        img = _image(32, 32)
        cfg = EdgeCompConfig(n_samples=16, normalize=True)
        s = extract_edge_sample(img, cfg=cfg)
        assert float(s.intensity.min()) >= -1e-9
        assert float(s.intensity.max()) <= 1.0 + 1e-9

    def test_no_normalize(self):
        img = _image(32, 32, seed=5)
        cfg = EdgeCompConfig(n_samples=16, normalize=False)
        s = extract_edge_sample(img, cfg=cfg)
        # Should not be clipped to [0,1] – raw values can exceed 1
        assert s.intensity is not None

    def test_default_config(self):
        img = _image(64, 64)
        s = extract_edge_sample(img, fragment_id=7)
        assert s.fragment_id == 7

    def test_invalid_image_dim(self):
        with pytest.raises(ValueError):
            extract_edge_sample(np.zeros((8,)), cfg=EdgeCompConfig())

    def test_narrow_image(self):
        img = np.ones((16, 1), dtype=np.uint8) * 128
        cfg = EdgeCompConfig(strip_width=4, n_samples=4)
        s = extract_edge_sample(img, cfg=cfg)
        assert s.n_samples == 4


# ─── TestCompareEdgeIntensity ─────────────────────────────────────────────────

class TestCompareEdgeIntensity:
    def test_identical_is_one(self):
        s = _sample(0, n=16, seed=42)
        score = compare_edge_intensity(s, s)
        assert abs(score - 1.0) < 1e-6

    def test_output_range(self):
        a = _sample(0, n=16, seed=1)
        b = _sample(1, n=16, seed=2)
        score = compare_edge_intensity(a, b)
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        a = _sample(0, n=8, seed=3)
        b = _sample(1, n=8, seed=4)
        assert isinstance(compare_edge_intensity(a, b), float)


# ─── TestCompareEdgeGradient ──────────────────────────────────────────────────

class TestCompareEdgeGradient:
    def test_identical_is_one(self):
        s = _sample(0, n=16, seed=10)
        assert abs(compare_edge_gradient(s, s) - 1.0) < 1e-6

    def test_output_range(self):
        a = _sample(0, n=16, seed=5)
        b = _sample(1, n=16, seed=6)
        score = compare_edge_gradient(a, b)
        assert 0.0 <= score <= 1.0


# ─── TestCompareEdgeTexture ───────────────────────────────────────────────────

class TestCompareEdgeTexture:
    def test_identical_is_one(self):
        s = _sample(0, n=16, seed=20)
        assert abs(compare_edge_texture(s, s) - 1.0) < 1e-6

    def test_output_range(self):
        a = _sample(0, n=16, seed=7)
        b = _sample(1, n=16, seed=8)
        score = compare_edge_texture(a, b)
        assert 0.0 <= score <= 1.0


# ─── TestScoreEdgeComparison ──────────────────────────────────────────────────

class TestScoreEdgeComparison:
    def test_all_components(self):
        cfg = EdgeCompConfig(use_gradient=True, use_texture=True)
        score = score_edge_comparison(0.9, 0.8, 0.7, cfg)
        expected = (0.9 + 0.8 + 0.7) / 3
        assert abs(score - expected) < 1e-6

    def test_no_gradient_no_texture(self):
        cfg = EdgeCompConfig(use_gradient=False, use_texture=False)
        score = score_edge_comparison(0.6, 0.9, 0.9, cfg)
        assert abs(score - 0.6) < 1e-6

    def test_only_gradient(self):
        cfg = EdgeCompConfig(use_gradient=True, use_texture=False)
        score = score_edge_comparison(0.4, 0.8, 0.0, cfg)
        assert abs(score - 0.6) < 1e-6

    def test_output_clipped(self):
        cfg = EdgeCompConfig()
        score = score_edge_comparison(1.0, 1.0, 1.0, cfg)
        assert 0.0 <= score <= 1.0

    def test_default_config(self):
        score = score_edge_comparison(0.5, 0.5, 0.5)
        assert 0.0 <= score <= 1.0


# ─── TestCompareEdgePair ──────────────────────────────────────────────────────

class TestCompareEdgePair:
    def test_basic(self):
        a = _sample(0, n=16, seed=0)
        b = _sample(1, n=16, seed=1)
        r = compare_edge_pair(a, b)
        assert r.pair == (0, 1)
        assert 0.0 <= r.total_score <= 1.0

    def test_symmetric_pair_ids(self):
        a = _sample(3, n=8)
        b = _sample(7, n=8, seed=9)
        r = compare_edge_pair(a, b)
        assert r.fragment_a == 3
        assert r.fragment_b == 7

    def test_identical_gives_high_score(self):
        s = _sample(0, n=32, seed=42)
        r = compare_edge_pair(s, s)
        assert r.total_score > 0.9

    def test_default_cfg(self):
        a = _sample(0, n=32)
        b = _sample(1, n=32, seed=5)
        r = compare_edge_pair(a, b)
        assert isinstance(r, EdgeCompResult)


# ─── TestBatchCompareEdges ────────────────────────────────────────────────────

class TestBatchCompareEdges:
    def test_pair_count(self):
        samples = [_sample(i, n=8, seed=i) for i in range(4)]
        results = batch_compare_edges(samples)
        # C(4,2) = 6
        assert len(results) == 6

    def test_all_scores_valid(self):
        samples = [_sample(i, n=8, seed=i) for i in range(3)]
        results = batch_compare_edges(samples)
        for r in results:
            assert 0.0 <= r.total_score <= 1.0

    def test_empty_list(self):
        assert batch_compare_edges([]) == []

    def test_single_sample(self):
        assert batch_compare_edges([_sample(0)]) == []

    def test_pairs_ordered(self):
        samples = [_sample(i, n=8, seed=i) for i in range(3)]
        results = batch_compare_edges(samples)
        pairs = [r.pair for r in results]
        assert (0, 1) in pairs
        assert (0, 2) in pairs
        assert (1, 2) in pairs

    def test_custom_cfg(self):
        cfg = EdgeCompConfig(n_samples=8, use_gradient=False, use_texture=False)
        samples = [_sample(i, n=8, seed=i) for i in range(3)]
        results = batch_compare_edges(samples, cfg)
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.total_score <= 1.0
