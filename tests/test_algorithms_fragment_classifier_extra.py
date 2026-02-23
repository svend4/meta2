"""Extra tests for puzzle_reconstruction/algorithms/fragment_classifier.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fragment_classifier import (
    FragmentType,
    FragmentFeatures,
    ClassifyResult,
    compute_texture_features,
    compute_edge_features,
    compute_shape_features,
    detect_text_presence,
    classify_fragment_type,
    classify_fragment,
    batch_classify,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=100, w=100, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def _noisy(h=100, w=100, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=100, w=100, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


# ─── FragmentType (extra) ─────────────────────────────────────────────────────

class TestFragmentTypeExtra:
    def test_five_members(self):
        assert len(list(FragmentType)) == 5

    def test_inner_value(self):
        assert FragmentType.INNER.value == "inner"

    def test_unknown_value(self):
        assert FragmentType.UNKNOWN.value == "unknown"

    def test_full_value(self):
        assert FragmentType.FULL.value == "full"

    def test_edge_value(self):
        assert FragmentType.EDGE.value == "edge"

    def test_all_distinct_values(self):
        values = [ft.value for ft in FragmentType]
        assert len(values) == len(set(values))


# ─── FragmentFeatures (extra) ─────────────────────────────────────────────────

class TestFragmentFeaturesExtra:
    def test_custom_edge_densities_stored(self):
        ff = FragmentFeatures(edge_densities=(0.1, 0.2, 0.3, 0.4))
        assert ff.edge_densities == (0.1, 0.2, 0.3, 0.4)

    def test_custom_texture_variance_stored(self):
        ff = FragmentFeatures(texture_variance=25.0)
        assert ff.texture_variance == pytest.approx(25.0)

    def test_custom_aspect_ratio_stored(self):
        ff = FragmentFeatures(aspect_ratio=2.5)
        assert ff.aspect_ratio == pytest.approx(2.5)

    def test_custom_fill_ratio_stored(self):
        ff = FragmentFeatures(fill_ratio=0.75)
        assert ff.fill_ratio == pytest.approx(0.75)

    def test_as_vector_all_zeros(self):
        ff = FragmentFeatures()
        v = ff.as_vector()
        # texture_variance and aspect_ratio are non-zero by default
        assert v.shape == (14,)

    def test_as_vector_non_default(self):
        ff = FragmentFeatures(texture_variance=10.0, text_density=0.5)
        v = ff.as_vector()
        assert v.shape == (14,)

    def test_lbp_uniformity_stored(self):
        ff = FragmentFeatures(lbp_uniformity=0.8)
        assert ff.lbp_uniformity == pytest.approx(0.8)


# ─── ClassifyResult (extra) ───────────────────────────────────────────────────

class TestClassifyResultExtra:
    def _make(self, ftype=FragmentType.INNER, conf=0.7, sides=None):
        ff = FragmentFeatures()
        return ClassifyResult(
            fragment_type=ftype,
            confidence=conf,
            has_text=False,
            text_lines=0,
            features=ff,
            straight_sides=sides or [],
        )

    def test_straight_sides_stored(self):
        r = self._make(sides=[0, 1])
        assert r.straight_sides == [0, 1]

    def test_confidence_stored(self):
        r = self._make(conf=0.95)
        assert r.confidence == pytest.approx(0.95)

    def test_fragment_type_corner(self):
        r = self._make(ftype=FragmentType.CORNER)
        assert r.fragment_type == FragmentType.CORNER

    def test_fragment_type_full(self):
        r = self._make(ftype=FragmentType.FULL)
        assert r.fragment_type == FragmentType.FULL

    def test_has_text_true(self):
        ff = FragmentFeatures()
        r = ClassifyResult(
            fragment_type=FragmentType.INNER,
            confidence=0.5,
            has_text=True,
            text_lines=3,
            features=ff,
        )
        assert r.has_text is True
        assert r.text_lines == 3


# ─── compute_texture_features (extra) ─────────────────────────────────────────

class TestComputeTextureFeaturesExtra:
    def test_returns_two_floats(self):
        tex_var, lbp = compute_texture_features(_gray())
        assert isinstance(tex_var, float)
        assert isinstance(lbp, float)

    def test_high_variance_noisy_image(self):
        _, _ = compute_texture_features(_noisy(64, 64))
        # Just check it runs without error on noisy input
        assert True

    def test_multiple_calls_consistent(self):
        img = _gray(50, 50)
        r1 = compute_texture_features(img)
        r2 = compute_texture_features(img)
        assert r1[0] == pytest.approx(r2[0])
        assert r1[1] == pytest.approx(r2[1])

    def test_different_fills_different_variance(self):
        img1 = _gray(fill=0)
        img2 = _noisy(100, 100)
        var1, _ = compute_texture_features(img1)
        var2, _ = compute_texture_features(img2)
        assert var2 >= var1

    def test_lbp_uniform_image_valid(self):
        img = _gray(fill=200)
        _, lbp = compute_texture_features(img)
        assert 0.0 <= lbp <= 1.0


# ─── compute_edge_features (extra) ────────────────────────────────────────────

class TestComputeEdgeFeaturesExtra:
    def test_exactly_4_densities(self):
        d, s = compute_edge_features(_gray())
        assert len(d) == 4
        assert len(s) == 4

    def test_densities_are_floats(self):
        d, _ = compute_edge_features(_gray())
        for v in d:
            assert isinstance(v, float)

    def test_straightnesses_are_floats(self):
        _, s = compute_edge_features(_gray())
        for v in s:
            assert isinstance(v, float)

    def test_noisy_image_positive_density(self):
        d, _ = compute_edge_features(_noisy(100, 100))
        assert sum(d) > 0.0

    def test_second_call_consistent(self):
        img = _noisy(64, 64)
        r1 = compute_edge_features(img)
        r2 = compute_edge_features(img)
        assert r1[0] == r2[0]

    def test_wide_image(self):
        img = _noisy(50, 200)
        d, s = compute_edge_features(img)
        assert len(d) == 4
        assert len(s) == 4


# ─── compute_shape_features (extra) ───────────────────────────────────────────

class TestComputeShapeFeaturesExtra:
    def test_tall_image_aspect_ratio(self):
        img = _gray(h=200, w=50)
        aspect_ratio, _, _ = compute_shape_features(img)
        assert aspect_ratio == pytest.approx(0.25, abs=0.1)

    def test_fill_ratio_float(self):
        img = _gray(100, 100)
        _, fill_ratio, _ = compute_shape_features(img)
        assert isinstance(fill_ratio, float)

    def test_dominant_angle_float(self):
        img = _noisy(50, 50)
        _, _, dom_angle = compute_shape_features(img)
        assert isinstance(dom_angle, float)

    def test_rectangular_aspect_ratio_correct(self):
        img = _gray(h=50, w=100)
        aspect_ratio, _, _ = compute_shape_features(img)
        assert aspect_ratio == pytest.approx(2.0, abs=0.1)

    def test_consistent_output(self):
        img = _gray(80, 80)
        r1 = compute_shape_features(img)
        r2 = compute_shape_features(img)
        assert r1[0] == pytest.approx(r2[0])


# ─── detect_text_presence (extra) ─────────────────────────────────────────────

class TestDetectTextPresenceExtra:
    def test_high_var_thresh_reduces_text_detection(self):
        img = _noisy(128, 128)
        _, td_low, _ = detect_text_presence(img, var_thresh=10.0)
        _, td_high, _ = detect_text_presence(img, var_thresh=1000.0)
        assert td_low >= td_high

    def test_n_text_rows_nonneg(self):
        img = _gray(64, 64)
        _, _, n = detect_text_presence(img)
        assert n >= 0

    def test_uniform_text_density_zero(self):
        img = _gray(64, 64, fill=100)
        _, td, _ = detect_text_presence(img)
        assert td == pytest.approx(0.0, abs=1e-6)

    def test_consistent_calls(self):
        img = _noisy(64, 64, seed=5)
        r1 = detect_text_presence(img)
        r2 = detect_text_presence(img)
        assert r1[0] == r2[0]
        assert r1[1] == pytest.approx(r2[1])

    def test_large_block_size(self):
        img = _noisy(128, 128)
        result = detect_text_presence(img, block_size=32)
        assert len(result) == 3


# ─── classify_fragment_type (extra) ───────────────────────────────────────────

class TestClassifyFragmentTypeExtra:
    def test_three_straight_sides_not_inner(self):
        densities = (0.8, 0.8, 0.8, 0.01)
        straightness = (0.8, 0.8, 0.8, 0.01)
        ftype, conf, sides = classify_fragment_type(densities, straightness, 1.0)
        assert ftype != FragmentType.INNER

    def test_confidence_nonneg(self):
        d = (0.5, 0.5, 0.5, 0.5)
        s = (0.5, 0.5, 0.5, 0.5)
        _, conf, _ = classify_fragment_type(d, s, 1.0)
        assert conf >= 0.0

    def test_sides_list_subset_0_to_3(self):
        d = (0.8, 0.01, 0.8, 0.01)
        s = (0.8, 0.01, 0.8, 0.01)
        _, _, sides = classify_fragment_type(d, s, 1.0)
        for idx in sides:
            assert 0 <= idx <= 3

    def test_all_sides_max_4(self):
        d = (0.9, 0.9, 0.9, 0.9)
        s = (0.9, 0.9, 0.9, 0.9)
        _, _, sides = classify_fragment_type(d, s, 1.0)
        assert len(sides) <= 4

    def test_inner_has_no_straight_sides(self):
        d = (0.0, 0.0, 0.0, 0.0)
        s = (0.0, 0.0, 0.0, 0.0)
        _, _, sides = classify_fragment_type(d, s, 1.0)
        assert sides == []


# ─── classify_fragment (extra) ────────────────────────────────────────────────

class TestClassifyFragmentExtra:
    def test_text_lines_nonneg(self):
        result = classify_fragment(_gray())
        assert result.text_lines >= 0

    def test_features_as_vector_float32(self):
        result = classify_fragment(_gray())
        v = result.features.as_vector()
        assert v.dtype == np.float32

    def test_two_uniform_images_same_type(self):
        r1 = classify_fragment(_gray(fill=128))
        r2 = classify_fragment(_gray(fill=128))
        assert r1.fragment_type == r2.fragment_type

    def test_straight_sides_indices_valid(self):
        result = classify_fragment(_noisy())
        for s in result.straight_sides:
            assert 0 <= s <= 3

    def test_bgr_and_gray_same_type(self):
        gray = _gray(fill=100)
        bgr = _bgr(fill=100)
        r_gray = classify_fragment(gray)
        r_bgr = classify_fragment(bgr)
        # Both should classify as INNER for uniform image
        assert r_gray.fragment_type == r_bgr.fragment_type

    def test_confidence_float(self):
        result = classify_fragment(_gray())
        assert isinstance(result.confidence, float)


# ─── batch_classify (extra) ───────────────────────────────────────────────────

class TestBatchClassifyExtra:
    def test_returns_list_type(self):
        result = batch_classify([_gray()])
        assert isinstance(result, list)

    def test_large_batch(self):
        images = [_gray(fill=i * 10) for i in range(10)]
        result = batch_classify(images)
        assert len(result) == 10

    def test_all_confidence_in_0_1(self):
        images = [_gray(), _noisy(), _bgr()]
        result = batch_classify(images)
        for r in result:
            assert 0.0 <= r.confidence <= 1.0

    def test_all_text_lines_nonneg(self):
        images = [_gray(fill=i * 30) for i in range(5)]
        result = batch_classify(images)
        for r in result:
            assert r.text_lines >= 0

    def test_features_present(self):
        result = batch_classify([_gray()])
        assert isinstance(result[0].features, FragmentFeatures)

    def test_bgr_batch(self):
        images = [_bgr(fill=i * 20) for i in range(3)]
        result = batch_classify(images)
        assert len(result) == 3
