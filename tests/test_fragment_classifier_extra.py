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


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=80, w=80, val=0):
    return np.full((h, w), val, dtype=np.uint8)


def _checker(h=80, w=80, cell=8):
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, cell):
        for j in range(0, w, cell):
            if (i // cell + j // cell) % 2 == 0:
                img[i:i + cell, j:j + cell] = 200
    return img


def _bgr(h=80, w=80):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _framed(h=80, w=80, border=8, val=255):
    img = np.zeros((h, w), dtype=np.uint8)
    img[:border, :] = val
    img[h - border:, :] = val
    img[:, :border] = val
    img[:, w - border:] = val
    return img


# ─── FragmentType (extra) ────────────────────────────────────────────────────

class TestFragmentTypeExtra:
    def test_five_values(self):
        assert len(list(FragmentType)) == 5

    def test_corner_value(self):
        assert FragmentType.CORNER.value == "corner"

    def test_edge_value(self):
        assert FragmentType.EDGE.value == "edge"

    def test_inner_value(self):
        assert FragmentType.INNER.value == "inner"

    def test_full_value(self):
        assert FragmentType.FULL.value == "full"

    def test_unknown_value(self):
        assert FragmentType.UNKNOWN.value == "unknown"

    def test_is_str_enum(self):
        assert isinstance(FragmentType.INNER, str)

    def test_ne_comparison(self):
        assert FragmentType.CORNER != FragmentType.INNER

    def test_eq_comparison(self):
        assert FragmentType.EDGE == FragmentType.EDGE

    def test_all_in_list(self):
        values = {ft.value for ft in FragmentType}
        assert "corner" in values
        assert "edge" in values
        assert "inner" in values
        assert "full" in values
        assert "unknown" in values


# ─── FragmentFeatures (extra) ────────────────────────────────────────────────

class TestFragmentFeaturesExtra:
    def test_default_edge_densities_zero(self):
        ff = FragmentFeatures()
        assert ff.edge_densities == (0., 0., 0., 0.)

    def test_default_aspect_ratio_one(self):
        assert FragmentFeatures().aspect_ratio == pytest.approx(1.0)

    def test_default_fill_ratio_one(self):
        assert FragmentFeatures().fill_ratio == pytest.approx(1.0)

    def test_as_vector_length_ge_12(self):
        assert len(FragmentFeatures().as_vector()) >= 12

    def test_as_vector_dtype_float32(self):
        assert FragmentFeatures().as_vector().dtype == np.float32

    def test_as_vector_two_calls_different_objects(self):
        ff = FragmentFeatures()
        v1 = ff.as_vector()
        v2 = ff.as_vector()
        assert v1 is not v2

    def test_custom_values_reflected(self):
        ff = FragmentFeatures(
            edge_densities=(0.1, 0.2, 0.3, 0.4),
            edge_straightness=(0.5, 0.6, 0.7, 0.8),
            texture_variance=5.0,
            text_density=0.25,
            aspect_ratio=2.0,
            fill_ratio=0.75,
            dominant_angle=30.0,
            lbp_uniformity=0.6,
        )
        v = ff.as_vector()
        assert v[0] == pytest.approx(0.1)
        assert v[3] == pytest.approx(0.4)
        assert v[4] == pytest.approx(0.5)
        assert v[8] == pytest.approx(5.0)
        assert v[10] == pytest.approx(2.0)

    def test_default_edge_straightness_zero(self):
        ff = FragmentFeatures()
        assert ff.edge_straightness == (0., 0., 0., 0.)


# ─── ClassifyResult (extra) ──────────────────────────────────────────────────

class TestClassifyResultExtra:
    def test_fragment_type_stored(self):
        r = ClassifyResult(fragment_type=FragmentType.EDGE, confidence=0.7,
                           has_text=False, text_lines=0, features=FragmentFeatures())
        assert r.fragment_type == FragmentType.EDGE

    def test_confidence_stored(self):
        r = ClassifyResult(fragment_type=FragmentType.CORNER, confidence=0.85,
                           has_text=True, text_lines=2, features=FragmentFeatures())
        assert r.confidence == pytest.approx(0.85)

    def test_has_text_bool(self):
        r = ClassifyResult(fragment_type=FragmentType.INNER, confidence=0.5,
                           has_text=True, text_lines=3, features=FragmentFeatures())
        assert r.has_text is True

    def test_text_lines_stored(self):
        r = ClassifyResult(fragment_type=FragmentType.INNER, confidence=0.5,
                           has_text=True, text_lines=5, features=FragmentFeatures())
        assert r.text_lines == 5

    def test_straight_sides_default_empty(self):
        r = ClassifyResult(fragment_type=FragmentType.INNER, confidence=0.5,
                           has_text=False, text_lines=0, features=FragmentFeatures())
        assert r.straight_sides == []

    def test_straight_sides_custom(self):
        r = ClassifyResult(fragment_type=FragmentType.CORNER, confidence=0.8,
                           has_text=False, text_lines=0, features=FragmentFeatures(),
                           straight_sides=[0, 1])
        assert 0 in r.straight_sides
        assert 1 in r.straight_sides

    def test_repr_has_classify_result(self):
        r = ClassifyResult(fragment_type=FragmentType.FULL, confidence=0.9,
                           has_text=False, text_lines=0, features=FragmentFeatures())
        assert "ClassifyResult" in repr(r)

    def test_repr_has_type_value(self):
        r = ClassifyResult(fragment_type=FragmentType.EDGE, confidence=0.6,
                           has_text=False, text_lines=0, features=FragmentFeatures())
        assert "edge" in repr(r)


# ─── compute_texture_features (extra) ────────────────────────────────────────

class TestComputeTextureFeaturesExtra:
    def test_returns_tuple_of_two(self):
        result = compute_texture_features(_gray())
        assert isinstance(result, tuple) and len(result) == 2

    def test_both_floats(self):
        tex_var, lbp = compute_texture_features(_checker())
        assert isinstance(tex_var, float)
        assert isinstance(lbp, float)

    def test_blank_variance_zero(self):
        tex_var, _ = compute_texture_features(_gray(val=0))
        assert tex_var == pytest.approx(0.0)

    def test_lbp_in_range(self):
        _, lbp = compute_texture_features(_checker())
        assert 0.0 <= lbp <= 1.0

    def test_checker_variance_positive(self):
        tex_var, _ = compute_texture_features(_checker())
        assert tex_var > 0.0

    def test_non_square_ok(self):
        img = np.zeros((16, 32), dtype=np.uint8)
        result = compute_texture_features(img)
        assert len(result) == 2

    def test_small_image_no_crash(self):
        img = np.array([[100, 200], [50, 150]], dtype=np.uint8)
        result = compute_texture_features(img)
        assert len(result) == 2


# ─── compute_edge_features (extra) ───────────────────────────────────────────

class TestComputeEdgeFeaturesExtra:
    def test_returns_two_four_tuples(self):
        dens, straight = compute_edge_features(_gray())
        assert len(dens) == 4
        assert len(straight) == 4

    def test_blank_densities_zero(self):
        dens, _ = compute_edge_features(_gray(val=0))
        assert all(d == pytest.approx(0.0) for d in dens)

    def test_densities_in_range(self):
        dens, _ = compute_edge_features(_checker())
        for d in dens:
            assert 0.0 <= d <= 1.0

    def test_straightness_in_range(self):
        _, straight = compute_edge_features(_checker())
        for s in straight:
            assert 0.0 <= s <= 1.0

    def test_framed_has_high_density(self):
        dens, _ = compute_edge_features(_framed(), canny_lo=20, canny_hi=80)
        assert max(dens) > 0.0

    def test_low_threshold_more_edges(self):
        img = _checker()
        dens_lo, _ = compute_edge_features(img, canny_lo=5, canny_hi=20)
        dens_hi, _ = compute_edge_features(img, canny_lo=200, canny_hi=250)
        assert sum(dens_lo) >= sum(dens_hi)

    def test_blank_straightness_zero(self):
        _, straight = compute_edge_features(_gray(val=0))
        assert all(s == pytest.approx(0.0) for s in straight)


# ─── compute_shape_features (extra) ──────────────────────────────────────────

class TestComputeShapeFeaturesExtra:
    def test_returns_three_floats(self):
        result = compute_shape_features(_gray())
        assert isinstance(result, tuple) and len(result) == 3
        for v in result:
            assert isinstance(v, float)

    def test_square_aspect_ratio_one(self):
        asp, _, _ = compute_shape_features(np.zeros((40, 40), dtype=np.uint8))
        assert asp == pytest.approx(1.0)

    def test_wide_aspect_ratio_three(self):
        asp, _, _ = compute_shape_features(np.zeros((40, 120), dtype=np.uint8))
        assert asp == pytest.approx(3.0)

    def test_fill_ratio_in_range(self):
        _, fill, _ = compute_shape_features(_checker())
        assert 0.0 <= fill <= 1.0

    def test_dominant_angle_in_range(self):
        _, _, angle = compute_shape_features(_checker())
        assert -90.0 <= angle <= 90.0

    def test_blank_fill_zero(self):
        _, fill, _ = compute_shape_features(_gray(val=0))
        assert fill == pytest.approx(0.0)

    def test_white_fill_one(self):
        _, fill, _ = compute_shape_features(np.full((40, 40), 255, dtype=np.uint8))
        assert fill == pytest.approx(1.0)


# ─── detect_text_presence (extra) ────────────────────────────────────────────

class TestDetectTextPresenceExtra:
    def test_returns_three_elements(self):
        result = detect_text_presence(_gray())
        assert isinstance(result, tuple) and len(result) == 3

    def test_blank_no_text(self):
        has_text, density, n_rows = detect_text_presence(_gray(val=0))
        assert has_text is False
        assert density == pytest.approx(0.0)
        assert n_rows == 0

    def test_density_in_range(self):
        _, density, _ = detect_text_presence(_checker())
        assert 0.0 <= density <= 1.0

    def test_n_rows_nonneg(self):
        _, _, n_rows = detect_text_presence(_checker())
        assert n_rows >= 0

    def test_high_var_text_detected(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (80, 80), dtype=np.uint8)
        has_text, density, _ = detect_text_presence(img, var_thresh=10.0)
        assert has_text is True
        assert density > 0.0

    def test_high_threshold_reduces_density(self):
        img = _checker()
        _, d_low, _ = detect_text_presence(img, var_thresh=1.0)
        _, d_high, _ = detect_text_presence(img, var_thresh=10000.0)
        assert d_low >= d_high

    def test_constant_no_text(self):
        img = np.full((64, 64), 100, dtype=np.uint8)
        has_text, _, _ = detect_text_presence(img, var_thresh=100.0)
        assert has_text is False


# ─── classify_fragment_type (extra) ──────────────────────────────────────────

class TestClassifyFragmentTypeExtra:
    def test_returns_three_elements(self):
        result = classify_fragment_type((0.,) * 4, (0.,) * 4, 1.0)
        assert len(result) == 3

    def test_all_zero_inner(self):
        ftype, _, straight = classify_fragment_type(
            (0., 0., 0., 0.), (0., 0., 0., 0.), 1.0)
        assert ftype == FragmentType.INNER
        assert straight == []

    def test_four_straight_sides_full(self):
        ftype, conf, _ = classify_fragment_type(
            (0.6, 0.6, 0.6, 0.6), (0.8, 0.8, 0.8, 0.8), 1.0)
        assert ftype == FragmentType.FULL
        assert conf >= 0.0

    def test_one_straight_side_edge(self):
        ftype, _, straight = classify_fragment_type(
            (0.5, 0., 0., 0.), (0.9, 0., 0., 0.), 1.0)
        assert ftype == FragmentType.EDGE
        assert 0 in straight

    def test_two_adjacent_corner_01(self):
        ftype, _, straight = classify_fragment_type(
            (0.5, 0.5, 0., 0.), (0.9, 0.9, 0., 0.), 1.0)
        assert ftype == FragmentType.CORNER
        assert 0 in straight
        assert 1 in straight

    def test_two_adjacent_corner_03(self):
        ftype, _, _ = classify_fragment_type(
            (0.5, 0., 0., 0.5), (0.9, 0., 0., 0.9), 1.0)
        assert ftype == FragmentType.CORNER

    def test_confidence_in_range(self):
        _, conf, _ = classify_fragment_type(
            (0.4, 0.4, 0., 0.), (0.8, 0.8, 0., 0.), 1.0)
        assert 0.0 <= conf <= 1.0

    def test_straight_sides_are_subset_of_0_3(self):
        _, _, straight = classify_fragment_type(
            (0.5, 0.5, 0.5, 0.5), (0.8, 0.8, 0.8, 0.8), 1.0)
        assert all(s in range(4) for s in straight)


# ─── classify_fragment (extra) ───────────────────────────────────────────────

class TestClassifyFragmentExtra:
    def test_returns_classify_result(self):
        assert isinstance(classify_fragment(_gray()), ClassifyResult)

    def test_bgr_input_ok(self):
        r = classify_fragment(_bgr())
        assert isinstance(r, ClassifyResult)

    def test_confidence_in_range(self):
        r = classify_fragment(_checker())
        assert 0.0 <= r.confidence <= 1.0

    def test_has_text_is_bool(self):
        r = classify_fragment(_gray())
        assert isinstance(r.has_text, bool)

    def test_text_lines_nonneg(self):
        r = classify_fragment(_gray())
        assert r.text_lines >= 0

    def test_features_type(self):
        r = classify_fragment(_gray())
        assert isinstance(r.features, FragmentFeatures)

    def test_features_vector_length_ge_12(self):
        r = classify_fragment(_gray())
        assert len(r.features.as_vector()) >= 12

    def test_straight_sides_subset_0_3(self):
        r = classify_fragment(_gray())
        assert all(s in range(4) for s in r.straight_sides)

    def test_blank_inner_or_unknown(self):
        r = classify_fragment(_gray(val=0))
        assert r.fragment_type in (FragmentType.INNER, FragmentType.UNKNOWN)

    def test_fragment_type_valid(self):
        r = classify_fragment(_checker())
        assert r.fragment_type in list(FragmentType)


# ─── batch_classify (extra) ──────────────────────────────────────────────────

class TestBatchClassifyExtra:
    def test_empty_returns_empty(self):
        assert batch_classify([]) == []

    def test_length_preserved(self):
        imgs = [_gray()] * 5
        assert len(batch_classify(imgs)) == 5

    def test_all_classify_results(self):
        for r in batch_classify([_gray(), _checker()]):
            assert isinstance(r, ClassifyResult)

    def test_mixed_types(self):
        results = batch_classify([_gray(), _bgr()])
        assert len(results) == 2

    def test_kwargs_forwarded(self):
        results = batch_classify([_gray()], var_thresh=50.0)
        assert isinstance(results[0], ClassifyResult)

    def test_single_image(self):
        results = batch_classify([_checker()])
        assert len(results) == 1

    def test_many_images(self):
        imgs = [_gray()] * 10
        assert len(batch_classify(imgs)) == 10
