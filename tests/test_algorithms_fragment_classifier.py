"""Тесты для puzzle_reconstruction/algorithms/fragment_classifier.py."""
import pytest
import numpy as np

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

def make_gray(h=100, w=100, fill=128, dtype=np.uint8):
    return np.full((h, w), fill, dtype=dtype)


def make_noisy_gray(h=100, w=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def make_bgr(h=100, w=100, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


# ─── FragmentType ─────────────────────────────────────────────────────────────

class TestFragmentType:
    def test_values(self):
        assert FragmentType.CORNER.value == "corner"
        assert FragmentType.EDGE.value == "edge"
        assert FragmentType.INNER.value == "inner"
        assert FragmentType.FULL.value == "full"
        assert FragmentType.UNKNOWN.value == "unknown"

    def test_is_str(self):
        assert isinstance(FragmentType.CORNER, str)

    def test_members(self):
        members = {ft.name for ft in FragmentType}
        assert {"CORNER", "EDGE", "INNER", "FULL", "UNKNOWN"}.issubset(members)


# ─── FragmentFeatures ─────────────────────────────────────────────────────────

class TestFragmentFeatures:
    def test_default_values(self):
        ff = FragmentFeatures()
        assert ff.edge_densities == (0., 0., 0., 0.)
        assert ff.edge_straightness == (0., 0., 0., 0.)
        assert ff.texture_variance == 0.0
        assert ff.text_density == 0.0
        assert ff.aspect_ratio == 1.0
        assert ff.fill_ratio == 1.0
        assert ff.dominant_angle == 0.0
        assert ff.lbp_uniformity == 0.0

    def test_as_vector_shape(self):
        ff = FragmentFeatures()
        v = ff.as_vector()
        assert v.shape == (14,)

    def test_as_vector_dtype_float32(self):
        ff = FragmentFeatures()
        v = ff.as_vector()
        assert v.dtype == np.float32

    def test_as_vector_values(self):
        ff = FragmentFeatures(
            edge_densities=(0.1, 0.2, 0.3, 0.4),
            edge_straightness=(0.5, 0.6, 0.7, 0.8),
            texture_variance=10.0,
            text_density=0.3,
            aspect_ratio=1.5,
            fill_ratio=0.9,
            dominant_angle=45.0,
            lbp_uniformity=0.7,
        )
        v = ff.as_vector()
        assert v[0] == pytest.approx(0.1, abs=1e-6)
        assert v[4] == pytest.approx(0.5, abs=1e-6)
        assert v[8] == pytest.approx(10.0, abs=1e-4)
        assert v[10] == pytest.approx(1.5, abs=1e-6)


# ─── ClassifyResult ───────────────────────────────────────────────────────────

class TestClassifyResult:
    def test_creation(self):
        ff = FragmentFeatures()
        cr = ClassifyResult(
            fragment_type=FragmentType.INNER,
            confidence=0.7,
            has_text=False,
            text_lines=0,
            features=ff,
            straight_sides=[],
        )
        assert cr.fragment_type == FragmentType.INNER
        assert cr.confidence == pytest.approx(0.7)
        assert cr.has_text is False
        assert cr.text_lines == 0
        assert cr.straight_sides == []

    def test_default_straight_sides(self):
        ff = FragmentFeatures()
        cr = ClassifyResult(
            fragment_type=FragmentType.EDGE,
            confidence=0.5,
            has_text=False,
            text_lines=0,
            features=ff,
        )
        assert cr.straight_sides == []


# ─── compute_texture_features ─────────────────────────────────────────────────

class TestComputeTextureFeatures:
    def test_returns_tuple_of_two(self):
        gray = make_gray()
        result = compute_texture_features(gray)
        assert len(result) == 2

    def test_texture_variance_nonnegative(self):
        gray = make_gray()
        tex_var, _ = compute_texture_features(gray)
        assert tex_var >= 0.0

    def test_lbp_uniformity_in_0_1(self):
        gray = make_noisy_gray()
        _, lbp = compute_texture_features(gray)
        assert 0.0 <= lbp <= 1.0

    def test_uniform_image_low_variance(self):
        gray = make_gray(fill=200)
        tex_var, _ = compute_texture_features(gray)
        assert tex_var == pytest.approx(0.0, abs=1e-4)

    def test_noisy_image_higher_variance(self):
        flat = make_gray(fill=128)
        noisy = make_noisy_gray()
        flat_var, _ = compute_texture_features(flat)
        noisy_var, _ = compute_texture_features(noisy)
        assert noisy_var > flat_var

    def test_small_image_lbp(self):
        # h<3 or w<3 → lbp_uniformity=0
        gray = np.array([[100, 200]], dtype=np.uint8)  # 1x2
        _, lbp = compute_texture_features(gray)
        assert lbp == pytest.approx(0.0)

    def test_works_with_various_sizes(self):
        for sz in [8, 32, 64, 128]:
            gray = make_noisy_gray(h=sz, w=sz)
            tex_var, lbp = compute_texture_features(gray)
            assert tex_var >= 0.0
            assert 0.0 <= lbp <= 1.0


# ─── compute_edge_features ────────────────────────────────────────────────────

class TestComputeEdgeFeatures:
    def test_returns_two_4_tuples(self):
        gray = make_gray()
        densities, straightnesses = compute_edge_features(gray)
        assert len(densities) == 4
        assert len(straightnesses) == 4

    def test_densities_nonnegative(self):
        gray = make_noisy_gray()
        densities, _ = compute_edge_features(gray)
        for d in densities:
            assert d >= 0.0

    def test_straightnesses_in_0_1(self):
        gray = make_noisy_gray()
        _, straightnesses = compute_edge_features(gray)
        for s in straightnesses:
            assert 0.0 <= s <= 1.0

    def test_uniform_image_zero_densities(self):
        gray = make_gray(fill=150)
        densities, _ = compute_edge_features(gray)
        for d in densities:
            assert d == pytest.approx(0.0, abs=1e-6)

    def test_custom_canny_thresholds(self):
        gray = make_noisy_gray()
        d1, _ = compute_edge_features(gray, canny_lo=10, canny_hi=50)
        d2, _ = compute_edge_features(gray, canny_lo=100, canny_hi=200)
        # Lower threshold → more edges
        assert sum(d1) >= sum(d2)


# ─── compute_shape_features ───────────────────────────────────────────────────

class TestComputeShapeFeatures:
    def test_returns_3_values(self):
        gray = make_gray()
        result = compute_shape_features(gray)
        assert len(result) == 3

    def test_aspect_ratio_square(self):
        gray = make_gray(h=100, w=100)
        aspect_ratio, _, _ = compute_shape_features(gray)
        assert aspect_ratio == pytest.approx(1.0, abs=0.01)

    def test_aspect_ratio_wide(self):
        gray = make_gray(h=50, w=200)
        aspect_ratio, _, _ = compute_shape_features(gray)
        assert aspect_ratio == pytest.approx(4.0, abs=0.1)

    def test_fill_ratio_in_0_1(self):
        gray = make_gray()
        _, fill_ratio, _ = compute_shape_features(gray)
        assert 0.0 <= fill_ratio <= 1.0

    def test_dominant_angle_in_range(self):
        gray = make_noisy_gray()
        _, _, dom_angle = compute_shape_features(gray)
        assert -90.0 <= dom_angle <= 90.0

    def test_black_image_full_fill(self):
        gray = np.zeros((50, 50), dtype=np.uint8)
        _, fill_ratio, _ = compute_shape_features(gray)
        # All pixels are 0, Otsu threshold may produce all-black binary → fill_ratio=0 or all non-zero
        assert 0.0 <= fill_ratio <= 1.0


# ─── detect_text_presence ─────────────────────────────────────────────────────

class TestDetectTextPresence:
    def test_returns_3_values(self):
        gray = make_gray()
        result = detect_text_presence(gray)
        assert len(result) == 3

    def test_has_text_is_bool(self):
        gray = make_gray()
        has_text, _, _ = detect_text_presence(gray)
        assert isinstance(has_text, bool)

    def test_text_density_in_0_1(self):
        gray = make_gray()
        _, text_density, _ = detect_text_presence(gray)
        assert 0.0 <= text_density <= 1.0

    def test_n_text_rows_nonnegative(self):
        gray = make_gray()
        _, _, n_rows = detect_text_presence(gray)
        assert n_rows >= 0

    def test_uniform_image_no_text(self):
        gray = make_gray(fill=128)
        has_text, text_density, n_rows = detect_text_presence(gray)
        assert not has_text
        assert text_density == pytest.approx(0.0)

    def test_has_text_if_density_above_005(self):
        # Noisy image should have high variance blocks
        gray = make_noisy_gray(h=128, w=128)
        has_text, text_density, _ = detect_text_presence(gray, var_thresh=10.0)
        if text_density > 0.05:
            assert has_text
        else:
            assert not has_text

    def test_custom_block_size(self):
        gray = make_noisy_gray()
        result = detect_text_presence(gray, block_size=8)
        assert len(result) == 3

    def test_high_variance_image_has_text(self):
        # Checkerboard pattern has very high local variance
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[::2, ::2] = 255
        gray[1::2, 1::2] = 255
        has_text, text_density, _ = detect_text_presence(gray, var_thresh=100.0)
        assert text_density > 0.05
        assert has_text


# ─── classify_fragment_type ───────────────────────────────────────────────────

class TestClassifyFragmentType:
    def test_returns_3_tuple(self):
        densities = (0.5, 0.5, 0.5, 0.5)
        straightness = (0.8, 0.8, 0.8, 0.8)
        result = classify_fragment_type(densities, straightness, 1.0)
        assert len(result) == 3

    def test_four_straight_sides_full(self):
        # d*s > 0.30 for all 4 sides
        densities = (0.8, 0.8, 0.8, 0.8)
        straightness = (0.8, 0.8, 0.8, 0.8)
        ftype, conf, sides = classify_fragment_type(densities, straightness, 1.0)
        assert ftype == FragmentType.FULL
        assert len(sides) == 4

    def test_zero_straight_sides_inner(self):
        # d*s < 0.05 for all sides
        densities = (0.01, 0.01, 0.01, 0.01)
        straightness = (0.01, 0.01, 0.01, 0.01)
        ftype, conf, sides = classify_fragment_type(densities, straightness, 1.0)
        assert ftype == FragmentType.INNER
        assert sides == []

    def test_one_straight_side_edge(self):
        # Only top side is straight
        densities = (0.8, 0.01, 0.01, 0.01)
        straightness = (0.8, 0.01, 0.01, 0.01)
        ftype, conf, sides = classify_fragment_type(densities, straightness, 1.0)
        assert ftype == FragmentType.EDGE
        assert len(sides) == 1

    def test_two_adjacent_straight_sides_corner(self):
        # Top (0) and right (1) are adjacent
        densities = (0.8, 0.8, 0.01, 0.01)
        straightness = (0.8, 0.8, 0.01, 0.01)
        ftype, conf, sides = classify_fragment_type(densities, straightness, 1.0)
        assert ftype == FragmentType.CORNER
        assert len(sides) == 2

    def test_two_opposite_sides_unknown(self):
        # Top (0) and bottom (2) are opposite → UNKNOWN
        densities = (0.8, 0.01, 0.8, 0.01)
        straightness = (0.8, 0.01, 0.8, 0.01)
        ftype, conf, sides = classify_fragment_type(densities, straightness, 1.0)
        assert ftype == FragmentType.UNKNOWN

    def test_confidence_in_0_1(self):
        densities = (0.5, 0.5, 0.5, 0.5)
        straightness = (0.7, 0.7, 0.7, 0.7)
        _, conf, _ = classify_fragment_type(densities, straightness, 1.0)
        assert 0.0 <= conf <= 1.0

    def test_straight_sides_indices(self):
        # Only right (1) and bottom (2) straight
        densities = (0.01, 0.8, 0.8, 0.01)
        straightness = (0.01, 0.8, 0.8, 0.01)
        ftype, conf, sides = classify_fragment_type(densities, straightness, 1.0)
        assert ftype == FragmentType.CORNER
        assert 1 in sides
        assert 2 in sides

    def test_top_left_adjacent_corner(self):
        # Top (0) and left (3): indices 0 and 3 are adjacent (wrap-around)
        densities = (0.8, 0.01, 0.01, 0.8)
        straightness = (0.8, 0.01, 0.01, 0.8)
        ftype, conf, sides = classify_fragment_type(densities, straightness, 1.0)
        assert ftype == FragmentType.CORNER


# ─── classify_fragment ────────────────────────────────────────────────────────

class TestClassifyFragment:
    def test_returns_classify_result(self):
        gray = make_gray()
        result = classify_fragment(gray)
        assert isinstance(result, ClassifyResult)

    def test_accepts_bgr(self):
        bgr = make_bgr()
        result = classify_fragment(bgr)
        assert isinstance(result, ClassifyResult)

    def test_result_fragment_type(self):
        gray = make_gray()
        result = classify_fragment(gray)
        assert isinstance(result.fragment_type, FragmentType)

    def test_result_confidence_in_0_1(self):
        gray = make_gray()
        result = classify_fragment(gray)
        assert 0.0 <= result.confidence <= 1.0

    def test_result_has_text_bool(self):
        gray = make_gray()
        result = classify_fragment(gray)
        assert isinstance(result.has_text, bool)

    def test_result_features_is_fragment_features(self):
        gray = make_gray()
        result = classify_fragment(gray)
        assert isinstance(result.features, FragmentFeatures)

    def test_result_straight_sides_list(self):
        gray = make_gray()
        result = classify_fragment(gray)
        assert isinstance(result.straight_sides, list)

    def test_uniform_gray_classified(self):
        gray = make_gray(fill=128, h=100, w=100)
        result = classify_fragment(gray)
        # Uniform image: no edges → INNER with high confidence
        assert result.fragment_type in (FragmentType.INNER, FragmentType.UNKNOWN)

    def test_noisy_image_classified(self):
        gray = make_noisy_gray(h=100, w=100)
        result = classify_fragment(gray)
        assert isinstance(result.fragment_type, FragmentType)

    def test_features_vector_shape(self):
        gray = make_gray()
        result = classify_fragment(gray)
        v = result.features.as_vector()
        assert v.shape == (14,)


# ─── batch_classify ───────────────────────────────────────────────────────────

class TestBatchClassify:
    def test_empty_list_returns_empty(self):
        result = batch_classify([])
        assert result == []

    def test_single_image(self):
        gray = make_gray()
        result = batch_classify([gray])
        assert len(result) == 1
        assert isinstance(result[0], ClassifyResult)

    def test_multiple_images(self):
        images = [make_gray(fill=f) for f in [64, 128, 192]]
        result = batch_classify(images)
        assert len(result) == 3
        for r in result:
            assert isinstance(r, ClassifyResult)

    def test_accepts_bgr(self):
        bgr = make_bgr()
        result = batch_classify([bgr])
        assert len(result) == 1

    def test_mixed_bgr_gray(self):
        images = [make_gray(), make_bgr(), make_noisy_gray()]
        result = batch_classify(images)
        assert len(result) == 3

    def test_custom_var_thresh(self):
        images = [make_noisy_gray(seed=i) for i in range(3)]
        result = batch_classify(images, var_thresh=50.0)
        assert len(result) == 3

    def test_returns_list_of_classify_results(self):
        images = [make_gray()] * 4
        result = batch_classify(images)
        for r in result:
            assert isinstance(r, ClassifyResult)
