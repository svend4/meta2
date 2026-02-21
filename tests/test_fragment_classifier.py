"""
Тесты для puzzle_reconstruction/algorithms/fragment_classifier.py

Покрывает:
    FragmentType          — значения, str-Enum, membership
    FragmentFeatures      — as_vector (12 элементов, float32), значения по умолч.
    ClassifyResult        — repr, straight_sides по умолчанию, поля
    compute_texture_features — (float, float), lbp_uniformity ∈ [0,1]
    compute_edge_features    — (4-tuple, 4-tuple), значения ∈ [0,1]
    compute_shape_features   — aspect_ratio=w/h, fill_ratio ∈ [0,1],
                               dominant_angle ∈ [-90,90]
    detect_text_presence     — (bool, float, int), density ∈ [0,1],
                               пустой образ → has_text=False
    classify_fragment_type   — INNER при нулевых densities, FULL при всех
                               высоких, CORNER при 2 соседних, EDGE при 1
    classify_fragment        — возвращает ClassifyResult, принимает BGR/gray
    batch_classify           — список ClassifyResult
"""
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


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def gray_blank():
    return np.zeros((80, 80), dtype=np.uint8)


@pytest.fixture
def gray_textured():
    """Шахматная доска — хорошая текстура."""
    img = np.zeros((80, 80), dtype=np.uint8)
    for i in range(0, 80, 8):
        for j in range(0, 80, 8):
            if (i // 8 + j // 8) % 2 == 0:
                img[i:i + 8, j:j + 8] = 200
    return img


@pytest.fixture
def bgr_blank():
    return np.zeros((80, 80, 3), dtype=np.uint8)


@pytest.fixture
def gray_with_edges():
    """Изображение с чёткими краями по всем четырём сторонам."""
    img = np.zeros((80, 80), dtype=np.uint8)
    img[0:8, :]  = 255   # top edge strip — белый
    img[72:, :]  = 255   # bottom edge strip — белый
    img[:, 0:8]  = 255   # left edge strip
    img[:, 72:]  = 255   # right edge strip
    return img


@pytest.fixture
def features_default():
    return FragmentFeatures()


# ─── FragmentType ─────────────────────────────────────────────────────────────

class TestFragmentType:
    def test_all_values_defined(self):
        expected = {"corner", "edge", "inner", "full", "unknown"}
        actual = {ft.value for ft in FragmentType}
        assert expected == actual

    def test_is_str_enum(self):
        assert isinstance(FragmentType.CORNER, str)
        assert FragmentType.CORNER == "corner"

    def test_membership(self):
        assert "corner" in [ft.value for ft in FragmentType]
        assert "inner"  in [ft.value for ft in FragmentType]

    def test_comparison(self):
        assert FragmentType.CORNER != FragmentType.EDGE
        assert FragmentType.FULL   != FragmentType.INNER

    def test_str_value(self):
        assert FragmentType.UNKNOWN.value == "unknown"
        assert FragmentType.FULL.value    == "full"


# ─── FragmentFeatures ─────────────────────────────────────────────────────────

class TestFragmentFeatures:
    def test_as_vector_length(self, features_default):
        v = features_default.as_vector()
        assert len(v) == 12

    def test_as_vector_dtype(self, features_default):
        v = features_default.as_vector()
        assert v.dtype == np.float32

    def test_default_edge_densities(self, features_default):
        assert features_default.edge_densities == (0., 0., 0., 0.)

    def test_default_aspect_ratio(self, features_default):
        assert features_default.aspect_ratio == pytest.approx(1.0)

    def test_default_fill_ratio(self, features_default):
        assert features_default.fill_ratio == pytest.approx(1.0)

    def test_as_vector_contains_all_fields(self):
        ff = FragmentFeatures(
            edge_densities=(0.1, 0.2, 0.3, 0.4),
            edge_straightness=(0.5, 0.6, 0.7, 0.8),
            texture_variance=10.0,
            text_density=0.3,
            aspect_ratio=1.5,
            fill_ratio=0.9,
            dominant_angle=15.0,
            lbp_uniformity=0.7,
        )
        v = ff.as_vector()
        assert v[0]  == pytest.approx(0.1)
        assert v[3]  == pytest.approx(0.4)
        assert v[4]  == pytest.approx(0.5)
        assert v[7]  == pytest.approx(0.8)
        assert v[8]  == pytest.approx(10.0)
        assert v[9]  == pytest.approx(0.3)
        assert v[10] == pytest.approx(1.5)
        assert v[11] == pytest.approx(0.9)

    def test_as_vector_returns_new_array(self, features_default):
        v1 = features_default.as_vector()
        v2 = features_default.as_vector()
        assert v1 is not v2


# ─── ClassifyResult ───────────────────────────────────────────────────────────

class TestClassifyResult:
    def test_fields(self):
        res = ClassifyResult(
            fragment_type=FragmentType.CORNER,
            confidence=0.85,
            has_text=True,
            text_lines=3,
            features=FragmentFeatures(),
            straight_sides=[0, 3],
        )
        assert res.fragment_type == FragmentType.CORNER
        assert res.confidence == pytest.approx(0.85)
        assert res.has_text is True
        assert res.text_lines == 3
        assert res.straight_sides == [0, 3]

    def test_straight_sides_default_empty(self):
        res = ClassifyResult(
            fragment_type=FragmentType.INNER,
            confidence=0.7,
            has_text=False,
            text_lines=0,
            features=FragmentFeatures(),
        )
        assert res.straight_sides == []

    def test_repr_contains_type(self):
        res = ClassifyResult(
            fragment_type=FragmentType.EDGE,
            confidence=0.6,
            has_text=False,
            text_lines=0,
            features=FragmentFeatures(),
        )
        r = repr(res)
        assert "ClassifyResult" in r
        assert "edge" in r
        assert "conf=" in r
        assert "text=" in r


# ─── compute_texture_features ─────────────────────────────────────────────────

class TestComputeTextureFeatures:
    def test_returns_two_floats(self, gray_blank):
        result = compute_texture_features(gray_blank)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_blank_has_zero_variance(self, gray_blank):
        tex_var, _ = compute_texture_features(gray_blank)
        assert tex_var == pytest.approx(0.0)

    def test_textured_has_positive_variance(self, gray_textured):
        tex_var, _ = compute_texture_features(gray_textured)
        assert tex_var > 0.0

    def test_lbp_uniformity_in_range(self, gray_textured):
        _, lbp = compute_texture_features(gray_textured)
        assert 0.0 <= lbp <= 1.0

    def test_small_image_no_crash(self):
        img = np.array([[100, 200], [50, 150]], dtype=np.uint8)
        result = compute_texture_features(img)
        assert len(result) == 2

    def test_1x1_image(self):
        img = np.array([[128]], dtype=np.uint8)
        tex_var, lbp = compute_texture_features(img)
        assert isinstance(tex_var, float)
        assert lbp == pytest.approx(0.0)


# ─── compute_edge_features ────────────────────────────────────────────────────

class TestComputeEdgeFeatures:
    def test_returns_two_four_tuples(self, gray_blank):
        dens, straight = compute_edge_features(gray_blank)
        assert len(dens) == 4
        assert len(straight) == 4

    def test_densities_in_range(self, gray_textured):
        dens, _ = compute_edge_features(gray_textured)
        for d in dens:
            assert 0.0 <= d <= 1.0

    def test_straightness_in_range(self, gray_textured):
        _, straight = compute_edge_features(gray_textured)
        for s in straight:
            assert 0.0 <= s <= 1.0

    def test_blank_has_zero_density(self, gray_blank):
        dens, _ = compute_edge_features(gray_blank)
        assert all(d == pytest.approx(0.0) for d in dens)

    def test_blank_has_zero_straightness(self, gray_blank):
        _, straight = compute_edge_features(gray_blank)
        assert all(s == pytest.approx(0.0) for s in straight)

    def test_framed_image_has_high_density(self, gray_with_edges):
        dens, _ = compute_edge_features(gray_with_edges, canny_lo=20,
                                         canny_hi=80)
        # У изображения с рамкой должна быть ненулевая плотность хотя бы на 1 стороне
        assert max(dens) > 0.0

    def test_custom_thresholds(self, gray_textured):
        dens1, _ = compute_edge_features(gray_textured, canny_lo=10, canny_hi=30)
        dens2, _ = compute_edge_features(gray_textured, canny_lo=200, canny_hi=250)
        # Низкий порог → больше краёв
        assert sum(dens1) >= sum(dens2)


# ─── compute_shape_features ───────────────────────────────────────────────────

class TestComputeShapeFeatures:
    def test_returns_three_floats(self, gray_blank):
        result = compute_shape_features(gray_blank)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_aspect_ratio_square(self, gray_blank):
        asp, _, _ = compute_shape_features(gray_blank)
        assert asp == pytest.approx(1.0)

    def test_aspect_ratio_wide(self):
        img = np.zeros((40, 120), dtype=np.uint8)
        asp, _, _ = compute_shape_features(img)
        assert asp == pytest.approx(3.0)

    def test_fill_ratio_in_range(self, gray_textured):
        _, fill, _ = compute_shape_features(gray_textured)
        assert 0.0 <= fill <= 1.0

    def test_dominant_angle_in_range(self, gray_textured):
        _, _, angle = compute_shape_features(gray_textured)
        assert -90.0 <= angle <= 90.0

    def test_blank_fill_ratio_zero(self, gray_blank):
        _, fill, _ = compute_shape_features(gray_blank)
        assert fill == pytest.approx(0.0)

    def test_white_fill_ratio_one(self):
        img = np.full((40, 40), 255, dtype=np.uint8)
        _, fill, _ = compute_shape_features(img)
        assert fill == pytest.approx(1.0)


# ─── detect_text_presence ─────────────────────────────────────────────────────

class TestDetectTextPresence:
    def test_returns_three_elements(self, gray_blank):
        result = detect_text_presence(gray_blank)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_blank_no_text(self, gray_blank):
        has_text, density, n_rows = detect_text_presence(gray_blank)
        assert has_text is False
        assert density == pytest.approx(0.0)
        assert n_rows == 0

    def test_density_in_range(self, gray_textured):
        _, density, _ = detect_text_presence(gray_textured)
        assert 0.0 <= density <= 1.0

    def test_n_rows_nonnegative(self, gray_textured):
        _, _, n_rows = detect_text_presence(gray_textured)
        assert n_rows >= 0

    def test_high_variance_image_may_have_text(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (80, 80), dtype=np.uint8)
        has_text, density, _ = detect_text_presence(img, var_thresh=10.0)
        assert has_text is True
        assert density > 0.0

    def test_constant_blocks_no_text(self):
        img = np.full((64, 64), 100, dtype=np.uint8)
        has_text, density, n_rows = detect_text_presence(img, var_thresh=100.0)
        assert has_text is False

    def test_high_threshold_reduces_text_blocks(self, gray_textured):
        _, d_low,  _ = detect_text_presence(gray_textured, var_thresh=1.0)
        _, d_high, _ = detect_text_presence(gray_textured, var_thresh=10000.0)
        assert d_low >= d_high


# ─── classify_fragment_type ───────────────────────────────────────────────────

class TestClassifyFragmentType:
    def test_inner_all_zero_densities(self):
        densities   = (0.0, 0.0, 0.0, 0.0)
        straightness = (0.0, 0.0, 0.0, 0.0)
        ftype, conf, straight = classify_fragment_type(densities, straightness, 1.0)
        assert ftype == FragmentType.INNER
        assert straight == []
        assert conf >= 0.0

    def test_corner_two_adjacent_straight_sides(self):
        # Стороны 0 и 1 (top и right) прямые и с высокой плотностью
        densities    = (0.5, 0.5, 0.0, 0.0)
        straightness = (0.9, 0.9, 0.0, 0.0)
        ftype, conf, straight = classify_fragment_type(densities, straightness, 1.0)
        assert ftype == FragmentType.CORNER
        assert 0 in straight
        assert 1 in straight
        assert conf > 0.0

    def test_edge_one_straight_side(self):
        densities    = (0.5, 0.0, 0.0, 0.0)
        straightness = (0.9, 0.0, 0.0, 0.0)
        ftype, conf, straight = classify_fragment_type(densities, straightness, 1.0)
        assert ftype == FragmentType.EDGE
        assert 0 in straight

    def test_full_four_straight_sides(self):
        densities    = (0.6, 0.6, 0.6, 0.6)
        straightness = (0.8, 0.8, 0.8, 0.8)
        ftype, conf, _ = classify_fragment_type(densities, straightness, 1.0)
        assert ftype == FragmentType.FULL
        assert conf >= 0.0

    def test_confidence_in_range(self):
        densities    = (0.4, 0.4, 0.0, 0.0)
        straightness = (0.8, 0.8, 0.0, 0.0)
        _, conf, _ = classify_fragment_type(densities, straightness, 1.0)
        assert 0.0 <= conf <= 1.0

    def test_returns_three_elements(self):
        result = classify_fragment_type((0.,)*4, (0.,)*4, 1.0)
        assert len(result) == 3

    def test_corner_sides_0_3_are_adjacent(self):
        """Стороны 0 (top) и 3 (left) — тоже соседние (угол)."""
        densities    = (0.5, 0.0, 0.0, 0.5)
        straightness = (0.9, 0.0, 0.0, 0.9)
        ftype, _, _ = classify_fragment_type(densities, straightness, 1.0)
        assert ftype == FragmentType.CORNER


# ─── classify_fragment ────────────────────────────────────────────────────────

class TestClassifyFragment:
    def test_returns_classify_result(self, gray_blank):
        res = classify_fragment(gray_blank)
        assert isinstance(res, ClassifyResult)

    def test_gray_input(self, gray_blank):
        res = classify_fragment(gray_blank)
        assert res.fragment_type in list(FragmentType)

    def test_bgr_input(self, bgr_blank):
        res = classify_fragment(bgr_blank)
        assert isinstance(res, ClassifyResult)

    def test_confidence_in_range(self, gray_textured):
        res = classify_fragment(gray_textured)
        assert 0.0 <= res.confidence <= 1.0

    def test_has_text_is_bool(self, gray_blank):
        res = classify_fragment(gray_blank)
        assert isinstance(res.has_text, bool)

    def test_text_lines_nonnegative(self, gray_blank):
        res = classify_fragment(gray_blank)
        assert res.text_lines >= 0

    def test_features_type(self, gray_blank):
        res = classify_fragment(gray_blank)
        assert isinstance(res.features, FragmentFeatures)

    def test_features_vector_length(self, gray_blank):
        res = classify_fragment(gray_blank)
        assert len(res.features.as_vector()) == 12

    def test_straight_sides_subset_of_0_3(self, gray_blank):
        res = classify_fragment(gray_blank)
        assert all(s in range(4) for s in res.straight_sides)

    def test_blank_classified_as_inner_or_unknown(self, gray_blank):
        res = classify_fragment(gray_blank)
        assert res.fragment_type in (FragmentType.INNER, FragmentType.UNKNOWN)

    def test_textured_no_crash(self, gray_textured):
        res = classify_fragment(gray_textured)
        assert isinstance(res, ClassifyResult)


# ─── batch_classify ───────────────────────────────────────────────────────────

class TestBatchClassify:
    def test_returns_list(self, gray_blank):
        results = batch_classify([gray_blank, gray_blank])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_is_classify_result(self, gray_blank):
        results = batch_classify([gray_blank])
        assert isinstance(results[0], ClassifyResult)

    def test_empty_list(self):
        assert batch_classify([]) == []

    def test_mixed_types(self, gray_blank, bgr_blank):
        results = batch_classify([gray_blank, bgr_blank])
        assert len(results) == 2
        for res in results:
            assert isinstance(res, ClassifyResult)

    def test_kwargs_forwarded(self, gray_blank):
        results = batch_classify([gray_blank], var_thresh=50.0)
        assert isinstance(results[0], ClassifyResult)

    def test_length_matches_input(self, gray_textured, gray_blank):
        imgs = [gray_textured] * 5 + [gray_blank] * 3
        results = batch_classify(imgs)
        assert len(results) == 8
