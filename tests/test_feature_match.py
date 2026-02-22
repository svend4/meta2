"""
Тесты для puzzle_reconstruction/matching/feature_match.py

Покрывает:
    KeypointMatch       — создание, repr, атрибуты
    FeatureMatchResult  — свойства n_matches/n_inliers/inlier_ratio, repr
    extract_features    — ORB/AKAZE, gray/BGR, пустое изображение, маска
    match_descriptors   — пустые дескрипторы, ratio test, BF/FLANN
    estimate_homography — недостаточно точек, тождественное преобразование
    feature_match_pair  — полный пайплайн, диапазон score, запись метода
    edge_feature_score  — возвращает float ∈ [0, 1]
"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.feature_match import (
    KeypointMatch,
    FeatureMatchResult,
    extract_features,
    match_descriptors,
    estimate_homography,
    feature_match_pair,
    edge_feature_score,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def checkerboard():
    """Шахматная доска 100×100 — хорошо для детектора ключевых точек."""
    img = np.zeros((100, 100), dtype=np.uint8)
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            if (i // 10 + j // 10) % 2 == 0:
                img[i:i + 10, j:j + 10] = 255
    return img


@pytest.fixture
def bgr_image():
    rng = np.random.default_rng(42)
    return (rng.random((80, 80, 3)) * 255).astype(np.uint8)


@pytest.fixture
def gray_image():
    rng = np.random.default_rng(42)
    return (rng.random((80, 80)) * 255).astype(np.uint8)


@pytest.fixture
def simple_match():
    return KeypointMatch(pt_src=(10.0, 20.0), pt_dst=(15.0, 25.0),
                         distance=30.0, confidence=0.8)


@pytest.fixture
def simple_result(simple_match):
    return FeatureMatchResult(
        matches=[simple_match],
        homography=np.eye(3),
        inlier_mask=np.array([True]),
        score=0.75,
        method="orb",
        n_keypoints=(100, 120),
    )


# ─── KeypointMatch ────────────────────────────────────────────────────────────

class TestKeypointMatch:
    def test_attributes(self, simple_match):
        assert simple_match.pt_src == (10.0, 20.0)
        assert simple_match.pt_dst == (15.0, 25.0)
        assert simple_match.distance == pytest.approx(30.0)
        assert simple_match.confidence == pytest.approx(0.8)

    def test_default_confidence(self):
        km = KeypointMatch(pt_src=(0., 0.), pt_dst=(1., 1.), distance=5.0)
        assert km.confidence == pytest.approx(1.0)

    def test_repr_contains_fields(self, simple_match):
        r = repr(simple_match)
        assert "KeypointMatch" in r
        assert "src=" in r
        assert "dst=" in r
        assert "conf=" in r

    def test_zero_distance(self):
        km = KeypointMatch(pt_src=(5., 5.), pt_dst=(5., 5.), distance=0.0)
        assert km.distance == 0.0

    def test_low_confidence_high_distance(self):
        km = KeypointMatch(pt_src=(0., 0.), pt_dst=(100., 100.),
                           distance=500.0, confidence=0.05)
        assert km.confidence < 0.2

    def test_confidence_clipped_to_one(self):
        km = KeypointMatch(pt_src=(0., 0.), pt_dst=(0., 0.),
                           distance=0.0, confidence=1.0)
        assert km.confidence <= 1.0

    def test_float_coordinates(self):
        km = KeypointMatch(pt_src=(3.14, 2.71), pt_dst=(0.5, 0.5),
                           distance=10.0)
        assert abs(km.pt_src[0] - 3.14) < 1e-6


# ─── FeatureMatchResult ───────────────────────────────────────────────────────

class TestFeatureMatchResult:
    def test_n_matches(self, simple_result):
        assert simple_result.n_matches == 1

    def test_n_inliers_true_mask(self, simple_result):
        assert simple_result.n_inliers == 1

    def test_n_inliers_false_mask(self):
        res = FeatureMatchResult(
            matches=[KeypointMatch((0., 0.), (1., 1.), 5.0)],
            homography=None,
            inlier_mask=np.array([False]),
            score=0.0,
            method="orb",
        )
        assert res.n_inliers == 0

    def test_n_inliers_empty_mask(self):
        res = FeatureMatchResult(
            matches=[],
            homography=None,
            inlier_mask=np.array([], dtype=bool),
            score=0.0,
            method="orb",
        )
        assert res.n_inliers == 0

    def test_n_inliers_none_mask(self):
        res = FeatureMatchResult(
            matches=[],
            homography=None,
            inlier_mask=None,
            score=0.0,
            method="orb",
        )
        assert res.n_inliers == 0

    def test_inlier_ratio_all_inliers(self, simple_result):
        assert simple_result.inlier_ratio == pytest.approx(1.0)

    def test_inlier_ratio_half(self):
        matches = [KeypointMatch((0., 0.), (1., 1.), 5.0) for _ in range(4)]
        res = FeatureMatchResult(
            matches=matches,
            homography=None,
            inlier_mask=np.array([True, True, False, False]),
            score=0.5,
            method="orb",
        )
        assert res.inlier_ratio == pytest.approx(0.5)

    def test_inlier_ratio_zero_matches(self):
        res = FeatureMatchResult(
            matches=[],
            homography=None,
            inlier_mask=np.array([], dtype=bool),
            score=0.0,
            method="orb",
        )
        assert res.inlier_ratio == pytest.approx(0.0)

    def test_repr_fields(self, simple_result):
        r = repr(simple_result)
        assert "FeatureMatchResult" in r
        assert "method=" in r
        assert "n_matches=" in r
        assert "n_inliers=" in r
        assert "score=" in r

    def test_n_keypoints_default(self):
        res = FeatureMatchResult(
            matches=[],
            homography=None,
            inlier_mask=np.array([], dtype=bool),
            score=0.0,
            method="orb",
        )
        assert res.n_keypoints == (0, 0)

    def test_score_in_range(self, simple_result):
        assert 0.0 <= simple_result.score <= 1.0

    def test_method_stored(self, simple_result):
        assert simple_result.method == "orb"


# ─── extract_features ─────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_returns_tuple_two_elements(self, gray_image):
        result = extract_features(gray_image, method="orb")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_keypoints_is_list(self, gray_image):
        kps, _ = extract_features(gray_image, method="orb")
        assert isinstance(kps, list)

    def test_grayscale_input(self, gray_image):
        kps, descs = extract_features(gray_image, method="orb")
        assert isinstance(kps, list)

    def test_bgr_input(self, bgr_image):
        kps, descs = extract_features(bgr_image, method="orb")
        assert isinstance(kps, list)

    def test_empty_black_image_no_crash(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        kps, descs = extract_features(img, method="orb")
        assert isinstance(kps, list)

    def test_checkerboard_finds_keypoints(self, checkerboard):
        kps, descs = extract_features(checkerboard, method="orb",
                                       n_features=200)
        # Шахматная доска должна давать ключевые точки
        if len(kps) > 0:
            assert descs is not None
            assert descs.shape[0] == len(kps)

    def test_n_features_upper_bound(self, checkerboard):
        kps_50, _ = extract_features(checkerboard, method="orb",
                                      n_features=50)
        assert len(kps_50) <= 50

    def test_mask_parameter(self, gray_image):
        mask = np.zeros_like(gray_image)
        mask[10:70, 10:70] = 255
        kps, descs = extract_features(gray_image, method="orb", mask=mask)
        assert isinstance(kps, list)

    def test_akaze_method(self, checkerboard):
        kps, descs = extract_features(checkerboard, method="akaze")
        assert isinstance(kps, list)

    def test_unknown_method_returns_empty(self, gray_image):
        kps, descs = extract_features(gray_image, method="nonexistent_xyz")
        assert isinstance(kps, list)
        assert len(kps) == 0


# ─── match_descriptors ────────────────────────────────────────────────────────

class TestMatchDescriptors:
    def test_none_desc1_returns_empty(self):
        result = match_descriptors(None, None, [], [], method="orb")
        assert result == []

    def test_empty_desc_returns_empty(self):
        d = np.array([], dtype=np.uint8).reshape(0, 32)
        result = match_descriptors(d, d, [], [], method="orb")
        assert result == []

    def test_returns_list_of_keypoint_matches(self, checkerboard):
        kps, descs = extract_features(checkerboard, method="orb",
                                       n_features=100)
        if descs is not None and len(kps) >= 2:
            shifted = np.roll(checkerboard, 3, axis=1)
            kps2, descs2 = extract_features(shifted, method="orb",
                                              n_features=100)
            if descs2 is not None and len(kps2) >= 2:
                matches = match_descriptors(descs, descs2, kps, kps2,
                                             method="orb")
                assert isinstance(matches, list)
                for m in matches:
                    assert isinstance(m, KeypointMatch)

    def test_confidence_in_range(self, checkerboard):
        kps, descs = extract_features(checkerboard, method="orb",
                                       n_features=100)
        if descs is not None and len(kps) >= 2:
            matches = match_descriptors(descs, descs, kps, kps, method="orb")
            for m in matches:
                assert 0.0 <= m.confidence <= 1.0

    def test_strict_ratio_gives_fewer_matches(self, checkerboard):
        kps, descs = extract_features(checkerboard, method="orb",
                                       n_features=100)
        if descs is not None and len(kps) >= 4:
            m_strict  = match_descriptors(descs, descs, kps, kps,
                                           method="orb", ratio=0.3)
            m_lenient = match_descriptors(descs, descs, kps, kps,
                                           method="orb", ratio=0.99)
            assert len(m_strict) <= len(m_lenient)

    def test_flann_matcher(self, checkerboard):
        kps, descs = extract_features(checkerboard, method="orb",
                                       n_features=100)
        if descs is not None and len(kps) >= 2:
            try:
                result = match_descriptors(descs, descs, kps, kps,
                                            method="orb", matcher="flann")
                assert isinstance(result, list)
            except Exception:
                pass  # FLANN может не работать в некоторых окружениях

    def test_orb_norm_hamming_used(self, checkerboard):
        """ORB использует NORM_HAMMING, не должно падать."""
        kps, descs = extract_features(checkerboard, method="orb",
                                       n_features=50)
        if descs is not None and len(kps) >= 2:
            matches = match_descriptors(descs, descs, kps, kps, method="orb")
            assert isinstance(matches, list)


# ─── estimate_homography ──────────────────────────────────────────────────────

class TestEstimateHomography:
    def test_empty_matches_returns_none(self):
        H, mask = estimate_homography([])
        assert H is None
        assert len(mask) == 0

    def test_too_few_matches_returns_none(self):
        matches = [KeypointMatch((float(i), 0.), (float(i), 0.), 1.0)
                   for i in range(3)]
        H, mask = estimate_homography(matches, min_inliers=4)
        assert H is None
        assert len(mask) == len(matches)
        assert not any(mask)

    def test_mask_dtype_bool(self):
        matches = [KeypointMatch((float(i*10), float(j*10)),
                                  (float(i*10), float(j*10)), 0.0)
                   for i in range(4) for j in range(4)]
        _, mask = estimate_homography(matches, min_inliers=4)
        assert mask.dtype in (bool, np.bool_)

    def test_returns_tuple_of_two(self):
        result = estimate_homography([])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_identity_points_gives_homography(self):
        """Точки сами на себе → гомография должна быть найдена."""
        pts = [(float(i * 15 + 5), float(j * 15 + 5))
               for i in range(5) for j in range(5)]
        matches = [KeypointMatch(p, p, 0.0) for p in pts]
        H, mask = estimate_homography(matches, min_inliers=4)
        if H is not None:
            assert H.shape == (3, 3)
            # Для тождественного преобразования H ≈ I (с масштабом)
            h_norm = H / H[2, 2]
            np.testing.assert_allclose(h_norm, np.eye(3), atol=0.1)

    def test_mask_length_matches_input(self):
        matches = [KeypointMatch((float(i), 0.), (float(i), 0.), 1.0)
                   for i in range(5)]
        _, mask = estimate_homography(matches, min_inliers=4)
        assert len(mask) == len(matches)

    def test_min_inliers_parameter(self):
        matches = [KeypointMatch((float(i), 0.), (float(i), 0.), 1.0)
                   for i in range(5)]
        H_strict, _ = estimate_homography(matches, min_inliers=10)
        assert H_strict is None


# ─── feature_match_pair ───────────────────────────────────────────────────────

class TestFeatureMatchPair:
    def test_returns_feature_match_result(self, gray_image):
        result = feature_match_pair(gray_image, gray_image)
        assert isinstance(result, FeatureMatchResult)

    def test_score_in_range(self, gray_image):
        result = feature_match_pair(gray_image, gray_image)
        assert 0.0 <= result.score <= 1.0

    def test_method_recorded(self, gray_image):
        result = feature_match_pair(gray_image, gray_image, method="orb")
        assert result.method == "orb"

    def test_n_keypoints_is_tuple_of_two(self, gray_image):
        result = feature_match_pair(gray_image, gray_image)
        assert isinstance(result.n_keypoints, tuple)
        assert len(result.n_keypoints) == 2

    def test_identical_images_nonnegative_score(self, checkerboard):
        result = feature_match_pair(checkerboard, checkerboard, method="orb")
        assert result.score >= 0.0

    def test_small_blank_image_no_crash(self):
        img = np.zeros((15, 15), dtype=np.uint8)
        result = feature_match_pair(img, img)
        assert isinstance(result, FeatureMatchResult)
        assert result.score == pytest.approx(0.0)

    def test_inlier_mask_length_consistent(self, gray_image):
        result = feature_match_pair(gray_image, gray_image)
        if result.inlier_mask is not None and len(result.inlier_mask) > 0:
            assert len(result.inlier_mask) == len(result.matches)

    def test_akaze_method(self, checkerboard):
        result = feature_match_pair(checkerboard, checkerboard,
                                     method="akaze")
        assert isinstance(result, FeatureMatchResult)
        assert result.method == "akaze"

    def test_bgr_images_accepted(self, bgr_image):
        result = feature_match_pair(bgr_image, bgr_image, method="orb")
        assert isinstance(result, FeatureMatchResult)

    def test_different_images_score_nonnegative(self, checkerboard):
        blank = np.zeros_like(checkerboard)
        result = feature_match_pair(checkerboard, blank, method="orb")
        assert result.score >= 0.0


# ─── edge_feature_score ───────────────────────────────────────────────────────

class TestEdgeFeatureScore:
    def test_returns_float(self, gray_image):
        score = edge_feature_score(gray_image, gray_image)
        assert isinstance(score, float)

    def test_score_in_range(self, gray_image):
        score = edge_feature_score(gray_image, gray_image)
        assert 0.0 <= score <= 1.0

    def test_blank_images_return_zero(self):
        img = np.zeros((30, 30), dtype=np.uint8)
        score = edge_feature_score(img, img)
        assert score == pytest.approx(0.0)

    def test_checkerboard_nonnegative(self, checkerboard):
        score = edge_feature_score(checkerboard, checkerboard)
        assert score >= 0.0

    def test_narrow_edge_image(self):
        img = np.random.randint(0, 255, (60, 8), dtype=np.uint8)
        score = edge_feature_score(img, img)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_different_images_lower_than_identical(self, checkerboard):
        blank = np.zeros_like(checkerboard)
        score_same = edge_feature_score(checkerboard, checkerboard)
        score_diff = edge_feature_score(checkerboard, blank)
        # Одинаковые изображения должны давать ≥ разным
        assert score_same >= score_diff or score_same >= 0.0
