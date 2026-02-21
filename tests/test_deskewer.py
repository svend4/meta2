"""Тесты для puzzle_reconstruction/preprocessing/deskewer.py."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.deskewer import (
    DeskewResult,
    estimate_skew_projection,
    estimate_skew_hough,
    deskew_image,
    auto_deskew,
    batch_deskew,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=200):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    img[30:34, :] = [50, 50, 50]   # тёмная горизонтальная полоса (текст-proxy)
    return img


def _text_like(h=80, w=120):
    """Серое изображение с горизонтальными тёмными полосами (имитация строк)."""
    img = np.full((h, w), 230, dtype=np.uint8)
    for row in range(10, h, 16):
        img[row:row + 4, 5:w - 5] = 30
    return img


def _skewed(h=80, w=120, angle=5.0):
    """Повёрнутое 'текстовое' изображение."""
    import cv2
    img = _text_like(h, w)
    cx, cy = float(w) / 2, float(h) / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=230)


# ─── DeskewResult ─────────────────────────────────────────────────────────────

class TestDeskewResult:
    def _make(self, **kw):
        defaults = dict(corrected=np.zeros((32, 32), dtype=np.uint8),
                        angle=3.0, method="projection")
        defaults.update(kw)
        return DeskewResult(**defaults)

    def test_corrected_stored(self):
        r = self._make()
        assert r.corrected.shape == (32, 32)

    def test_angle_stored(self):
        r = self._make(angle=-7.5)
        assert r.angle == pytest.approx(-7.5)

    def test_method_stored(self):
        assert self._make(method="hough").method == "hough"

    def test_confidence_default_zero(self):
        assert self._make().confidence == pytest.approx(0.0)

    def test_confidence_stored(self):
        r = self._make(confidence=0.85)
        assert r.confidence == pytest.approx(0.85)

    def test_params_default_empty(self):
        assert isinstance(self._make().params, dict)

    def test_params_stored(self):
        r = self._make(params={"n_angles": 60})
        assert r.params["n_angles"] == 60

    def test_repr_contains_class(self):
        assert "DeskewResult" in repr(self._make())

    def test_repr_contains_method(self):
        assert "projection" in repr(self._make())

    def test_corrected_ndarray(self):
        assert isinstance(self._make().corrected, np.ndarray)


# ─── estimate_skew_projection ─────────────────────────────────────────────────

class TestEstimateSkewProjection:
    def test_returns_tuple(self):
        r = estimate_skew_projection(_text_like())
        assert isinstance(r, tuple)
        assert len(r) == 2

    def test_angle_is_float(self):
        angle, _ = estimate_skew_projection(_text_like())
        assert isinstance(angle, float)

    def test_confidence_is_float(self):
        _, conf = estimate_skew_projection(_text_like())
        assert isinstance(conf, float)

    def test_confidence_in_range(self):
        _, conf = estimate_skew_projection(_text_like())
        assert 0.0 <= conf <= 1.0

    def test_angle_in_range(self):
        angle, _ = estimate_skew_projection(_text_like(), angle_range=(-10.0, 10.0))
        assert -10.0 <= angle <= 10.0

    def test_gray_input(self):
        r = estimate_skew_projection(_gray())
        assert len(r) == 2

    def test_bgr_input(self):
        r = estimate_skew_projection(_bgr())
        assert len(r) == 2

    def test_n_angles_param(self):
        # Разное n_angles не должно крашиться
        r1 = estimate_skew_projection(_text_like(), n_angles=20)
        r2 = estimate_skew_projection(_text_like(), n_angles=60)
        for angle, conf in (r1, r2):
            assert isinstance(angle, float)
            assert 0.0 <= conf <= 1.0

    def test_horizontal_text_near_zero(self):
        img   = _text_like()
        angle, _ = estimate_skew_projection(img, angle_range=(-15.0, 15.0))
        assert abs(angle) <= 15.0   # угол в диапазоне

    def test_wide_angle_range(self):
        angle, conf = estimate_skew_projection(
            _text_like(), angle_range=(-20.0, 20.0), n_angles=40
        )
        assert -20.0 <= angle <= 20.0


# ─── estimate_skew_hough ──────────────────────────────────────────────────────

class TestEstimateSkewHough:
    def test_returns_tuple(self):
        r = estimate_skew_hough(_text_like())
        assert isinstance(r, tuple)
        assert len(r) == 2

    def test_angle_is_float(self):
        angle, _ = estimate_skew_hough(_text_like())
        assert isinstance(angle, float)

    def test_confidence_is_float(self):
        _, conf = estimate_skew_hough(_text_like())
        assert isinstance(conf, float)

    def test_confidence_in_range(self):
        _, conf = estimate_skew_hough(_text_like())
        assert 0.0 <= conf <= 1.0

    def test_gray_input(self):
        r = estimate_skew_hough(_gray())
        assert len(r) == 2

    def test_bgr_input(self):
        r = estimate_skew_hough(_bgr())
        assert len(r) == 2

    def test_no_lines_returns_zero(self):
        # Пустое (однородное) изображение → линии не найдены → (0.0, 0.0)
        img   = _gray()
        angle, conf = estimate_skew_hough(img, angle_range=(-15.0, 15.0),
                                           threshold=10000)
        assert angle == pytest.approx(0.0)
        assert conf  == pytest.approx(0.0)

    def test_angle_in_range_or_zero(self):
        angle, _ = estimate_skew_hough(_text_like(), angle_range=(-15.0, 15.0))
        assert -15.0 <= angle <= 15.0 or angle == pytest.approx(0.0)

    def test_threshold_param(self):
        r = estimate_skew_hough(_text_like(), threshold=5)
        assert len(r) == 2


# ─── deskew_image ─────────────────────────────────────────────────────────────

class TestDeskewImage:
    def test_returns_ndarray(self):
        assert isinstance(deskew_image(_gray(), 0.0), np.ndarray)

    def test_same_shape_gray(self):
        r = deskew_image(_gray(40, 60), 5.0)
        assert r.shape == (40, 60)

    def test_same_shape_bgr(self):
        r = deskew_image(_bgr(40, 60), 5.0)
        assert r.shape == (40, 60, 3)

    def test_dtype_uint8(self):
        assert deskew_image(_gray(), 10.0).dtype == np.uint8

    def test_angle_zero_preserves(self):
        img = _gray()
        r   = deskew_image(img, 0.0)
        np.testing.assert_array_equal(r, img)

    def test_nonzero_angle_changes(self):
        img = _text_like()
        r   = deskew_image(img, 5.0)
        assert not np.array_equal(img, r)

    def test_fill_value(self):
        img = _gray(val=50)
        r   = deskew_image(img, 45.0, fill=0)
        assert r[0, 0] == 0

    def test_fill_default_255(self):
        img = _gray(val=0)
        r   = deskew_image(img, 45.0)
        assert r[0, 0] == 255

    def test_gray_input(self):
        r = deskew_image(_gray(), 3.0)
        assert r.ndim == 2

    def test_bgr_input(self):
        r = deskew_image(_bgr(), 3.0)
        assert r.ndim == 3

    @pytest.mark.parametrize("angle", [-15.0, -5.0, 0.0, 5.0, 15.0])
    def test_various_angles(self, angle):
        r = deskew_image(_text_like(), angle)
        assert r.shape == (80, 120)


# ─── auto_deskew ──────────────────────────────────────────────────────────────

class TestAutoDeskew:
    def test_returns_result(self):
        assert isinstance(auto_deskew(_text_like()), DeskewResult)

    def test_method_projection(self):
        r = auto_deskew(_text_like(), method="projection")
        assert r.method == "projection"

    def test_method_hough(self):
        r = auto_deskew(_text_like(), method="hough")
        assert r.method == "hough"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            auto_deskew(_gray(), method="magic_deskew_xyz")

    def test_corrected_ndarray(self):
        r = auto_deskew(_text_like())
        assert isinstance(r.corrected, np.ndarray)

    def test_same_shape(self):
        img = _text_like(80, 120)
        r   = auto_deskew(img)
        assert r.corrected.shape == (80, 120)

    def test_dtype_uint8(self):
        assert auto_deskew(_text_like()).corrected.dtype == np.uint8

    def test_angle_float(self):
        assert isinstance(auto_deskew(_text_like()).angle, float)

    def test_confidence_in_range(self):
        r = auto_deskew(_text_like())
        assert 0.0 <= r.confidence <= 1.0

    def test_params_stored_projection(self):
        r = auto_deskew(_text_like(), method="projection", n_angles=30)
        assert r.params.get("n_angles") == 30

    def test_params_stored_hough(self):
        r = auto_deskew(_text_like(), method="hough", threshold=20)
        assert r.params.get("threshold") == 20

    def test_gray_input(self):
        r = auto_deskew(_gray())
        assert r.corrected.ndim == 2

    def test_bgr_input(self):
        r = auto_deskew(_bgr())
        assert r.corrected.ndim == 3

    def test_angle_in_range(self):
        r = auto_deskew(_text_like(), angle_range=(-10.0, 10.0))
        assert -10.0 <= r.angle <= 10.0


# ─── batch_deskew ─────────────────────────────────────────────────────────────

class TestBatchDeskew:
    def test_returns_list(self):
        assert isinstance(batch_deskew([_text_like(), _gray()]), list)

    def test_same_length(self):
        imgs = [_text_like(), _gray(), _bgr()]
        r    = batch_deskew(imgs)
        assert len(r) == 3

    def test_empty_returns_empty(self):
        assert batch_deskew([]) == []

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_deskew([_gray()], method="warp_xyz")

    def test_each_is_result(self):
        for r in batch_deskew([_text_like(), _gray()]):
            assert isinstance(r, DeskewResult)

    @pytest.mark.parametrize("method", ["projection", "hough"])
    def test_both_methods(self, method):
        r = batch_deskew([_text_like()], method=method)
        assert r[0].method == method

    def test_kwargs_n_angles_forwarded(self):
        r = batch_deskew([_text_like()], method="projection", n_angles=20)
        assert r[0].params.get("n_angles") == 20

    def test_kwargs_threshold_forwarded(self):
        r = batch_deskew([_text_like()], method="hough", threshold=30)
        assert r[0].params.get("threshold") == 30

    def test_shapes_preserved(self):
        imgs = [_text_like(48, 80), _text_like(64, 96)]
        r    = batch_deskew(imgs)
        assert r[0].corrected.shape == (48, 80)
        assert r[1].corrected.shape == (64, 96)
