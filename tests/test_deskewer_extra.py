"""Extra tests for puzzle_reconstruction.preprocessing.deskewer."""
import cv2
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=200):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    img[30:34, :] = [50, 50, 50]
    return img


def _text_like(h=80, w=120):
    img = np.full((h, w), 230, dtype=np.uint8)
    for row in range(10, h, 16):
        img[row:row + 4, 5:w - 5] = 30
    return img


def _skewed(h=80, w=120, angle=5.0):
    img = _text_like(h, w)
    cx, cy = float(w) / 2, float(h) / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=230)


# ─── TestDeskewResultExtra ──────────────────────────────────────────────────

class TestDeskewResultExtra:
    def _make(self, **kw):
        defaults = dict(corrected=np.zeros((32, 32), dtype=np.uint8),
                        angle=3.0, method="projection")
        defaults.update(kw)
        return DeskewResult(**defaults)

    def test_angle_zero_ok(self):
        r = self._make(angle=0.0)
        assert r.angle == pytest.approx(0.0)

    def test_negative_angle_ok(self):
        r = self._make(angle=-12.5)
        assert r.angle == pytest.approx(-12.5)

    def test_method_hough(self):
        r = self._make(method="hough")
        assert r.method == "hough"

    def test_confidence_zero(self):
        r = self._make(confidence=0.0)
        assert r.confidence == pytest.approx(0.0)

    def test_confidence_one(self):
        r = self._make(confidence=1.0)
        assert r.confidence == pytest.approx(1.0)

    def test_params_multiple_keys(self):
        r = self._make(params={"n_angles": 60, "angle_range": (-10, 10)})
        assert r.params["n_angles"] == 60
        assert r.params["angle_range"] == (-10, 10)

    def test_corrected_dtype(self):
        r = self._make()
        assert r.corrected.dtype == np.uint8

    def test_repr_type(self):
        r = self._make()
        assert isinstance(repr(r), str)


# ─── TestEstimateSkewProjectionExtra ─────────────────────────────────────────

class TestEstimateSkewProjectionExtra:
    def test_small_image(self):
        img = _gray(16, 24)
        angle, conf = estimate_skew_projection(img)
        assert isinstance(angle, float)
        assert 0.0 <= conf <= 1.0

    def test_large_image(self):
        img = _text_like(160, 240)
        angle, conf = estimate_skew_projection(img)
        assert isinstance(angle, float)
        assert 0.0 <= conf <= 1.0

    def test_rectangular_image(self):
        img = _text_like(48, 200)
        angle, conf = estimate_skew_projection(img)
        assert isinstance(angle, float)

    def test_bgr_accepted(self):
        img = _bgr(80, 120)
        angle, conf = estimate_skew_projection(img)
        assert isinstance(angle, float)

    def test_n_angles_1(self):
        angle, conf = estimate_skew_projection(_text_like(), n_angles=1)
        assert isinstance(angle, float)
        assert 0.0 <= conf <= 1.0

    def test_n_angles_100(self):
        angle, conf = estimate_skew_projection(_text_like(), n_angles=100)
        assert isinstance(angle, float)

    def test_narrow_range(self):
        angle, conf = estimate_skew_projection(_text_like(),
                                               angle_range=(-2.0, 2.0))
        assert -2.0 <= angle <= 2.0

    def test_all_white_image(self):
        img = np.full((64, 96), 255, dtype=np.uint8)
        angle, conf = estimate_skew_projection(img)
        assert isinstance(angle, float)


# ─── TestEstimateSkewHoughExtra ─────────────────────────────────────────────

class TestEstimateSkewHoughExtra:
    def test_small_image(self):
        img = _gray(16, 24)
        angle, conf = estimate_skew_hough(img)
        assert isinstance(angle, float)

    def test_bgr_accepted(self):
        angle, conf = estimate_skew_hough(_bgr())
        assert isinstance(angle, float)

    def test_high_threshold_zero(self):
        img = _gray()
        angle, conf = estimate_skew_hough(img, threshold=100000)
        assert angle == pytest.approx(0.0)
        assert conf == pytest.approx(0.0)

    def test_low_threshold_runs(self):
        angle, conf = estimate_skew_hough(_text_like(), threshold=1)
        assert isinstance(angle, float)

    def test_narrow_range(self):
        angle, conf = estimate_skew_hough(_text_like(),
                                          angle_range=(-3.0, 3.0))
        assert -3.0 <= angle <= 3.0 or angle == pytest.approx(0.0)

    def test_large_image(self):
        img = _text_like(160, 240)
        angle, conf = estimate_skew_hough(img)
        assert isinstance(angle, float)


# ─── TestDeskewImageExtra ───────────────────────────────────────────────────

class TestDeskewImageExtra:
    @pytest.mark.parametrize("angle", [-20.0, -10.0, 0.0, 10.0, 20.0])
    def test_angle_range_gray(self, angle):
        r = deskew_image(_gray(40, 60), angle)
        assert r.shape == (40, 60)

    @pytest.mark.parametrize("angle", [-5.0, 0.0, 5.0])
    def test_angle_range_bgr(self, angle):
        r = deskew_image(_bgr(40, 60), angle)
        assert r.shape == (40, 60, 3)

    def test_fill_50(self):
        img = _gray(val=200)
        r = deskew_image(img, 30.0, fill=50)
        assert r[0, 0] == 50

    def test_fill_0(self):
        img = _gray(val=200)
        r = deskew_image(img, 30.0, fill=0)
        assert r[0, 0] == 0

    def test_large_angle_no_crash(self):
        r = deskew_image(_text_like(), 89.0)
        assert r.dtype == np.uint8

    def test_small_image(self):
        img = np.full((4, 4), 100, dtype=np.uint8)
        r = deskew_image(img, 5.0)
        assert r.shape == (4, 4)

    def test_returns_uint8(self):
        r = deskew_image(_gray(), 15.0)
        assert r.dtype == np.uint8


# ─── TestAutoDeskewExtra ───────────────────────────────────────────────────

class TestAutoDeskewExtra:
    def test_projection_corrected_shape(self):
        img = _text_like()
        r = auto_deskew(img, method="projection")
        assert r.corrected.shape == (80, 120)

    def test_hough_corrected_shape(self):
        img = _text_like()
        r = auto_deskew(img, method="hough")
        assert r.corrected.shape == (80, 120)

    def test_confidence_projection(self):
        r = auto_deskew(_text_like(), method="projection")
        assert 0.0 <= r.confidence <= 1.0

    def test_confidence_hough(self):
        r = auto_deskew(_text_like(), method="hough")
        assert 0.0 <= r.confidence <= 1.0

    def test_bgr_shape(self):
        r = auto_deskew(_bgr(80, 120))
        assert r.corrected.shape == (80, 120, 3)

    def test_angle_is_float(self):
        r = auto_deskew(_text_like())
        assert isinstance(r.angle, float)

    def test_n_angles_30_projection(self):
        r = auto_deskew(_text_like(), method="projection", n_angles=30)
        assert r.params["n_angles"] == 30

    def test_threshold_5_hough(self):
        r = auto_deskew(_text_like(), method="hough", threshold=5)
        assert r.params["threshold"] == 5

    def test_small_image(self):
        img = _gray(16, 24)
        r = auto_deskew(img)
        assert r.corrected.shape == (16, 24)

    def test_angle_range_param(self):
        r = auto_deskew(_text_like(), angle_range=(-5.0, 5.0))
        assert -5.0 <= r.angle <= 5.0


# ─── TestBatchDeskewExtra ──────────────────────────────────────────────────

class TestBatchDeskewExtra:
    def test_single_image(self):
        result = batch_deskew([_text_like()])
        assert len(result) == 1
        assert isinstance(result[0], DeskewResult)

    def test_five_images(self):
        result = batch_deskew([_text_like()] * 5)
        assert len(result) == 5

    def test_mixed_gray_bgr(self):
        imgs = [_gray(), _bgr(), _text_like()]
        result = batch_deskew(imgs)
        assert len(result) == 3

    def test_all_results_deskew_type(self):
        result = batch_deskew([_gray(), _text_like()])
        for r in result:
            assert isinstance(r, DeskewResult)

    def test_hough_method(self):
        result = batch_deskew([_text_like()], method="hough")
        assert result[0].method == "hough"

    def test_projection_method(self):
        result = batch_deskew([_text_like()], method="projection")
        assert result[0].method == "projection"

    def test_n_angles_forwarded(self):
        result = batch_deskew([_text_like()], method="projection", n_angles=25)
        assert result[0].params.get("n_angles") == 25

    def test_corrected_same_shapes(self):
        imgs = [_text_like(48, 80), _text_like(64, 96)]
        result = batch_deskew(imgs)
        assert result[0].corrected.shape == (48, 80)
        assert result[1].corrected.shape == (64, 96)
