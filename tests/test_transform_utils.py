"""Тесты для puzzle_reconstruction/utils/transform_utils.py."""
import numpy as np
import pytest

from puzzle_reconstruction.utils.transform_utils import (
    rotate_image,
    flip_image,
    scale_image,
    crop_region,
    affine_from_params,
    compose_affines,
    apply_affine,
    apply_homography,
    batch_rotate,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 100
    img[:, :, 2] = 50
    return img


def _noisy(h=64, w=64, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _identity_affine():
    return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


def _identity_homography():
    return np.eye(3, dtype=np.float32)


# ─── rotate_image ─────────────────────────────────────────────────────────────

class TestRotateImage:
    def test_returns_ndarray(self):
        assert isinstance(rotate_image(_gray(), 0.0), np.ndarray)

    def test_same_shape_gray(self):
        r = rotate_image(_gray(40, 60), 45.0)
        assert r.shape == (40, 60)

    def test_same_shape_bgr(self):
        r = rotate_image(_bgr(40, 60), 45.0)
        assert r.shape == (40, 60, 3)

    def test_dtype_uint8(self):
        assert rotate_image(_gray(), 30.0).dtype == np.uint8

    def test_angle_zero_preserves(self):
        img = _gray()
        r   = rotate_image(img, 0.0)
        np.testing.assert_array_equal(r, img)

    def test_angle_nonzero_changes(self):
        img = _noisy()
        r   = rotate_image(img, 45.0)
        assert not np.array_equal(img, r)

    def test_fill_value_in_corner(self):
        img = _gray(val=50)
        r   = rotate_image(img, 45.0, fill=0)
        # При вращении угол изображения должен заполниться fill=0
        assert r[0, 0] == 0

    def test_custom_center(self):
        img = _gray()
        r1  = rotate_image(img, 30.0, center=None)
        r2  = rotate_image(img, 30.0, center=(32.0, 32.0))
        assert isinstance(r2, np.ndarray)
        assert r2.shape == img.shape

    def test_fill_default_255(self):
        img = _gray(val=0)
        r   = rotate_image(img, 45.0)
        # Угол должен быть заполнен fill=255 (по умолчанию)
        assert r[0, 0] == 255

    def test_bgr_input(self):
        r = rotate_image(_bgr(), 15.0)
        assert r.ndim == 3

    @pytest.mark.parametrize("angle", [0, 90, 180, 270])
    def test_common_angles(self, angle):
        r = rotate_image(_gray(), float(angle))
        assert r.shape == (64, 64)

    def test_center_tuple(self):
        r = rotate_image(_gray(50, 80), 20.0, center=(40.0, 25.0))
        assert r.shape == (50, 80)


# ─── flip_image ───────────────────────────────────────────────────────────────

class TestFlipImage:
    def test_returns_ndarray(self):
        assert isinstance(flip_image(_gray()), np.ndarray)

    def test_same_shape_gray(self):
        assert flip_image(_gray(40, 60), mode=1).shape == (40, 60)

    def test_same_shape_bgr(self):
        assert flip_image(_bgr(40, 60), mode=1).shape == (40, 60, 3)

    def test_dtype_uint8(self):
        assert flip_image(_gray()).dtype == np.uint8

    def test_horizontal_flip_content(self):
        img = _noisy()
        r   = flip_image(img, mode=1)
        # Строки зеркальны
        np.testing.assert_array_equal(r[0], img[0, ::-1])

    def test_vertical_flip_content(self):
        img = _noisy()
        r   = flip_image(img, mode=0)
        np.testing.assert_array_equal(r[:, 0], img[::-1, 0])

    def test_both_flip_mode_minus1(self):
        img = _noisy()
        r   = flip_image(img, mode=-1)
        expected = np.flipud(np.fliplr(img))
        np.testing.assert_array_equal(r, expected)

    def test_double_flip_restores(self):
        img = _noisy()
        np.testing.assert_array_equal(flip_image(flip_image(img, 1), 1), img)

    def test_bgr_mode1(self):
        r = flip_image(_bgr(), mode=1)
        assert r.ndim == 3

    def test_default_mode_1(self):
        img = _noisy()
        r1  = flip_image(img)
        r2  = flip_image(img, mode=1)
        np.testing.assert_array_equal(r1, r2)


# ─── scale_image ──────────────────────────────────────────────────────────────

class TestScaleImage:
    def test_returns_ndarray(self):
        assert isinstance(scale_image(_gray()), np.ndarray)

    def test_dtype_uint8(self):
        assert scale_image(_gray(), sx=2.0).dtype == np.uint8

    def test_sx_1_same_size(self):
        r = scale_image(_gray(40, 60), sx=1.0)
        assert r.shape == (40, 60)

    def test_sx_2_doubles_width_height(self):
        r = scale_image(_gray(40, 60), sx=2.0)
        assert r.shape == (80, 120)

    def test_sy_none_proportional(self):
        r = scale_image(_gray(40, 60), sx=2.0, sy=None)
        assert r.shape == (80, 120)

    def test_sx_sy_different(self):
        r = scale_image(_gray(40, 60), sx=2.0, sy=0.5)
        assert r.shape == (20, 120)

    def test_min_size_1(self):
        r = scale_image(_gray(), sx=0.001)
        assert r.shape[0] >= 1
        assert r.shape[1] >= 1

    def test_gray_input(self):
        r = scale_image(_gray(), sx=1.5)
        assert r.ndim == 2

    def test_bgr_input(self):
        r = scale_image(_bgr(), sx=1.5)
        assert r.ndim == 3

    def test_large_scale(self):
        r = scale_image(_gray(10, 10), sx=5.0)
        assert r.shape == (50, 50)


# ─── crop_region ──────────────────────────────────────────────────────────────

class TestCropRegion:
    def test_returns_ndarray(self):
        assert isinstance(crop_region(_gray(), 0, 0, 30, 30), np.ndarray)

    def test_correct_shape(self):
        r = crop_region(_gray(64, 64), x=10, y=5, w=20, h=15)
        assert r.shape == (15, 20)

    def test_full_image(self):
        img = _gray(40, 60)
        r   = crop_region(img, 0, 0, 60, 40)
        np.testing.assert_array_equal(r, img)

    def test_single_pixel(self):
        r = crop_region(_gray(), 10, 10, 1, 1)
        assert r.shape == (1, 1)

    def test_clamp_out_of_bounds(self):
        r = crop_region(_gray(64, 64), x=50, y=50, w=100, h=100, clamp=True)
        assert r.shape[0] >= 1
        assert r.shape[1] >= 1

    def test_empty_after_clamp_raises(self):
        with pytest.raises(ValueError):
            crop_region(_gray(64, 64), x=64, y=64, w=10, h=10, clamp=True)

    def test_bgr_shape(self):
        r = crop_region(_bgr(64, 64), 5, 5, 20, 30)
        assert r.shape == (30, 20, 3)

    def test_values_preserved(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        img[5:10, 5:10] = 200
        r = crop_region(img, 5, 5, 5, 5)
        assert r.min() == 200

    def test_x_zero_y_zero(self):
        img = _noisy(40, 50)
        r   = crop_region(img, 0, 0, 30, 25)
        np.testing.assert_array_equal(r, img[:25, :30])


# ─── affine_from_params ───────────────────────────────────────────────────────

class TestAffineFromParams:
    def test_returns_ndarray(self):
        assert isinstance(affine_from_params(), np.ndarray)

    def test_shape_2x3(self):
        assert affine_from_params().shape == (2, 3)

    def test_dtype_float32(self):
        assert affine_from_params().dtype == np.float32

    def test_identity_default(self):
        M = affine_from_params(angle=0.0, tx=0.0, ty=0.0, sx=1.0, sy=1.0)
        expected = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        np.testing.assert_allclose(M, expected, atol=1e-5)

    def test_translation_tx(self):
        M = affine_from_params(tx=10.0)
        assert M[0, 2] == pytest.approx(10.0, abs=1e-4)

    def test_translation_ty(self):
        M = affine_from_params(ty=7.0)
        assert M[1, 2] == pytest.approx(7.0, abs=1e-4)

    def test_scale_sx(self):
        M = affine_from_params(sx=2.0, sy=2.0, angle=0.0)
        assert M[0, 0] == pytest.approx(2.0, abs=1e-4)
        assert M[1, 1] == pytest.approx(2.0, abs=1e-4)

    def test_sy_none_equals_sx(self):
        M1 = affine_from_params(sx=1.5, sy=None)
        M2 = affine_from_params(sx=1.5, sy=1.5)
        np.testing.assert_allclose(M1, M2, atol=1e-5)

    def test_angle_90_rotation(self):
        M = affine_from_params(angle=90.0, sx=1.0, sy=1.0)
        # cos(90°)=0, sin(90°)=1 → M[0,0]≈0, M[0,1]≈-1
        assert abs(M[0, 0]) < 1e-4
        assert M[0, 1] == pytest.approx(-1.0, abs=1e-4)

    def test_center_of_rotation_stored(self):
        M = affine_from_params(angle=90.0, cx=32.0, cy=32.0)
        assert M.shape == (2, 3)


# ─── compose_affines ──────────────────────────────────────────────────────────

class TestComposeAffines:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compose_affines([])

    def test_single_identity(self):
        M   = _identity_affine()
        res = compose_affines([M])
        np.testing.assert_allclose(res, M, atol=1e-5)

    def test_returns_2x3(self):
        assert compose_affines([_identity_affine()]).shape == (2, 3)

    def test_dtype_float32(self):
        assert compose_affines([_identity_affine()]).dtype == np.float32

    def test_identity_composed_twice(self):
        I = _identity_affine()
        res = compose_affines([I, I])
        np.testing.assert_allclose(res, I, atol=1e-5)

    def test_translation_composes(self):
        T1 = affine_from_params(tx=5.0)
        T2 = affine_from_params(tx=3.0)
        res = compose_affines([T1, T2])
        # Суммарный перенос ≈ 8.0
        assert res[0, 2] == pytest.approx(8.0, abs=1e-4)

    def test_scale_composes(self):
        S1  = affine_from_params(sx=2.0, sy=2.0)
        S2  = affine_from_params(sx=3.0, sy=3.0)
        res = compose_affines([S1, S2])
        # Суммарный масштаб ≈ 6.0
        assert res[0, 0] == pytest.approx(6.0, abs=1e-4)

    def test_three_matrices(self):
        I   = _identity_affine()
        res = compose_affines([I, I, I])
        np.testing.assert_allclose(res, I, atol=1e-5)


# ─── apply_affine ─────────────────────────────────────────────────────────────

class TestApplyAffine:
    def test_returns_ndarray(self):
        assert isinstance(apply_affine(_gray(), _identity_affine()), np.ndarray)

    def test_same_shape_default(self):
        r = apply_affine(_gray(40, 60), _identity_affine())
        assert r.shape == (40, 60)

    def test_dtype_uint8(self):
        assert apply_affine(_gray(), _identity_affine()).dtype == np.uint8

    def test_identity_preserves(self):
        img = _gray()
        r   = apply_affine(img, _identity_affine())
        np.testing.assert_array_equal(r, img)

    def test_custom_size(self):
        r = apply_affine(_gray(40, 60), _identity_affine(), size=(80, 120))
        assert r.shape == (120, 80)

    def test_fill_value(self):
        M = affine_from_params(tx=100.0)  # сдвигает содержимое за пределы
        r = apply_affine(_gray(20, 20, val=50), M, fill=0)
        # Большинство пикселей должно быть заполнено нулями
        assert r.min() == 0

    def test_gray_input(self):
        r = apply_affine(_gray(), _identity_affine())
        assert r.ndim == 2

    def test_bgr_input(self):
        r = apply_affine(_bgr(), _identity_affine())
        assert r.ndim == 3

    def test_rotation_affine(self):
        M = affine_from_params(angle=30.0, cx=32.0, cy=32.0)
        r = apply_affine(_gray(), M)
        assert r.shape == (64, 64)


# ─── apply_homography ─────────────────────────────────────────────────────────

class TestApplyHomography:
    def test_returns_ndarray(self):
        assert isinstance(apply_homography(_gray(), _identity_homography()), np.ndarray)

    def test_same_shape_default(self):
        r = apply_homography(_gray(40, 60), _identity_homography())
        assert r.shape == (40, 60)

    def test_dtype_uint8(self):
        assert apply_homography(_gray(), _identity_homography()).dtype == np.uint8

    def test_identity_preserves(self):
        img = _gray()
        r   = apply_homography(img, _identity_homography())
        np.testing.assert_array_equal(r, img)

    def test_custom_size(self):
        r = apply_homography(_gray(40, 60), _identity_homography(), size=(80, 120))
        assert r.shape == (120, 80)

    def test_fill_value(self):
        H = np.eye(3, dtype=np.float32)
        H[0, 2] = 200.0   # большой сдвиг
        r = apply_homography(_gray(20, 20, val=50), H, fill=0)
        assert r.min() == 0

    def test_gray_input(self):
        r = apply_homography(_gray(), _identity_homography())
        assert r.ndim == 2

    def test_bgr_input(self):
        r = apply_homography(_bgr(), _identity_homography())
        assert r.ndim == 3

    def test_float64_homography(self):
        H = np.eye(3, dtype=np.float64)
        r = apply_homography(_gray(), H)
        assert r.shape == (64, 64)


# ─── batch_rotate ─────────────────────────────────────────────────────────────

class TestBatchRotate:
    def test_returns_list(self):
        assert isinstance(batch_rotate([_gray(), _gray()], 0.0), list)

    def test_same_length(self):
        imgs = [_gray(), _gray(), _gray()]
        r    = batch_rotate(imgs, 45.0)
        assert len(r) == 3

    def test_empty_returns_empty(self):
        assert batch_rotate([], 45.0) == []

    def test_each_is_ndarray(self):
        for r in batch_rotate([_gray(), _gray()], 30.0):
            assert isinstance(r, np.ndarray)

    def test_each_same_shape_gray(self):
        imgs = [_gray(40, 60)] * 3
        for r in batch_rotate(imgs, 15.0):
            assert r.shape == (40, 60)

    def test_angle_zero_preserves(self):
        imgs = [_gray()]
        r    = batch_rotate(imgs, 0.0)
        np.testing.assert_array_equal(r[0], imgs[0])

    def test_bgr_input(self):
        r = batch_rotate([_bgr()], 30.0)
        assert r[0].ndim == 3

    def test_fill_forwarded(self):
        imgs = [_gray(val=50)]
        r    = batch_rotate(imgs, 45.0, fill=0)
        assert r[0][0, 0] == 0

    def test_different_sizes(self):
        imgs = [_gray(32, 32), _gray(64, 128)]
        r    = batch_rotate(imgs, 10.0)
        assert r[0].shape == (32, 32)
        assert r[1].shape == (64, 128)
