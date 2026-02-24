"""Extra tests for puzzle_reconstruction/algorithms/synthesis.py."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

from puzzle_reconstruction.algorithms.synthesis import (
    compute_fractal_signature,
    build_edge_signatures,
    _normalize_curve,
    _build_one_edge,
)
from puzzle_reconstruction.models import (
    EdgeSide,
    EdgeSignature,
    FractalSignature,
    Fragment,
    ShapeClass,
    TangramSignature,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _circle(n=64, r=30.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1).astype(np.float32)


def _square(side=40.0, n=64):
    s = n // 4
    pts = []
    for i in range(s):
        pts.append([i * side / s, 0.0])
    for i in range(s):
        pts.append([side, i * side / s])
    for i in range(s):
        pts.append([side - i * side / s, side])
    for i in range(s):
        pts.append([0.0, side - i * side / s])
    return np.array(pts, dtype=np.float32)


def _fractal():
    return FractalSignature(
        fd_box=1.2, fd_divider=1.3,
        ifs_coeffs=np.ones(8, dtype=float),
        css_image=[(1.0, [0.1, 0.5, 0.9])],
        chain_code="01234567",
        curve=_circle(256),
    )


def _tangram():
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    return TangramSignature(
        polygon=polygon, shape_class=ShapeClass.RECTANGLE,
        centroid=np.array([0.5, 0.5]), angle=0.0, scale=1.0, area=1.0,
    )


def _fragment(fid=0, with_tangram=True, with_fractal=True):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:54, 10:54] = 255
    return Fragment(
        fragment_id=fid, image=img, mask=mask, contour=_square(),
        tangram=_tangram() if with_tangram else None,
        fractal=_fractal() if with_fractal else None,
    )


def _call_fractal(contour=None, **kw):
    if contour is None:
        contour = _circle()
    with (
        patch("puzzle_reconstruction.algorithms.fractal.box_counting.box_counting_fd",
              return_value=kw.get("fd_box", 1.2)),
        patch("puzzle_reconstruction.algorithms.fractal.divider.divider_fd",
              return_value=kw.get("fd_divider", 1.3)),
        patch("puzzle_reconstruction.algorithms.fractal.ifs.fit_ifs_coefficients",
              return_value=kw.get("ifs", np.ones(8))),
        patch("puzzle_reconstruction.algorithms.fractal.css.curvature_scale_space",
              return_value=[(1.0, [0.1])]),
        patch("puzzle_reconstruction.algorithms.fractal.css.css_to_feature_vector",
              return_value=np.ones(32)),
        patch("puzzle_reconstruction.algorithms.fractal.css.freeman_chain_code",
              return_value=kw.get("chain", "0123")),
    ):
        return compute_fractal_signature(contour)


def _call_build(n_points=32, n_sides=4, alpha=0.5, frag=None, **kw):
    if frag is None:
        frag = _fragment(**kw)
    with (
        patch("puzzle_reconstruction.algorithms.synthesis.extract_tangram_edge",
              return_value=np.zeros((n_points, 2), dtype=float)),
        patch("puzzle_reconstruction.algorithms.fractal.css.curvature_scale_space",
              return_value=[(1.0, [0.1])]),
        patch("puzzle_reconstruction.algorithms.fractal.css.css_to_feature_vector",
              return_value=np.ones(32)),
    ):
        return build_edge_signatures(frag, alpha=alpha,
                                     n_sides=n_sides, n_points=n_points)


# ─── compute_fractal_signature (extra) ────────────────────────────────────────

class TestComputeFractalSignatureExtra:
    def test_css_image_list_type(self):
        sig = _call_fractal()
        assert isinstance(sig.css_image, list)

    def test_curve_shape_matches_input(self):
        sig = _call_fractal(contour=_circle(128))
        assert sig.curve.shape[0] == 256  # resampled to 256

    def test_ifs_coeffs_dtype(self):
        sig = _call_fractal()
        assert sig.ifs_coeffs.dtype in (np.float32, np.float64)

    def test_fd_values_positive(self):
        sig = _call_fractal(fd_box=1.5, fd_divider=1.6)
        assert sig.fd_box > 0
        assert sig.fd_divider > 0

    def test_chain_code_string(self):
        sig = _call_fractal(chain="ABCDEF")
        assert sig.chain_code == "ABCDEF"

    def test_curve_is_2d(self):
        sig = _call_fractal()
        assert sig.curve.ndim == 2
        assert sig.curve.shape[1] == 2


# ─── build_edge_signatures (extra) ───────────────────────────────────────────

class TestBuildEdgeSignaturesExtra:
    def test_5_sides(self):
        result = _call_build(n_sides=5)
        assert len(result) == 5

    def test_1_side(self):
        result = _call_build(n_sides=1)
        assert len(result) == 1

    def test_edge_ids_start_at_zero(self):
        sigs = _call_build(n_sides=4)
        assert sigs[0].edge_id == 0

    def test_all_sides_are_edge_side(self):
        for sig in _call_build():
            assert isinstance(sig.side, EdgeSide)

    def test_css_vec_ndarray(self):
        for sig in _call_build():
            assert isinstance(sig.css_vec, np.ndarray)

    def test_ifs_from_fractal(self):
        frag = _fragment()
        sigs = _call_build(frag=frag)
        for sig in sigs:
            np.testing.assert_allclose(sig.ifs_coeffs, frag.fractal.ifs_coeffs)

    def test_length_positive(self):
        for sig in _call_build():
            assert sig.length >= 0.0

    def test_virtual_curve_shape(self):
        sigs = _call_build(n_points=48)
        for sig in sigs:
            assert sig.virtual_curve.shape == (48, 2)

    def test_different_n_points(self):
        s16 = _call_build(n_points=16)
        s64 = _call_build(n_points=64)
        assert s16[0].virtual_curve.shape[0] == 16
        assert s64[0].virtual_curve.shape[0] == 64


# ─── _normalize_curve (extra) ────────────────────────────────────────────────

class TestNormalizeCurveExtra:
    def test_output_shape(self):
        c = np.random.randn(15, 2)
        result = _normalize_curve(c)
        assert result.shape == (15, 2)

    def test_centroid_zero(self):
        c = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = _normalize_curve(c)
        np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_max_abs_one_or_zero(self):
        c = np.array([[5.0, 0.0], [-5.0, 0.0]])
        result = _normalize_curve(c)
        assert float(np.abs(result).max()) == pytest.approx(1.0, abs=1e-6)

    def test_single_point_zero(self):
        c = np.array([[7.0, 3.0]])
        result = _normalize_curve(c)
        np.testing.assert_allclose(result, [[0.0, 0.0]], atol=1e-10)

    def test_large_values_scaled(self):
        c = np.array([[1e6, 0.0], [-1e6, 0.0]])
        result = _normalize_curve(c)
        assert float(np.abs(result).max()) == pytest.approx(1.0, abs=1e-6)

    def test_negative_values_ok(self):
        c = np.array([[-100.0, -200.0], [100.0, 200.0]])
        result = _normalize_curve(c)
        np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0], atol=1e-10)
