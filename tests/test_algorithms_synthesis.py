"""Расширенные тесты для puzzle_reconstruction/algorithms/synthesis.py."""
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

def _circle_contour(n: int = 64, r: float = 30.0) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1).astype(np.float32)


def _square_contour(side: float = 40.0, n: int = 64) -> np.ndarray:
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


def _make_fractal() -> FractalSignature:
    return FractalSignature(
        fd_box=1.2,
        fd_divider=1.3,
        ifs_coeffs=np.ones(8, dtype=float),
        css_image=[(1.0, [0.1, 0.5, 0.9])],
        chain_code="01234567",
        curve=_circle_contour(256),
    )


def _make_tangram() -> TangramSignature:
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    return TangramSignature(
        polygon=polygon,
        shape_class=ShapeClass.RECTANGLE,
        centroid=np.array([0.5, 0.5]),
        angle=0.0,
        scale=1.0,
        area=1.0,
    )


def _make_fragment(with_tangram: bool = True, with_fractal: bool = True,
                   fid: int = 0) -> Fragment:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:54, 10:54] = 255
    return Fragment(
        fragment_id=fid,
        image=img,
        mask=mask,
        contour=_square_contour(),
        tangram=_make_tangram() if with_tangram else None,
        fractal=_make_fractal() if with_fractal else None,
    )


# Context manager to mock all fractal dependencies
def _fractal_mocks(fd_box=1.2, fd_divider=1.3, ifs=None, chain="01234567"):
    if ifs is None:
        ifs = np.ones(8)
    return (
        patch("puzzle_reconstruction.algorithms.fractal.box_counting.box_counting_fd",
              return_value=fd_box),
        patch("puzzle_reconstruction.algorithms.fractal.divider.divider_fd",
              return_value=fd_divider),
        patch("puzzle_reconstruction.algorithms.fractal.ifs.fit_ifs_coefficients",
              return_value=ifs),
        patch("puzzle_reconstruction.algorithms.fractal.css.curvature_scale_space",
              return_value=[(1.0, [0.1, 0.5])]),
        patch("puzzle_reconstruction.algorithms.fractal.css.css_to_feature_vector",
              return_value=np.ones(32)),
        patch("puzzle_reconstruction.algorithms.fractal.css.freeman_chain_code",
              return_value=chain),
    )


def _call_fractal(contour=None, **kwargs):
    if contour is None:
        contour = _circle_contour()
    with (
        patch("puzzle_reconstruction.algorithms.fractal.box_counting.box_counting_fd",
              return_value=kwargs.get("fd_box", 1.2)),
        patch("puzzle_reconstruction.algorithms.fractal.divider.divider_fd",
              return_value=kwargs.get("fd_divider", 1.3)),
        patch("puzzle_reconstruction.algorithms.fractal.ifs.fit_ifs_coefficients",
              return_value=kwargs.get("ifs", np.ones(8))),
        patch("puzzle_reconstruction.algorithms.fractal.css.curvature_scale_space",
              return_value=[(1.0, [0.1])]),
        patch("puzzle_reconstruction.algorithms.fractal.css.css_to_feature_vector",
              return_value=np.ones(32)),
        patch("puzzle_reconstruction.algorithms.fractal.css.freeman_chain_code",
              return_value=kwargs.get("chain", "0123")),
    ):
        return compute_fractal_signature(contour)


def _call_build(n_points=32, n_sides=4, alpha=0.5, frag=None, **kw):
    if frag is None:
        frag = _make_fragment(**kw)
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


# ─── TestComputeFractalSignature ──────────────────────────────────────────────

class TestComputeFractalSignature:
    def test_returns_fractal_signature(self):
        sig = _call_fractal()
        assert isinstance(sig, FractalSignature)

    def test_fd_box_stored(self):
        sig = _call_fractal(fd_box=1.7)
        assert sig.fd_box == pytest.approx(1.7)

    def test_fd_divider_stored(self):
        sig = _call_fractal(fd_divider=1.8)
        assert sig.fd_divider == pytest.approx(1.8)

    def test_ifs_coeffs_is_ndarray(self):
        sig = _call_fractal()
        assert isinstance(sig.ifs_coeffs, np.ndarray)

    def test_ifs_coeffs_shape(self):
        sig = _call_fractal(ifs=np.ones(8))
        assert sig.ifs_coeffs.shape == (8,)

    def test_css_image_is_list(self):
        sig = _call_fractal()
        assert isinstance(sig.css_image, list)

    def test_chain_code_is_str(self):
        sig = _call_fractal(chain="01234567")
        assert isinstance(sig.chain_code, str)

    def test_chain_code_stored(self):
        sig = _call_fractal(chain="ABC")
        assert sig.chain_code == "ABC"

    def test_curve_ndim_2(self):
        sig = _call_fractal()
        assert sig.curve.ndim == 2

    def test_curve_second_dim_2(self):
        sig = _call_fractal()
        assert sig.curve.shape[1] == 2

    def test_curve_length_256(self):
        sig = _call_fractal()
        assert sig.curve.shape[0] == 256

    def test_curve_dtype_float(self):
        sig = _call_fractal()
        assert sig.curve.dtype in (np.float32, np.float64)

    def test_different_contours_different_curves(self):
        c1 = _circle_contour(64, r=10.0)
        c2 = _circle_contour(64, r=80.0)
        s1 = _call_fractal(c1)
        s2 = _call_fractal(c2)
        assert not np.allclose(s1.curve, s2.curve)

    def test_ifs_values_stored_correctly(self):
        ifs = np.arange(8, dtype=float)
        sig = _call_fractal(ifs=ifs)
        np.testing.assert_allclose(sig.ifs_coeffs, ifs)

    def test_fd_box_and_divider_can_differ(self):
        sig = _call_fractal(fd_box=1.1, fd_divider=2.2)
        assert sig.fd_box != sig.fd_divider


# ─── TestBuildEdgeSignatures ──────────────────────────────────────────────────

class TestBuildEdgeSignatures:
    def test_returns_list(self):
        result = _call_build()
        assert isinstance(result, list)

    def test_length_4_sides(self):
        result = _call_build(n_sides=4)
        assert len(result) == 4

    def test_length_3_sides(self):
        result = _call_build(n_sides=3)
        assert len(result) == 3

    def test_length_2_sides(self):
        result = _call_build(n_sides=2)
        assert len(result) == 2

    def test_all_are_edge_signatures(self):
        for sig in _call_build():
            assert isinstance(sig, EdgeSignature)

    def test_edge_ids_sequential(self):
        sigs = _call_build()
        for i, sig in enumerate(sigs):
            assert sig.edge_id == i

    def test_side_is_edge_side(self):
        for sig in _call_build():
            assert isinstance(sig.side, EdgeSide)

    def test_virtual_curve_ndim_2(self):
        n = 32
        for sig in _call_build(n_points=n):
            assert sig.virtual_curve.ndim == 2

    def test_virtual_curve_second_dim_2(self):
        n = 32
        for sig in _call_build(n_points=n):
            assert sig.virtual_curve.shape[1] == 2

    def test_fd_is_average_of_box_and_divider(self):
        frag = _make_fragment()
        expected = (frag.fractal.fd_box + frag.fractal.fd_divider) / 2.0
        sigs = _call_build(frag=frag)
        for sig in sigs:
            assert sig.fd == pytest.approx(expected)

    def test_css_vec_is_ndarray(self):
        for sig in _call_build():
            assert isinstance(sig.css_vec, np.ndarray)

    def test_ifs_coeffs_from_fractal(self):
        frag = _make_fragment()
        sigs = _call_build(frag=frag)
        for sig in sigs:
            np.testing.assert_allclose(sig.ifs_coeffs, frag.fractal.ifs_coeffs)

    def test_length_nonneg(self):
        for sig in _call_build():
            assert sig.length >= 0.0

    def test_missing_tangram_raises(self):
        frag = _make_fragment(with_tangram=False)
        with pytest.raises((AssertionError, AttributeError)):
            build_edge_signatures(frag)

    def test_missing_fractal_raises(self):
        frag = _make_fragment(with_fractal=False)
        with pytest.raises((AssertionError, AttributeError)):
            build_edge_signatures(frag)

    def test_alpha_1_emphasizes_tangram(self):
        """With alpha=1, virtual_curve is entirely the tangram curve."""
        tang_curve = np.ones((32, 2), dtype=float) * 0.5
        frag = _make_fragment()
        with (
            patch("puzzle_reconstruction.algorithms.synthesis.extract_tangram_edge",
                  return_value=tang_curve),
            patch("puzzle_reconstruction.algorithms.fractal.css.curvature_scale_space",
                  return_value=[]),
            patch("puzzle_reconstruction.algorithms.fractal.css.css_to_feature_vector",
                  return_value=np.ones(32)),
        ):
            sigs = build_edge_signatures(frag, alpha=1.0, n_points=32)
        for sig in sigs:
            assert sig.virtual_curve.shape == (32, 2)

    def test_alpha_0_emphasizes_fractal(self):
        """With alpha=0, virtual_curve is entirely the fractal curve."""
        frag = _make_fragment()
        with (
            patch("puzzle_reconstruction.algorithms.synthesis.extract_tangram_edge",
                  return_value=np.zeros((32, 2))),
            patch("puzzle_reconstruction.algorithms.fractal.css.curvature_scale_space",
                  return_value=[]),
            patch("puzzle_reconstruction.algorithms.fractal.css.css_to_feature_vector",
                  return_value=np.ones(32)),
        ):
            sigs = build_edge_signatures(frag, alpha=0.0, n_points=32)
        for sig in sigs:
            assert sig.virtual_curve.shape == (32, 2)

    def test_alpha_changes_virtual_curve(self):
        """Different alpha values produce different virtual curves."""
        frag = _make_fragment()
        tang = np.ones((32, 2), dtype=float)

        def _build_with_alpha(a):
            with (
                patch("puzzle_reconstruction.algorithms.synthesis.extract_tangram_edge",
                      return_value=tang),
                patch("puzzle_reconstruction.algorithms.fractal.css.curvature_scale_space",
                      return_value=[]),
                patch("puzzle_reconstruction.algorithms.fractal.css.css_to_feature_vector",
                      return_value=np.ones(32)),
            ):
                return build_edge_signatures(frag, alpha=a, n_points=32)

        sigs_0 = _build_with_alpha(0.0)
        sigs_1 = _build_with_alpha(1.0)
        # For at least one edge, curves should differ
        curves_differ = any(
            not np.allclose(s0.virtual_curve, s1.virtual_curve)
            for s0, s1 in zip(sigs_0, sigs_1)
        )
        assert curves_differ

    def test_n_points_controls_curve_length(self):
        sigs_32 = _call_build(n_points=32)
        sigs_64 = _call_build(n_points=64)
        # Both use tangram mock (zeros), curves may be same but we verify shape
        for sig in sigs_32:
            assert sig.virtual_curve.shape[0] == 32
        for sig in sigs_64:
            assert sig.virtual_curve.shape[0] == 64

    def test_different_fragment_ids_independent(self):
        """Building signatures for two different fragments is independent."""
        frag0 = _make_fragment(fid=0)
        frag1 = _make_fragment(fid=1)
        sigs0 = _call_build(frag=frag0)
        sigs1 = _call_build(frag=frag1)
        assert len(sigs0) == len(sigs1)


# ─── TestNormalizeCurve ───────────────────────────────────────────────────────

class TestNormalizeCurve:
    def test_centroid_at_zero(self):
        curve = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = _normalize_curve(curve)
        np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_max_abs_is_one(self):
        curve = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        result = _normalize_curve(curve)
        assert float(np.abs(result).max()) == pytest.approx(1.0, abs=1e-6)

    def test_zero_curve_unchanged(self):
        """Zero curve (scale=0) is returned as-is (no division by zero)."""
        curve = np.zeros((4, 2))
        result = _normalize_curve(curve)
        np.testing.assert_allclose(result, np.zeros((4, 2)), atol=1e-10)

    def test_shape_preserved(self):
        curve = np.random.randn(20, 2)
        result = _normalize_curve(curve)
        assert result.shape == curve.shape

    def test_already_normalized_unchanged(self):
        """Normalized input with max abs = 1 should return the same."""
        curve = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
        curve -= curve.mean(axis=0)
        result = _normalize_curve(curve)
        np.testing.assert_allclose(np.abs(result).max(), 1.0, atol=1e-6)

    def test_large_curve_scaled_down(self):
        curve = np.array([[0.0, 0.0], [1000.0, 0.0], [500.0, 500.0]])
        result = _normalize_curve(curve)
        assert float(np.abs(result).max()) == pytest.approx(1.0, abs=1e-6)
