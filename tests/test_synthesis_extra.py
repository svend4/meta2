"""Extra tests for puzzle_reconstruction.algorithms.synthesis."""
import pytest
import numpy as np
from unittest.mock import patch

from puzzle_reconstruction.algorithms.synthesis import (
    compute_fractal_signature,
    build_edge_signatures,
)
from puzzle_reconstruction.models import (
    FractalSignature,
    TangramSignature,
    EdgeSignature,
    EdgeSide,
    Fragment,
    ShapeClass,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _contour(n=64, r=30.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
    return pts.astype(np.float32)


def _square_contour(side=40.0, n=64):
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
        chain_code="0123",
        curve=_contour(256),
    )


def _tangram():
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    return TangramSignature(
        polygon=polygon,
        shape_class=ShapeClass.RECTANGLE,
        centroid=np.array([0.5, 0.5]),
        angle=0.0, scale=1.0, area=1.0,
    )


def _fragment(with_tangram=True, with_fractal=True):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:54, 10:54] = 255
    return Fragment(
        fragment_id=0,
        image=img, mask=mask,
        contour=_square_contour(),
        tangram=_tangram() if with_tangram else None,
        fractal=_fractal() if with_fractal else None,
    )


_FRACTAL_PATCHES = (
    ("puzzle_reconstruction.algorithms.fractal.box_counting.box_counting_fd",
     1.0),
    ("puzzle_reconstruction.algorithms.fractal.divider.divider_fd",
     1.1),
    ("puzzle_reconstruction.algorithms.fractal.ifs.fit_ifs_coefficients",
     np.ones(8)),
    ("puzzle_reconstruction.algorithms.fractal.css.curvature_scale_space",
     [(1.0, [0.1, 0.5])]),
    ("puzzle_reconstruction.algorithms.fractal.css.css_to_feature_vector",
     np.ones(32)),
    ("puzzle_reconstruction.algorithms.fractal.css.freeman_chain_code",
     "01234567"),
)


def _make_fractal_sig(contour=None):
    if contour is None:
        contour = _contour()
    ctx = []
    for path, rv in _FRACTAL_PATCHES:
        ctx.append(patch(path, return_value=rv))
    with ctx[0], ctx[1], ctx[2], ctx[3], ctx[4], ctx[5]:
        return compute_fractal_signature(contour)


def _build(n_sides=4, n_points=32, alpha=0.5, **kw):
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


# ─── compute_fractal_signature extras ─────────────────────────────────────────

class TestComputeFractalSignatureExtra:
    def test_fd_box_float(self):
        sig = _make_fractal_sig()
        assert isinstance(sig.fd_box, float)

    def test_fd_divider_float(self):
        sig = _make_fractal_sig()
        assert isinstance(sig.fd_divider, float)

    def test_ifs_coeffs_8_elements(self):
        sig = _make_fractal_sig()
        assert len(sig.ifs_coeffs) == 8

    def test_css_image_list_type(self):
        sig = _make_fractal_sig()
        assert isinstance(sig.css_image, list)

    def test_chain_code_string_non_empty(self):
        sig = _make_fractal_sig()
        assert isinstance(sig.chain_code, str)
        assert len(sig.chain_code) > 0

    def test_curve_is_2d_float(self):
        sig = _make_fractal_sig()
        assert sig.curve.ndim == 2
        assert sig.curve.shape[1] == 2

    def test_curve_length_256(self):
        sig = _make_fractal_sig()
        assert sig.curve.shape[0] == 256

    def test_small_contour_8_pts(self):
        sig = _make_fractal_sig(contour=_contour(n=8, r=5.0))
        assert isinstance(sig, FractalSignature)

    def test_large_contour_512_pts(self):
        sig = _make_fractal_sig(contour=_contour(n=512, r=100.0))
        assert isinstance(sig, FractalSignature)

    def test_square_contour(self):
        sig = _make_fractal_sig(contour=_square_contour(side=20.0, n=64))
        assert isinstance(sig, FractalSignature)

    def test_fd_values_stored(self):
        sig = _make_fractal_sig()
        assert sig.fd_box == pytest.approx(1.0)
        assert sig.fd_divider == pytest.approx(1.1)

    def test_different_radii_different_curves(self):
        c1 = _contour(n=64, r=10.0)
        c2 = _contour(n=64, r=80.0)
        with (
            patch("puzzle_reconstruction.algorithms.fractal.box_counting.box_counting_fd",
                  return_value=1.0),
            patch("puzzle_reconstruction.algorithms.fractal.divider.divider_fd",
                  return_value=1.0),
            patch("puzzle_reconstruction.algorithms.fractal.ifs.fit_ifs_coefficients",
                  return_value=np.ones(8)),
            patch("puzzle_reconstruction.algorithms.fractal.css.curvature_scale_space",
                  return_value=[]),
            patch("puzzle_reconstruction.algorithms.fractal.css.css_to_feature_vector",
                  return_value=np.ones(32)),
            patch("puzzle_reconstruction.algorithms.fractal.css.freeman_chain_code",
                  return_value="0"),
        ):
            s1 = compute_fractal_signature(c1)
            s2 = compute_fractal_signature(c2)
        assert not np.allclose(s1.curve, s2.curve)


# ─── build_edge_signatures extras ─────────────────────────────────────────────

class TestBuildEdgeSignaturesExtra:
    def test_two_sides(self):
        result = _build(n_sides=2)
        assert len(result) == 2

    def test_six_sides(self):
        result = _build(n_sides=6)
        assert len(result) == 6

    def test_edge_ids_zero_based(self):
        result = _build(n_sides=4)
        assert [e.edge_id for e in result] == [0, 1, 2, 3]

    def test_all_edge_sides_valid(self):
        result = _build(n_sides=4)
        for sig in result:
            assert isinstance(sig.side, EdgeSide)

    def test_virtual_curve_n_points_32(self):
        result = _build(n_points=32)
        for sig in result:
            assert sig.virtual_curve.shape == (32, 2)

    def test_virtual_curve_n_points_16(self):
        result = _build(n_points=16)
        for sig in result:
            assert sig.virtual_curve.shape == (16, 2)

    def test_css_vec_shape(self):
        result = _build()
        for sig in result:
            assert sig.css_vec.ndim == 1

    def test_length_nonneg_all_edges(self):
        result = _build(n_sides=4)
        for sig in result:
            assert sig.length >= 0.0

    def test_ifs_coeffs_length_8(self):
        result = _build()
        for sig in result:
            assert len(sig.ifs_coeffs) == 8

    def test_alpha_half_produces_valid_curves(self):
        result = _build(alpha=0.5)
        for sig in result:
            assert np.all(np.isfinite(sig.virtual_curve))

    def test_returns_edge_signatures(self):
        result = _build()
        for sig in result:
            assert isinstance(sig, EdgeSignature)

    def test_fragment_id_not_in_edge(self):
        # Edges have edge_id, not fragment_id; just confirm no crash
        result = _build()
        assert all(hasattr(e, "edge_id") for e in result)
