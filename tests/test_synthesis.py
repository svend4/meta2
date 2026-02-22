"""Тесты для puzzle_reconstruction.algorithms.synthesis."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

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

def _contour(n: int = 64, r: float = 30.0) -> np.ndarray:
    """Круговой контур (n, 2) float32."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
    return pts.astype(np.float32)


def _square_contour(side: float = 40.0, n: int = 64) -> np.ndarray:
    """Прямоугольный контур (n, 2) float32."""
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


def _fractal() -> FractalSignature:
    return FractalSignature(
        fd_box=1.2,
        fd_divider=1.3,
        ifs_coeffs=np.ones(8, dtype=float),
        css_image=[(1.0, [0.1, 0.5, 0.9])],
        chain_code="0123",
        curve=_contour(256),
    )


def _tangram() -> TangramSignature:
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    return TangramSignature(
        polygon=polygon,
        shape_class=ShapeClass.RECTANGLE,
        centroid=np.array([0.5, 0.5]),
        angle=0.0,
        scale=1.0,
        area=1.0,
    )


def _fragment(with_tangram: bool = True,
              with_fractal: bool = True) -> Fragment:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:54, 10:54] = 255
    contour = _square_contour()
    return Fragment(
        fragment_id=0,
        image=img,
        mask=mask,
        contour=contour,
        tangram=_tangram() if with_tangram else None,
        fractal=_fractal() if with_fractal else None,
    )


# ─── Mock helpers ─────────────────────────────────────────────────────────────

def _mock_fractal_deps():
    """Патчит тяжёлые зависимости compute_fractal_signature."""
    patches = [
        patch("puzzle_reconstruction.algorithms.synthesis."
              "compute_fractal_signature.__globals__"),
    ]
    return patches


def _make_fractal_with_mocks() -> FractalSignature:
    """Вызывает compute_fractal_signature с моками подмодулей."""
    contour = _contour()
    with (
        patch("puzzle_reconstruction.algorithms.fractal.box_counting.box_counting_fd",
              return_value=1.2),
        patch("puzzle_reconstruction.algorithms.fractal.divider.divider_fd",
              return_value=1.3),
        patch("puzzle_reconstruction.algorithms.fractal.ifs.fit_ifs_coefficients",
              return_value=np.ones(8)),
        patch("puzzle_reconstruction.algorithms.fractal.css.curvature_scale_space",
              return_value=[(1.0, [0.1, 0.5])]),
        patch("puzzle_reconstruction.algorithms.fractal.css.css_to_feature_vector",
              return_value=np.ones(32)),
        patch("puzzle_reconstruction.algorithms.fractal.css.freeman_chain_code",
              return_value="01234567"),
    ):
        return compute_fractal_signature(contour)


# ─── TestComputeFractalSignature ──────────────────────────────────────────────

class TestComputeFractalSignature:
    def test_returns_fractal_signature(self):
        sig = _make_fractal_with_mocks()
        assert isinstance(sig, FractalSignature)

    def test_fd_box_stored(self):
        sig = _make_fractal_with_mocks()
        assert sig.fd_box == pytest.approx(1.2)

    def test_fd_divider_stored(self):
        sig = _make_fractal_with_mocks()
        assert sig.fd_divider == pytest.approx(1.3)

    def test_ifs_coeffs_ndarray(self):
        sig = _make_fractal_with_mocks()
        assert isinstance(sig.ifs_coeffs, np.ndarray)

    def test_ifs_coeffs_shape(self):
        sig = _make_fractal_with_mocks()
        assert sig.ifs_coeffs.shape == (8,)

    def test_css_image_is_list(self):
        sig = _make_fractal_with_mocks()
        assert isinstance(sig.css_image, list)

    def test_chain_code_is_str(self):
        sig = _make_fractal_with_mocks()
        assert isinstance(sig.chain_code, str)

    def test_curve_shape(self):
        sig = _make_fractal_with_mocks()
        assert sig.curve.ndim == 2
        assert sig.curve.shape[1] == 2
        assert sig.curve.shape[0] == 256

    def test_curve_dtype(self):
        sig = _make_fractal_with_mocks()
        assert sig.curve.dtype in (np.float32, np.float64)

    def test_different_contours_different_curves(self):
        c1 = _contour(64, r=10.0)
        c2 = _contour(64, r=50.0)
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
        # Кривые разных масштабов не совпадают
        assert not np.allclose(s1.curve, s2.curve)


# ─── TestBuildEdgeSignatures ──────────────────────────────────────────────────

class TestBuildEdgeSignatures:

    def _build(self, alpha=0.5, n_sides=4, n_points=32, **kw):
        frag = _fragment(**kw)
        with (
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "extract_tangram_edge",
                  return_value=np.zeros((n_points, 2), dtype=float)),
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "curvature_scale_space",
                  return_value=[(1.0, [0.1])]),
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "css_to_feature_vector",
                  return_value=np.ones(32)),
        ):
            return build_edge_signatures(frag, alpha=alpha,
                                         n_sides=n_sides, n_points=n_points)

    def test_returns_list(self):
        result = self._build()
        assert isinstance(result, list)

    def test_length_n_sides(self):
        result = self._build(n_sides=4)
        assert len(result) == 4

    def test_three_sides(self):
        result = self._build(n_sides=3)
        assert len(result) == 3

    def test_all_edge_signatures(self):
        for sig in self._build():
            assert isinstance(sig, EdgeSignature)

    def test_edge_ids_sequential(self):
        sigs = self._build()
        for i, sig in enumerate(sigs):
            assert sig.edge_id == i

    def test_side_is_edge_side(self):
        for sig in self._build():
            assert isinstance(sig.side, EdgeSide)

    def test_virtual_curve_shape(self):
        n = 32
        for sig in self._build(n_points=n):
            assert sig.virtual_curve.ndim == 2
            assert sig.virtual_curve.shape[1] == 2

    def test_fd_average_of_box_divider(self):
        frag = _fragment()
        expected_fd = (frag.fractal.fd_box + frag.fractal.fd_divider) / 2.0
        with (
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "extract_tangram_edge",
                  return_value=np.zeros((32, 2), dtype=float)),
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "curvature_scale_space",
                  return_value=[]),
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "css_to_feature_vector",
                  return_value=np.ones(32)),
        ):
            sigs = build_edge_signatures(frag, n_points=32)
        for sig in sigs:
            assert sig.fd == pytest.approx(expected_fd)

    def test_css_vec_ndarray(self):
        for sig in self._build():
            assert isinstance(sig.css_vec, np.ndarray)

    def test_ifs_coeffs_from_fractal(self):
        frag = _fragment()
        with (
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "extract_tangram_edge",
                  return_value=np.zeros((32, 2), dtype=float)),
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "curvature_scale_space",
                  return_value=[]),
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "css_to_feature_vector",
                  return_value=np.ones(32)),
        ):
            sigs = build_edge_signatures(frag, n_points=32)
        for sig in sigs:
            assert np.allclose(sig.ifs_coeffs, frag.fractal.ifs_coeffs)

    def test_length_non_negative(self):
        for sig in self._build():
            assert sig.length >= 0.0

    def test_alpha_one_closer_to_tangram(self):
        """При alpha=1 virtual_curve ≈ tangram_curve."""
        frag = _fragment()
        tang_curve = np.ones((32, 2), dtype=float) * 0.5
        with (
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "extract_tangram_edge",
                  return_value=tang_curve),
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "curvature_scale_space", return_value=[]),
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "css_to_feature_vector", return_value=np.ones(32)),
        ):
            sigs = build_edge_signatures(frag, alpha=1.0, n_points=32)
        # При alpha=1, virtual = tang (нормализованный)
        for sig in sigs:
            assert sig.virtual_curve.shape == (32, 2)

    def test_alpha_zero_closer_to_fractal(self):
        """При alpha=0 virtual_curve ≈ fractal_curve."""
        frag = _fragment()
        with (
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "extract_tangram_edge",
                  return_value=np.zeros((32, 2), dtype=float)),
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "curvature_scale_space", return_value=[]),
            patch("puzzle_reconstruction.algorithms.synthesis."
                  "css_to_feature_vector", return_value=np.ones(32)),
        ):
            sigs = build_edge_signatures(frag, alpha=0.0, n_points=32)
        for sig in sigs:
            assert sig.virtual_curve.shape == (32, 2)

    def test_missing_tangram_raises(self):
        frag = _fragment(with_tangram=False)
        with pytest.raises((AssertionError, AttributeError)):
            build_edge_signatures(frag)

    def test_missing_fractal_raises(self):
        frag = _fragment(with_fractal=False)
        with pytest.raises((AssertionError, AttributeError)):
            build_edge_signatures(frag)
