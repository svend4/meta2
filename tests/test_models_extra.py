"""Extra tests for puzzle_reconstruction/models.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
    FractalSignature,
    ShapeClass,
    TangramSignature,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _edge_sig(edge_id: int = 0, side: EdgeSide = EdgeSide.LEFT) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=np.zeros((8, 2)),
        fd=0.5,
        css_vec=np.zeros(4),
        ifs_coeffs=np.zeros(4),
        length=50.0,
    )


def _fragment(fid: int = 0, h: int = 64, w: int = 64) -> Fragment:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.ones((h, w), dtype=np.uint8)
    contour = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    return Fragment(fragment_id=fid, image=img, mask=mask, contour=contour)


def _assembly(n: int = 2) -> Assembly:
    frags = [_fragment(i) for i in range(n)]
    placements = {i: (np.array([float(i * 70), 0.0]), 0.0) for i in range(n)}
    return Assembly(fragments=frags, placements=placements,
                    compat_matrix=np.zeros((n, n)), total_score=0.75)


def _compat(fid_a: int = 0, fid_b: int = 1, score: float = 0.8) -> CompatEntry:
    ea = _edge_sig(fid_a * 10, EdgeSide.RIGHT)
    eb = _edge_sig(fid_b * 10, EdgeSide.LEFT)
    return CompatEntry(edge_i=ea, edge_j=eb, score=score,
                       dtw_dist=0.1, css_sim=0.9, fd_diff=0.05, text_score=0.7)


# ─── ShapeClass (extra) ───────────────────────────────────────────────────────

class TestShapeClassExtra:
    def test_triangle_value(self):
        assert ShapeClass.TRIANGLE == "triangle"

    def test_rectangle_value(self):
        assert ShapeClass.RECTANGLE == "rectangle"

    def test_trapezoid_value(self):
        assert ShapeClass.TRAPEZOID == "trapezoid"

    def test_parallelogram_value(self):
        assert ShapeClass.PARALLELOGRAM == "parallelogram"

    def test_pentagon_value(self):
        assert ShapeClass.PENTAGON == "pentagon"

    def test_hexagon_value(self):
        assert ShapeClass.HEXAGON == "hexagon"

    def test_polygon_value(self):
        assert ShapeClass.POLYGON == "polygon"

    def test_all_members(self):
        assert len(ShapeClass) == 7

    def test_is_str(self):
        assert isinstance(ShapeClass.TRIANGLE, str)

    def test_comparison_with_string(self):
        assert ShapeClass.RECTANGLE == "rectangle"


# ─── EdgeSide (extra) ─────────────────────────────────────────────────────────

class TestEdgeSideExtra:
    def test_top_value(self):
        assert EdgeSide.TOP == "top"

    def test_bottom_value(self):
        assert EdgeSide.BOTTOM == "bottom"

    def test_left_value(self):
        assert EdgeSide.LEFT == "left"

    def test_right_value(self):
        assert EdgeSide.RIGHT == "right"

    def test_unknown_value(self):
        assert EdgeSide.UNKNOWN == "unknown"

    def test_all_members(self):
        assert len(EdgeSide) == 5

    def test_is_str(self):
        assert isinstance(EdgeSide.LEFT, str)

    def test_from_string(self):
        assert EdgeSide("top") == EdgeSide.TOP


# ─── FractalSignature (extra) ─────────────────────────────────────────────────

class TestFractalSignatureExtra:
    def _make(self):
        return FractalSignature(
            fd_box=1.2,
            fd_divider=1.1,
            ifs_coeffs=np.array([0.5, 0.3]),
            css_image=[(1.0, [0, 5, 10])],
            chain_code="001234567",
            curve=np.random.rand(20, 2),
        )

    def test_fd_box_stored(self):
        fs = self._make()
        assert fs.fd_box == pytest.approx(1.2)

    def test_fd_divider_stored(self):
        fs = self._make()
        assert fs.fd_divider == pytest.approx(1.1)

    def test_ifs_coeffs_stored(self):
        fs = self._make()
        assert fs.ifs_coeffs.shape == (2,)

    def test_css_image_stored(self):
        fs = self._make()
        assert len(fs.css_image) == 1

    def test_chain_code_stored(self):
        fs = self._make()
        assert isinstance(fs.chain_code, str)

    def test_curve_shape(self):
        fs = self._make()
        assert fs.curve.shape[1] == 2


# ─── TangramSignature (extra) ─────────────────────────────────────────────────

class TestTangramSignatureExtra:
    def _make(self):
        return TangramSignature(
            polygon=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
            shape_class=ShapeClass.TRIANGLE,
            centroid=np.array([0.5, 0.33]),
            angle=0.3,
            scale=1.0,
            area=0.5,
        )

    def test_polygon_shape(self):
        ts = self._make()
        assert ts.polygon.shape == (3, 2)

    def test_shape_class_stored(self):
        ts = self._make()
        assert ts.shape_class == ShapeClass.TRIANGLE

    def test_centroid_stored(self):
        ts = self._make()
        assert ts.centroid.shape == (2,)

    def test_angle_stored(self):
        ts = self._make()
        assert ts.angle == pytest.approx(0.3)

    def test_scale_stored(self):
        ts = self._make()
        assert ts.scale == pytest.approx(1.0)

    def test_area_stored(self):
        ts = self._make()
        assert ts.area == pytest.approx(0.5)


# ─── EdgeSignature (extra) ────────────────────────────────────────────────────

class TestEdgeSignatureExtra:
    def test_edge_id_stored(self):
        e = _edge_sig(edge_id=42)
        assert e.edge_id == 42

    def test_side_stored(self):
        e = _edge_sig(side=EdgeSide.TOP)
        assert e.side == EdgeSide.TOP

    def test_virtual_curve_shape(self):
        e = _edge_sig()
        assert e.virtual_curve.ndim == 2
        assert e.virtual_curve.shape[1] == 2

    def test_fd_stored(self):
        e = _edge_sig()
        assert e.fd == pytest.approx(0.5)

    def test_css_vec_stored(self):
        e = _edge_sig()
        assert isinstance(e.css_vec, np.ndarray)

    def test_ifs_coeffs_stored(self):
        e = _edge_sig()
        assert isinstance(e.ifs_coeffs, np.ndarray)

    def test_length_stored(self):
        e = _edge_sig()
        assert e.length == pytest.approx(50.0)

    def test_all_sides_usable(self):
        for side in EdgeSide:
            e = _edge_sig(side=side)
            assert e.side == side


# ─── Fragment (extra) ─────────────────────────────────────────────────────────

class TestFragmentExtra:
    def test_fragment_id_stored(self):
        f = _fragment(fid=5)
        assert f.fragment_id == 5

    def test_image_shape(self):
        f = _fragment(h=32, w=48)
        assert f.image.shape == (32, 48, 3)

    def test_mask_shape(self):
        f = _fragment(h=32, w=48)
        assert f.mask.shape == (32, 48)

    def test_contour_shape(self):
        f = _fragment()
        assert f.contour.ndim == 2
        assert f.contour.shape[1] == 2

    def test_tangram_default_none(self):
        assert _fragment().tangram is None

    def test_fractal_default_none(self):
        assert _fragment().fractal is None

    def test_edges_default_empty(self):
        assert _fragment().edges == []

    def test_placed_default_false(self):
        assert _fragment().placed is False

    def test_position_default_none(self):
        assert _fragment().position is None

    def test_rotation_default_zero(self):
        assert _fragment().rotation == pytest.approx(0.0)

    def test_edges_added(self):
        f = _fragment()
        f.edges.append(_edge_sig())
        assert len(f.edges) == 1

    def test_custom_position(self):
        f = _fragment()
        f.position = np.array([10.0, 20.0])
        assert f.position[0] == pytest.approx(10.0)


# ─── CompatEntry (extra) ──────────────────────────────────────────────────────

class TestCompatEntryExtra:
    def test_edge_i_stored(self):
        c = _compat()
        assert c.edge_i is not None

    def test_edge_j_stored(self):
        c = _compat()
        assert c.edge_j is not None

    def test_score_stored(self):
        c = _compat(score=0.65)
        assert c.score == pytest.approx(0.65)

    def test_dtw_dist_stored(self):
        c = _compat()
        assert c.dtw_dist == pytest.approx(0.1)

    def test_css_sim_stored(self):
        c = _compat()
        assert c.css_sim == pytest.approx(0.9)

    def test_fd_diff_stored(self):
        c = _compat()
        assert c.fd_diff == pytest.approx(0.05)

    def test_text_score_stored(self):
        c = _compat()
        assert c.text_score == pytest.approx(0.7)

    def test_edge_sides_independent(self):
        c = _compat(fid_a=0, fid_b=1)
        assert c.edge_i.side == EdgeSide.RIGHT
        assert c.edge_j.side == EdgeSide.LEFT


# ─── Assembly (extra) ─────────────────────────────────────────────────────────

class TestAssemblyExtra:
    def test_fragments_stored(self):
        a = _assembly(3)
        assert len(a.fragments) == 3

    def test_placements_stored(self):
        a = _assembly(2)
        assert len(a.placements) == 2

    def test_total_score_stored(self):
        a = _assembly(2)
        assert a.total_score == pytest.approx(0.75)

    def test_ocr_score_default_zero(self):
        a = _assembly(2)
        assert a.ocr_score == pytest.approx(0.0)

    def test_compat_matrix_stored(self):
        a = _assembly(2)
        assert a.compat_matrix.shape == (2, 2)

    def test_placement_contains_pos_and_angle(self):
        a = _assembly(2)
        pos, angle = a.placements[0]
        assert isinstance(pos, np.ndarray)
        assert isinstance(angle, float)

    def test_all_fragment_ids_in_placements(self):
        a = _assembly(3)
        for f in a.fragments:
            assert f.fragment_id in a.placements

    def test_empty_assembly(self):
        a = Assembly(fragments=[], placements={},
                     compat_matrix=np.array([]), total_score=0.0)
        assert len(a.fragments) == 0

    def test_custom_ocr_score(self):
        frags = [_fragment(0)]
        a = Assembly(fragments=frags, placements={},
                     compat_matrix=np.array([]), ocr_score=0.9)
        assert a.ocr_score == pytest.approx(0.9)
