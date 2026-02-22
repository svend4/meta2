"""Tests for puzzle_reconstruction/models.py — core data structures."""
import pytest
import numpy as np

from puzzle_reconstruction.models import (
    ShapeClass,
    EdgeSide,
    FractalSignature,
    TangramSignature,
    EdgeSignature,
    Fragment,
    CompatEntry,
    Assembly,
)


# ─── ShapeClass ───────────────────────────────────────────────────────────────

class TestShapeClass:
    def test_is_str_enum(self):
        assert isinstance(ShapeClass.TRIANGLE, str)

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

    def test_seven_members(self):
        assert len(ShapeClass) == 7

    def test_lookup_by_value(self):
        assert ShapeClass("triangle") is ShapeClass.TRIANGLE

    def test_lookup_rectangle_by_value(self):
        assert ShapeClass("rectangle") is ShapeClass.RECTANGLE

    def test_members_are_strings(self):
        for sc in ShapeClass:
            assert isinstance(sc, str)


# ─── EdgeSide ─────────────────────────────────────────────────────────────────

class TestEdgeSide:
    def test_is_str_enum(self):
        assert isinstance(EdgeSide.TOP, str)

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

    def test_five_members(self):
        assert len(EdgeSide) == 5

    def test_lookup_by_value(self):
        assert EdgeSide("left") is EdgeSide.LEFT

    def test_lookup_right(self):
        assert EdgeSide("right") is EdgeSide.RIGHT

    def test_members_are_strings(self):
        for es in EdgeSide:
            assert isinstance(es, str)


# ─── FractalSignature ─────────────────────────────────────────────────────────

def _make_fractal():
    return FractalSignature(
        fd_box=1.2,
        fd_divider=1.3,
        ifs_coeffs=np.ones(8, dtype=float),
        css_image=[(1.0, [0.1, 0.5])],
        chain_code="012345",
        curve=np.zeros((64, 2), dtype=float),
    )


class TestFractalSignature:
    def test_instantiation(self):
        sig = _make_fractal()
        assert sig is not None

    def test_fd_box_stored(self):
        sig = _make_fractal()
        assert sig.fd_box == pytest.approx(1.2)

    def test_fd_divider_stored(self):
        sig = _make_fractal()
        assert sig.fd_divider == pytest.approx(1.3)

    def test_ifs_coeffs_ndarray(self):
        sig = _make_fractal()
        assert isinstance(sig.ifs_coeffs, np.ndarray)

    def test_css_image_list(self):
        sig = _make_fractal()
        assert isinstance(sig.css_image, list)

    def test_chain_code_str(self):
        sig = _make_fractal()
        assert isinstance(sig.chain_code, str)

    def test_curve_ndarray(self):
        sig = _make_fractal()
        assert isinstance(sig.curve, np.ndarray)

    def test_curve_shape(self):
        sig = _make_fractal()
        assert sig.curve.ndim == 2
        assert sig.curve.shape[1] == 2

    def test_custom_values(self):
        ifs = np.array([0.1, 0.2, 0.3])
        curve = np.zeros((32, 2))
        sig = FractalSignature(
            fd_box=2.5, fd_divider=2.7,
            ifs_coeffs=ifs, css_image=[],
            chain_code="", curve=curve,
        )
        assert sig.fd_box == pytest.approx(2.5)
        assert sig.fd_divider == pytest.approx(2.7)


# ─── TangramSignature ─────────────────────────────────────────────────────────

def _make_tangram():
    return TangramSignature(
        polygon=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
        shape_class=ShapeClass.RECTANGLE,
        centroid=np.array([0.5, 0.5]),
        angle=0.0,
        scale=1.0,
        area=1.0,
    )


class TestTangramSignature:
    def test_instantiation(self):
        sig = _make_tangram()
        assert sig is not None

    def test_shape_class_stored(self):
        sig = _make_tangram()
        assert sig.shape_class is ShapeClass.RECTANGLE

    def test_polygon_ndarray(self):
        sig = _make_tangram()
        assert isinstance(sig.polygon, np.ndarray)

    def test_polygon_shape(self):
        sig = _make_tangram()
        assert sig.polygon.shape == (4, 2)

    def test_centroid_ndarray(self):
        sig = _make_tangram()
        assert isinstance(sig.centroid, np.ndarray)

    def test_centroid_shape(self):
        sig = _make_tangram()
        assert sig.centroid.shape == (2,)

    def test_angle_stored(self):
        sig = TangramSignature(
            polygon=np.zeros((3, 2)),
            shape_class=ShapeClass.TRIANGLE,
            centroid=np.zeros(2),
            angle=1.57,
            scale=0.5,
            area=0.5,
        )
        assert sig.angle == pytest.approx(1.57)

    def test_scale_stored(self):
        sig = _make_tangram()
        assert sig.scale == pytest.approx(1.0)

    def test_area_stored(self):
        sig = _make_tangram()
        assert sig.area == pytest.approx(1.0)

    def test_various_shape_classes(self):
        for sc in ShapeClass:
            sig = TangramSignature(
                polygon=np.zeros((4, 2)),
                shape_class=sc,
                centroid=np.zeros(2),
                angle=0.0, scale=1.0, area=1.0,
            )
            assert sig.shape_class is sc


# ─── EdgeSignature ────────────────────────────────────────────────────────────

def _make_edge(edge_id=0, side=EdgeSide.RIGHT):
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=np.zeros((16, 2)),
        fd=1.3,
        css_vec=np.zeros(32),
        ifs_coeffs=np.zeros(8),
        length=64.0,
    )


class TestEdgeSignature:
    def test_instantiation(self):
        sig = _make_edge()
        assert sig is not None

    def test_edge_id_stored(self):
        sig = _make_edge(edge_id=5)
        assert sig.edge_id == 5

    def test_side_stored(self):
        sig = _make_edge(side=EdgeSide.LEFT)
        assert sig.side is EdgeSide.LEFT

    def test_virtual_curve_ndarray(self):
        sig = _make_edge()
        assert isinstance(sig.virtual_curve, np.ndarray)

    def test_virtual_curve_shape(self):
        sig = _make_edge()
        assert sig.virtual_curve.ndim == 2
        assert sig.virtual_curve.shape[1] == 2

    def test_fd_stored(self):
        sig = _make_edge()
        assert sig.fd == pytest.approx(1.3)

    def test_css_vec_ndarray(self):
        sig = _make_edge()
        assert isinstance(sig.css_vec, np.ndarray)

    def test_ifs_coeffs_ndarray(self):
        sig = _make_edge()
        assert isinstance(sig.ifs_coeffs, np.ndarray)

    def test_length_stored(self):
        sig = _make_edge()
        assert sig.length == pytest.approx(64.0)

    def test_all_edge_sides(self):
        for side in EdgeSide:
            sig = _make_edge(side=side)
            assert sig.side is side

    def test_different_edge_ids(self):
        for eid in (0, 1, 100, 999):
            sig = _make_edge(edge_id=eid)
            assert sig.edge_id == eid


# ─── Fragment ─────────────────────────────────────────────────────────────────

def _make_fragment(fid=0):
    return Fragment(
        fragment_id=fid,
        image=np.zeros((50, 50, 3), dtype=np.uint8),
        mask=np.zeros((50, 50), dtype=np.uint8),
        contour=np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=float),
    )


class TestFragment:
    def test_instantiation(self):
        frag = _make_fragment()
        assert frag is not None

    def test_fragment_id_stored(self):
        frag = _make_fragment(fid=7)
        assert frag.fragment_id == 7

    def test_tangram_default_none(self):
        frag = _make_fragment()
        assert frag.tangram is None

    def test_fractal_default_none(self):
        frag = _make_fragment()
        assert frag.fractal is None

    def test_edges_default_empty_list(self):
        frag = _make_fragment()
        assert frag.edges == []
        assert isinstance(frag.edges, list)

    def test_edges_default_factory_independent(self):
        """Two fragments must NOT share the same edges list."""
        f1 = _make_fragment(0)
        f2 = _make_fragment(1)
        f1.edges.append(_make_edge())
        assert len(f2.edges) == 0

    def test_placed_default_false(self):
        frag = _make_fragment()
        assert frag.placed is False

    def test_position_default_none(self):
        frag = _make_fragment()
        assert frag.position is None

    def test_rotation_default_zero(self):
        frag = _make_fragment()
        assert frag.rotation == 0.0

    def test_image_stored(self):
        frag = _make_fragment()
        assert isinstance(frag.image, np.ndarray)

    def test_mask_stored(self):
        frag = _make_fragment()
        assert isinstance(frag.mask, np.ndarray)

    def test_contour_stored(self):
        frag = _make_fragment()
        assert isinstance(frag.contour, np.ndarray)

    def test_with_tangram(self):
        frag = _make_fragment()
        frag.tangram = _make_tangram()
        assert frag.tangram is not None
        assert frag.tangram.shape_class is ShapeClass.RECTANGLE

    def test_with_edges(self):
        frag = _make_fragment()
        frag.edges = [_make_edge(0), _make_edge(1)]
        assert len(frag.edges) == 2

    def test_placed_can_be_set(self):
        frag = _make_fragment()
        frag.placed = True
        assert frag.placed is True

    def test_position_can_be_set(self):
        frag = _make_fragment()
        frag.position = np.array([10.0, 20.0])
        assert frag.position is not None

    def test_rotation_can_be_set(self):
        frag = _make_fragment()
        frag.rotation = 1.57
        assert frag.rotation == pytest.approx(1.57)


# ─── CompatEntry ──────────────────────────────────────────────────────────────

class TestCompatEntry:
    def _make(self, score=0.8):
        e1 = _make_edge(edge_id=0)
        e2 = _make_edge(edge_id=1)
        return CompatEntry(
            edge_i=e1, edge_j=e2,
            score=score, dtw_dist=0.2,
            css_sim=0.7, fd_diff=0.05, text_score=0.3,
        )

    def test_instantiation(self):
        entry = self._make()
        assert entry is not None

    def test_score_stored(self):
        entry = self._make(score=0.75)
        assert entry.score == pytest.approx(0.75)

    def test_edges_stored(self):
        e1 = _make_edge(edge_id=2)
        e2 = _make_edge(edge_id=3)
        entry = CompatEntry(
            edge_i=e1, edge_j=e2,
            score=0.5, dtw_dist=0.3,
            css_sim=0.4, fd_diff=0.1, text_score=0.0,
        )
        assert entry.edge_i.edge_id == 2
        assert entry.edge_j.edge_id == 3

    def test_dtw_dist_stored(self):
        entry = self._make()
        assert entry.dtw_dist == pytest.approx(0.2)

    def test_css_sim_stored(self):
        entry = self._make()
        assert entry.css_sim == pytest.approx(0.7)

    def test_fd_diff_stored(self):
        entry = self._make()
        assert entry.fd_diff == pytest.approx(0.05)

    def test_text_score_stored(self):
        entry = self._make()
        assert entry.text_score == pytest.approx(0.3)

    def test_score_range(self):
        for score in (0.0, 0.5, 1.0):
            entry = self._make(score=score)
            assert 0.0 <= entry.score <= 1.0


# ─── Assembly ─────────────────────────────────────────────────────────────────

class TestAssembly:
    def _make(self):
        frags = [_make_fragment(0)]
        return Assembly(
            fragments=frags,
            placements={0: (np.array([0.0, 0.0]), 0.0)},
            compat_matrix=np.zeros((4, 4)),
        )

    def test_instantiation(self):
        asm = self._make()
        assert asm is not None

    def test_total_score_default_zero(self):
        asm = self._make()
        assert asm.total_score == 0.0

    def test_ocr_score_default_zero(self):
        asm = self._make()
        assert asm.ocr_score == 0.0

    def test_fragments_stored(self):
        asm = self._make()
        assert len(asm.fragments) == 1

    def test_placements_stored(self):
        asm = self._make()
        assert 0 in asm.placements

    def test_compat_matrix_ndarray(self):
        asm = self._make()
        assert isinstance(asm.compat_matrix, np.ndarray)

    def test_custom_scores(self):
        frags = [_make_fragment(0)]
        asm = Assembly(
            fragments=frags,
            placements={0: (np.array([0.0, 0.0]), 0.0)},
            compat_matrix=np.zeros((4, 4)),
            total_score=0.85,
            ocr_score=0.72,
        )
        assert asm.total_score == pytest.approx(0.85)
        assert asm.ocr_score == pytest.approx(0.72)

    def test_multiple_fragments(self):
        frags = [_make_fragment(i) for i in range(5)]
        placements = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
        asm = Assembly(
            fragments=frags,
            placements=placements,
            compat_matrix=np.zeros((20, 20)),
        )
        assert len(asm.fragments) == 5
        assert len(asm.placements) == 5

    def test_empty_assembly(self):
        asm = Assembly(
            fragments=[],
            placements={},
            compat_matrix=np.zeros((0, 0)),
        )
        assert len(asm.fragments) == 0
        assert asm.total_score == 0.0

    def test_placement_stores_position_and_angle(self):
        frags = [_make_fragment(0)]
        pos = np.array([10.0, 20.0])
        angle = 1.57
        asm = Assembly(
            fragments=frags,
            placements={0: (pos, angle)},
            compat_matrix=np.zeros((4, 4)),
        )
        stored_pos, stored_angle = asm.placements[0]
        np.testing.assert_array_equal(stored_pos, pos)
        assert stored_angle == pytest.approx(angle)
