"""Extra tests for puzzle_reconstruction.ui.viewer."""
import numpy as np
import pytest

from puzzle_reconstruction.ui.viewer import (
    AssemblyViewer,
    COLOR_HIGH,
    COLOR_LOW,
    COLOR_MED,
    COLOR_SELECT,
)
from puzzle_reconstruction.models import Assembly, Fragment


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _frag(fid=0, size=80):
    contour = np.array([[0, 0], [size, 0], [size, size], [0, size]],
                       dtype=np.float64)
    return Fragment(
        fragment_id=fid,
        image=np.zeros((size, size, 3), dtype=np.uint8),
        mask=np.zeros((size, size), dtype=np.uint8),
        contour=contour,
    )


def _asm(n=3):
    frags = [_frag(i) for i in range(n)]
    placements = {i: (np.array([float(i * 100), 0.0]), 0.0) for i in range(n)}
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=np.eye(n, dtype=np.float32),
        total_score=0.5,
        ocr_score=0.8,
    )


def _viewer(assembly=None, **kw):
    return AssemblyViewer(assembly or _asm(), scale=kw.get("scale", 1.0),
                          canvas_size=kw.get("canvas_size", (800, 600)))


# ─── TestColorConstantsExtra ────────────────────────────────────────────────

class TestColorConstantsExtra:
    def test_color_high_tuple(self):
        assert isinstance(COLOR_HIGH, tuple)

    def test_color_med_tuple(self):
        assert isinstance(COLOR_MED, tuple)

    def test_color_low_tuple(self):
        assert isinstance(COLOR_LOW, tuple)

    def test_color_select_tuple(self):
        assert isinstance(COLOR_SELECT, tuple)

    def test_all_ints(self):
        for c in (COLOR_HIGH, COLOR_MED, COLOR_LOW, COLOR_SELECT):
            assert all(isinstance(v, int) for v in c)

    def test_high_not_equal_low(self):
        assert COLOR_HIGH != COLOR_LOW

    def test_select_not_equal_med(self):
        assert COLOR_SELECT != COLOR_MED


# ─── TestViewerInitExtra ────────────────────────────────────────────────────

class TestViewerInitExtra:
    def test_scale_stored(self):
        v = _viewer(scale=2.0)
        assert v.scale == pytest.approx(2.0)

    def test_canvas_w(self):
        v = _viewer(canvas_size=(1024, 768))
        assert v.canvas_w == 1024

    def test_canvas_h(self):
        v = _viewer(canvas_size=(1024, 768))
        assert v.canvas_h == 768

    def test_offset_array(self):
        v = _viewer()
        assert v.offset.shape == (2,)

    def test_drag_fid_none(self):
        assert _viewer()._drag_fid is None

    def test_running_false(self):
        assert _viewer()._running is False

    def test_history_empty(self):
        assert _viewer()._history == []

    def test_cache_empty(self):
        assert _viewer()._cache == {}

    def test_assembly_reference(self):
        asm = _asm()
        v = AssemblyViewer(asm)
        assert v.assembly is asm

    def test_default_output_path(self):
        v = _viewer()
        assert isinstance(v.output_path, str)

    def test_custom_output_path(self):
        asm = _asm()
        v = AssemblyViewer(asm, output_path="/tmp/out.png")
        assert v.output_path == "/tmp/out.png"


# ─── TestConfidenceColorExtra ───────────────────────────────────────────────

class TestConfidenceColorExtra:
    def test_one_returns_high(self):
        assert AssemblyViewer._confidence_color(1.0) == COLOR_HIGH

    def test_zero_returns_low(self):
        assert AssemblyViewer._confidence_color(0.0) == COLOR_LOW

    def test_returns_tuple_of_3(self):
        result = AssemblyViewer._confidence_color(0.5)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_medium_range(self):
        result = AssemblyViewer._confidence_color(0.75)
        assert result == COLOR_MED

    def test_boundary_086_is_high(self):
        assert AssemblyViewer._confidence_color(0.86) == COLOR_HIGH

    def test_boundary_065_is_low(self):
        assert AssemblyViewer._confidence_color(0.65) == COLOR_LOW

    def test_result_values_in_range(self):
        for score in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
            c = AssemblyViewer._confidence_color(score)
            assert all(0 <= v <= 255 for v in c)


# ─── TestCoordinateConversionExtra ──────────────────────────────────────────

class TestCoordinateConversionExtra:
    def test_roundtrip(self):
        v = _viewer()
        world = np.array([42.0, 73.0])
        recovered = v._screen_to_world(v._world_to_screen(world))
        np.testing.assert_allclose(recovered, world, atol=1e-9)

    def test_origin_maps_to_offset(self):
        v = _viewer()
        result = v._world_to_screen(np.array([0.0, 0.0]))
        np.testing.assert_allclose(result, v.offset)

    def test_scale_2(self):
        asm = _asm()
        v = AssemblyViewer(asm, scale=2.0, canvas_size=(800, 600))
        pos = np.array([5.0, 10.0])
        result = v._world_to_screen(pos)
        np.testing.assert_allclose(result, pos * 2.0 + v.offset)

    def test_screen_to_world_inverse_scale_2(self):
        asm = _asm()
        v = AssemblyViewer(asm, scale=2.0, canvas_size=(800, 600))
        screen = np.array([100.0, 200.0])
        expected = (screen - v.offset) / 2.0
        np.testing.assert_allclose(v._screen_to_world(screen), expected)

    def test_negative_world_coords(self):
        v = _viewer()
        world = np.array([-50.0, -100.0])
        screen = v._world_to_screen(world)
        recovered = v._screen_to_world(screen)
        np.testing.assert_allclose(recovered, world, atol=1e-9)


# ─── TestFragmentConfidenceExtra ────────────────────────────────────────────

class TestFragmentConfidenceExtra:
    def test_returns_float(self):
        v = _viewer()
        assert isinstance(v._fragment_confidence(0), float)

    def test_missing_frag_half(self):
        v = _viewer()
        assert v._fragment_confidence(999) == pytest.approx(0.5)

    def test_range(self):
        v = _viewer()
        c = v._fragment_confidence(0)
        assert 0.0 <= c <= 1.0


# ─── TestPickFragmentExtra ──────────────────────────────────────────────────

class TestPickFragmentExtra:
    def test_empty_assembly_none(self):
        asm = Assembly(fragments=[], placements={},
                       compat_matrix=np.zeros((0, 0)))
        v = AssemblyViewer(asm)
        assert v._pick_fragment(np.array([0.0, 0.0])) is None

    def test_hit_inside_contour(self):
        asm = _asm(n=1)
        asm.placements[0] = (np.array([0.0, 0.0]), 0.0)
        v = AssemblyViewer(asm, scale=1.0)
        assert v._pick_fragment(np.array([40.0, 40.0])) == 0

    def test_miss_far_away(self):
        asm = _asm(n=1)
        asm.placements[0] = (np.array([0.0, 0.0]), 0.0)
        v = AssemblyViewer(asm)
        assert v._pick_fragment(np.array([9999.0, 9999.0])) is None

    def test_closest_returned(self):
        asm = _asm(n=2)
        asm.placements[0] = (np.array([0.0, 0.0]), 0.0)
        asm.placements[1] = (np.array([200.0, 0.0]), 0.0)
        v = AssemblyViewer(asm, scale=1.0)
        result = v._pick_fragment(np.array([40.0, 40.0]))
        assert result == 0

    def test_no_contour_radius_fallback(self):
        frag = Fragment(
            fragment_id=0,
            image=np.zeros((32, 32, 3), dtype=np.uint8),
            mask=np.zeros((32, 32), dtype=np.uint8),
            contour=np.zeros((0, 2)),
        )
        asm = Assembly(
            fragments=[frag],
            placements={0: (np.array([0.0, 0.0]), 0.0)},
            compat_matrix=np.eye(1, dtype=np.float32),
        )
        v = AssemblyViewer(asm)
        assert v._pick_fragment(np.array([10.0, 0.0])) == 0


# ─── TestHistoryExtra ───────────────────────────────────────────────────────

class TestHistoryExtra:
    def test_push_increments_len(self):
        v = _viewer()
        v._push_history()
        v._push_history()
        assert len(v._history) == 2

    def test_undo_decrements_len(self):
        v = _viewer()
        v._push_history()
        v._push_history()
        v._undo()
        assert len(v._history) == 1

    def test_undo_empty_no_error(self):
        v = _viewer()
        v._undo()  # should not raise

    def test_undo_restores(self):
        v = _viewer()
        orig = {fid: (pos.copy(), angle)
                for fid, (pos, angle) in v.assembly.placements.items()}
        v._push_history()
        v.assembly.placements[0] = (np.array([888.0, 888.0]), 0.0)
        v._undo()
        for fid in orig:
            np.testing.assert_allclose(
                v.assembly.placements[fid][0], orig[fid][0], atol=1e-9)

    def test_cap_at_20(self):
        v = _viewer()
        for _ in range(30):
            v._push_history()
        assert len(v._history) <= 20

    def test_undo_clears_cache(self):
        v = _viewer()
        v._cache["test"] = 1
        v._push_history()
        v._undo()
        assert v._cache == {}

    def test_multiple_undo(self):
        v = _viewer()
        for _ in range(5):
            v._push_history()
        for _ in range(5):
            v._undo()
        assert len(v._history) == 0

    def test_window_is_string(self):
        assert isinstance(AssemblyViewer.WINDOW, str)
