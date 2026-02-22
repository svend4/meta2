"""Тесты для puzzle_reconstruction/ui/viewer.py.

Покрывает не-GUI части: константы, инициализацию, вспомогательные методы
(цвет уверенности, мировые координаты, история отмены, выбор фрагмента).
Интерактивный event-loop (run()) не тестируется.
"""
import pytest
import numpy as np

from puzzle_reconstruction.ui.viewer import (
    AssemblyViewer,
    COLOR_HIGH,
    COLOR_MED,
    COLOR_LOW,
    COLOR_SELECT,
)
from puzzle_reconstruction.models import Assembly, Fragment, EdgeSignature, EdgeSide


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_fragment(fid: int, with_contour: bool = True) -> Fragment:
    contour = np.array([[0, 0], [80, 0], [80, 80], [0, 80]], dtype=np.float64) if with_contour else None
    return Fragment(
        fragment_id=fid,
        image=np.zeros((80, 80, 3), dtype=np.uint8),
        mask=np.zeros((80, 80), dtype=np.uint8),
        contour=contour if with_contour else np.zeros((0, 2)),
    )


def _make_assembly(n: int = 3) -> Assembly:
    frags = [_make_fragment(i) for i in range(n)]
    placements = {i: (np.array([float(i * 100), 0.0]), 0.0) for i in range(n)}
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=np.eye(n, dtype=np.float32),
        total_score=0.5,
        ocr_score=0.8,
    )


def _make_viewer(assembly=None) -> AssemblyViewer:
    asm = assembly or _make_assembly()
    return AssemblyViewer(asm, scale=1.0, canvas_size=(800, 600))


# ─── Цветовые константы ───────────────────────────────────────────────────────

class TestColorConstants:
    def test_color_high_is_tuple_of_3(self):
        assert len(COLOR_HIGH) == 3

    def test_color_med_is_tuple_of_3(self):
        assert len(COLOR_MED) == 3

    def test_color_low_is_tuple_of_3(self):
        assert len(COLOR_LOW) == 3

    def test_color_select_is_tuple_of_3(self):
        assert len(COLOR_SELECT) == 3

    def test_color_high_values_in_range(self):
        for v in COLOR_HIGH:
            assert 0 <= v <= 255

    def test_color_med_values_in_range(self):
        for v in COLOR_MED:
            assert 0 <= v <= 255

    def test_color_low_values_in_range(self):
        for v in COLOR_LOW:
            assert 0 <= v <= 255

    def test_color_select_values_in_range(self):
        for v in COLOR_SELECT:
            assert 0 <= v <= 255

    def test_colors_distinct(self):
        colors = [COLOR_HIGH, COLOR_MED, COLOR_LOW, COLOR_SELECT]
        assert len(set(colors)) == 4


# ─── AssemblyViewer.__init__ ──────────────────────────────────────────────────

class TestViewerInit:
    def test_assembly_stored(self):
        asm = _make_assembly()
        v = AssemblyViewer(asm)
        assert v.assembly is asm

    def test_default_scale(self):
        v = _make_viewer()
        assert v.scale == pytest.approx(1.0)

    def test_custom_scale(self):
        asm = _make_assembly()
        v = AssemblyViewer(asm, scale=0.5)
        assert v.scale == pytest.approx(0.5)

    def test_canvas_dimensions_stored(self):
        asm = _make_assembly()
        v = AssemblyViewer(asm, canvas_size=(1200, 800))
        assert v.canvas_w == 1200
        assert v.canvas_h == 800

    def test_output_path_stored(self):
        asm = _make_assembly()
        v = AssemblyViewer(asm, output_path="test_out.png")
        assert v.output_path == "test_out.png"

    def test_offset_is_50_50(self):
        v = _make_viewer()
        np.testing.assert_array_equal(v.offset, [50.0, 50.0])

    def test_drag_fid_none(self):
        v = _make_viewer()
        assert v._drag_fid is None

    def test_running_false(self):
        v = _make_viewer()
        assert v._running is False

    def test_history_empty(self):
        v = _make_viewer()
        assert v._history == []

    def test_cache_empty(self):
        v = _make_viewer()
        assert v._cache == {}

    def test_window_title_is_string(self):
        assert isinstance(AssemblyViewer.WINDOW, str)


# ─── _confidence_color ────────────────────────────────────────────────────────

class TestConfidenceColor:
    def test_high_score_returns_color_high(self):
        assert AssemblyViewer._confidence_color(0.9) == COLOR_HIGH

    def test_boundary_high_returns_color_high(self):
        assert AssemblyViewer._confidence_color(0.86) == COLOR_HIGH

    def test_medium_score_returns_color_med(self):
        assert AssemblyViewer._confidence_color(0.75) == COLOR_MED

    def test_boundary_medium_exact_returns_color_med(self):
        assert AssemblyViewer._confidence_color(0.66) == COLOR_MED

    def test_low_score_returns_color_low(self):
        assert AssemblyViewer._confidence_color(0.3) == COLOR_LOW

    def test_zero_score_returns_color_low(self):
        assert AssemblyViewer._confidence_color(0.0) == COLOR_LOW

    def test_boundary_at_065_returns_low(self):
        assert AssemblyViewer._confidence_color(0.65) == COLOR_LOW

    def test_boundary_at_085_returns_high(self):
        assert AssemblyViewer._confidence_color(0.85) == COLOR_MED

    def test_returns_tuple(self):
        result = AssemblyViewer._confidence_color(0.5)
        assert isinstance(result, tuple)
        assert len(result) == 3


# ─── _world_to_screen / _screen_to_world ──────────────────────────────────────

class TestCoordinateConversion:
    def test_world_to_screen_offset_added(self):
        v = _make_viewer()
        pos = np.array([0.0, 0.0])
        result = v._world_to_screen(pos)
        # result = pos * scale + offset = [0,0]*1 + [50,50] = [50,50]
        np.testing.assert_allclose(result, [50.0, 50.0])

    def test_world_to_screen_scale_applied(self):
        asm = _make_assembly()
        v = AssemblyViewer(asm, scale=2.0, canvas_size=(800, 600))
        pos = np.array([10.0, 20.0])
        result = v._world_to_screen(pos)
        expected = pos * 2.0 + v.offset
        np.testing.assert_allclose(result, expected)

    def test_screen_to_world_inverse(self):
        v = _make_viewer()
        world = np.array([100.0, 150.0])
        screen = v._world_to_screen(world)
        recovered = v._screen_to_world(screen)
        np.testing.assert_allclose(recovered, world, atol=1e-9)

    def test_screen_to_world_scale_applied(self):
        asm = _make_assembly()
        v = AssemblyViewer(asm, scale=2.0, canvas_size=(800, 600))
        screen = np.array([150.0, 200.0])
        result = v._screen_to_world(screen)
        expected = (screen - v.offset) / 2.0
        np.testing.assert_allclose(result, expected)


# ─── _fragment_confidence ─────────────────────────────────────────────────────

class TestFragmentConfidence:
    def test_missing_fragment_returns_half(self):
        v = _make_viewer()
        result = v._fragment_confidence(999)  # non-existent frag
        assert result == pytest.approx(0.5)

    def test_fragment_no_edges_returns_half(self):
        asm = _make_assembly(n=2)
        # Remove edges from fragment 0
        asm.fragments[0].edges = []
        v = AssemblyViewer(asm)
        result = v._fragment_confidence(0)
        assert result == pytest.approx(0.5)

    def test_returns_float(self):
        v = _make_viewer()
        result = v._fragment_confidence(0)
        assert isinstance(result, float)

    def test_proportional_to_total_score(self):
        asm = _make_assembly(n=3)
        asm.total_score = 1.5
        v = AssemblyViewer(asm)
        # confidence = total_score / max(1, n_frags) = 1.5 / 3 = 0.5
        result = v._fragment_confidence(0)
        assert result == pytest.approx(0.5)


# ─── _pick_fragment ───────────────────────────────────────────────────────────

class TestPickFragment:
    def test_empty_assembly_returns_none(self):
        asm = Assembly(fragments=[], placements={},
                       compat_matrix=np.zeros((0, 0)))
        v = AssemblyViewer(asm)
        result = v._pick_fragment(np.array([0.0, 0.0]))
        assert result is None

    def test_hit_fragment_with_contour(self):
        asm = _make_assembly(n=1)
        # Fragment 0 placed at (0, 0), contour spans [0..80, 0..80]
        asm.placements[0] = (np.array([0.0, 0.0]), 0.0)
        v = AssemblyViewer(asm, scale=1.0)
        # Query point inside the bounding box of contour
        result = v._pick_fragment(np.array([40.0, 40.0]))
        assert result == 0

    def test_miss_returns_none(self):
        asm = _make_assembly(n=1)
        asm.placements[0] = (np.array([0.0, 0.0]), 0.0)
        v = AssemblyViewer(asm)
        # Far outside the contour
        result = v._pick_fragment(np.array([500.0, 500.0]))
        assert result is None

    def test_no_contour_uses_radius(self):
        asm = Assembly(
            fragments=[Fragment(
                fragment_id=0,
                image=np.zeros((32, 32, 3), dtype=np.uint8),
                mask=np.zeros((32, 32), dtype=np.uint8),
                contour=np.zeros((0, 2)),  # empty contour
            )],
            placements={0: (np.array([0.0, 0.0]), 0.0)},
            compat_matrix=np.eye(1, dtype=np.float32),
        )
        v = AssemblyViewer(asm)
        # Within 60px of fragment center
        result = v._pick_fragment(np.array([30.0, 0.0]))
        assert result == 0


# ─── _push_history / _undo ────────────────────────────────────────────────────

class TestHistory:
    def test_push_adds_to_history(self):
        v = _make_viewer()
        assert len(v._history) == 0
        v._push_history()
        assert len(v._history) == 1

    def test_undo_restores_placements(self):
        v = _make_viewer()
        original = {fid: (pos.copy(), angle)
                    for fid, (pos, angle) in v.assembly.placements.items()}
        v._push_history()
        # Modify placements
        v.assembly.placements[0] = (np.array([999.0, 999.0]), 0.0)
        v._undo()
        for fid in original:
            np.testing.assert_allclose(
                v.assembly.placements[fid][0],
                original[fid][0],
                atol=1e-9,
            )

    def test_undo_empty_history_no_error(self):
        v = _make_viewer()
        v._undo()  # Should not raise

    def test_undo_pops_from_history(self):
        v = _make_viewer()
        v._push_history()
        v._push_history()
        assert len(v._history) == 2
        v._undo()
        assert len(v._history) == 1

    def test_history_capped_at_20(self):
        v = _make_viewer()
        for _ in range(25):
            v._push_history()
        assert len(v._history) <= 20

    def test_undo_clears_cache(self):
        v = _make_viewer()
        v._cache[0] = np.zeros((10, 10, 3), dtype=np.uint8)
        v._push_history()
        v._undo()
        assert v._cache == {}
