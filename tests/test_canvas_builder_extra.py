"""Extra tests for puzzle_reconstruction/assembly/canvas_builder.py"""
import numpy as np
import pytest

from puzzle_reconstruction.assembly.canvas_builder import (
    CanvasConfig,
    CanvasResult,
    FragmentPlacement,
    batch_build_canvases,
    build_canvas,
    compute_canvas_size,
    crop_to_content,
    make_empty_canvas,
    place_fragment,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=20, w=20, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=20, w=20, val=(0, 128, 255)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = val
    return img


def _place(fid, x, y, h=20, w=20, color=128):
    return FragmentPlacement(fragment_id=fid, image=_gray(h, w, color), x=x, y=y)


# ─── TestCanvasConfigExtra ────────────────────────────────────────────────────

class TestCanvasConfigExtra:
    def test_black_bg_color_valid(self):
        cfg = CanvasConfig(bg_color=(0, 0, 0))
        assert cfg.bg_color == (0, 0, 0)

    def test_white_bg_color_valid(self):
        cfg = CanvasConfig(bg_color=(255, 255, 255))
        assert cfg.bg_color == (255, 255, 255)

    def test_overwrite_blend_mode(self):
        cfg = CanvasConfig(blend_mode="overwrite")
        assert cfg.blend_mode == "overwrite"

    def test_padding_10_valid(self):
        cfg = CanvasConfig(padding=10)
        assert cfg.padding == 10

    def test_dtype_uint8_valid(self):
        cfg = CanvasConfig(dtype="uint8")
        assert cfg.dtype == "uint8"

    def test_padding_0_valid(self):
        cfg = CanvasConfig(padding=0)
        assert cfg.padding == 0

    def test_average_blend_stored(self):
        cfg = CanvasConfig(blend_mode="average")
        assert cfg.blend_mode == "average"


# ─── TestFragmentPlacementExtra ───────────────────────────────────────────────

class TestFragmentPlacementExtra:
    def test_zero_x_y_valid(self):
        fp = FragmentPlacement(fragment_id=0, image=_gray(), x=0, y=0)
        assert fp.x == 0
        assert fp.y == 0

    def test_large_fragment_id(self):
        fp = FragmentPlacement(fragment_id=9999, image=_gray(), x=0, y=0)
        assert fp.fragment_id == 9999

    def test_x2_large(self):
        fp = FragmentPlacement(fragment_id=0, image=_gray(h=10, w=50), x=100, y=0)
        assert fp.x2 == 150

    def test_y2_large(self):
        fp = FragmentPlacement(fragment_id=0, image=_gray(h=30, w=10), x=0, y=50)
        assert fp.y2 == 80

    def test_h_and_w_correct(self):
        fp = FragmentPlacement(fragment_id=0, image=_gray(h=15, w=25), x=0, y=0)
        assert fp.h == 15
        assert fp.w == 25

    def test_bgr_fragment(self):
        fp = FragmentPlacement(fragment_id=1, image=_bgr(10, 10), x=0, y=0)
        assert fp.image.ndim == 3

    def test_fragment_id_1(self):
        fp = FragmentPlacement(fragment_id=1, image=_gray(), x=5, y=3)
        assert fp.fragment_id == 1


# ─── TestCanvasResultExtra ────────────────────────────────────────────────────

class TestCanvasResultExtra:
    def test_coverage_0_valid(self):
        r = CanvasResult(canvas=np.zeros((10, 10, 3), dtype=np.uint8),
                         coverage=0.0, n_placed=0, canvas_w=10, canvas_h=10)
        assert r.coverage == pytest.approx(0.0)

    def test_coverage_1_valid(self):
        r = CanvasResult(canvas=np.zeros((10, 10, 3), dtype=np.uint8),
                         coverage=1.0, n_placed=1, canvas_w=10, canvas_h=10)
        assert r.coverage == pytest.approx(1.0)

    def test_n_placed_0_valid(self):
        r = CanvasResult(canvas=np.zeros((10, 10, 3), dtype=np.uint8),
                         coverage=0.0, n_placed=0, canvas_w=10, canvas_h=10)
        assert r.n_placed == 0

    def test_shape_gray(self):
        r = CanvasResult(canvas=np.zeros((10, 20), dtype=np.uint8),
                         coverage=0.0, n_placed=0, canvas_w=20, canvas_h=10)
        assert r.shape == (10, 20)

    def test_canvas_w_h_stored(self):
        r = CanvasResult(canvas=np.zeros((30, 40, 3), dtype=np.uint8),
                         coverage=0.5, n_placed=2, canvas_w=40, canvas_h=30)
        assert r.canvas_w == 40
        assert r.canvas_h == 30


# ─── TestComputeCanvasSizeExtra ───────────────────────────────────────────────

class TestComputeCanvasSizeExtra:
    def test_two_side_by_side(self):
        pls = [_place(0, x=0, y=0, w=10, h=10), _place(1, x=10, y=0, w=10, h=10)]
        w, h = compute_canvas_size(pls)
        assert w == 20
        assert h == 10

    def test_three_in_grid(self):
        pls = [_place(0, x=0, y=0, w=10, h=10),
               _place(1, x=10, y=0, w=10, h=10),
               _place(2, x=0, y=10, w=10, h=10)]
        w, h = compute_canvas_size(pls)
        assert w == 20
        assert h == 20

    def test_with_padding_10(self):
        pls = [_place(0, x=0, y=0, w=20, h=20)]
        w, h = compute_canvas_size(pls, padding=10)
        assert w == 30
        assert h == 30

    def test_result_tuple(self):
        pls = [_place(0, x=5, y=3, w=10, h=10)]
        result = compute_canvas_size(pls)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_offset_placement(self):
        # A fragment placed at (50, 50) with size 10x10
        pls = [_place(0, x=50, y=50, w=10, h=10)]
        w, h = compute_canvas_size(pls)
        assert w == 60
        assert h == 60


# ─── TestMakeEmptyCanvasExtra ─────────────────────────────────────────────────

class TestMakeEmptyCanvasExtra:
    def test_large_canvas(self):
        c = make_empty_canvas(200, 150)
        assert c.shape == (150, 200, 3)

    def test_non_square_canvas(self):
        c = make_empty_canvas(30, 50)
        assert c.shape == (50, 30, 3)

    def test_custom_color_stored(self):
        cfg = CanvasConfig(bg_color=(100, 150, 200))
        c = make_empty_canvas(10, 10, cfg)
        assert int(c[0, 0, 0]) == 100
        assert int(c[0, 0, 1]) == 150
        assert int(c[0, 0, 2]) == 200

    def test_float32_canvas_white(self):
        cfg = CanvasConfig(dtype="float32", bg_color=(255, 255, 255))
        c = make_empty_canvas(5, 5, cfg)
        assert c.dtype == np.float32

    def test_dimensions_1x1(self):
        c = make_empty_canvas(1, 1)
        assert c.shape == (1, 1, 3)


# ─── TestPlaceFragmentExtra ───────────────────────────────────────────────────

class TestPlaceFragmentExtra:
    def test_bgr_fragment_on_bgr_canvas(self):
        canvas = make_empty_canvas(40, 40)
        fp = FragmentPlacement(fragment_id=0, image=_bgr(10, 10, (50, 60, 70)), x=5, y=5)
        place_fragment(canvas, fp)
        assert canvas[5, 5, 0] == 50

    def test_fragment_at_origin(self):
        canvas = make_empty_canvas(40, 40, CanvasConfig(bg_color=(255, 255, 255)))
        fp = FragmentPlacement(fragment_id=0, image=_gray(5, 5, 0), x=0, y=0)
        place_fragment(canvas, fp)
        assert int(canvas[0, 0, 0]) == 0

    def test_fragment_at_edge(self):
        canvas = make_empty_canvas(40, 40)
        fp = FragmentPlacement(fragment_id=0, image=_gray(10, 10, 0), x=30, y=30)
        place_fragment(canvas, fp)
        # Should clip without raising
        assert canvas.shape == (40, 40, 3)

    def test_overwrite_second_fragment(self):
        canvas = make_empty_canvas(40, 40, CanvasConfig(bg_color=(0, 0, 0)))
        fp1 = FragmentPlacement(fragment_id=0, image=_gray(10, 10, 100), x=0, y=0)
        fp2 = FragmentPlacement(fragment_id=1, image=_gray(10, 10, 200), x=0, y=0)
        place_fragment(canvas, fp1)
        place_fragment(canvas, fp2)
        assert int(canvas[0, 0, 0]) == 200

    def test_returns_canvas_same_object(self):
        canvas = make_empty_canvas(40, 40)
        fp = _place(0, x=0, y=0)
        result = place_fragment(canvas, fp)
        assert result is canvas


# ─── TestBuildCanvasExtra ─────────────────────────────────────────────────────

class TestBuildCanvasExtra:
    def test_single_fragment(self):
        result = build_canvas([_place(0, x=0, y=0, w=30, h=30)])
        assert isinstance(result, CanvasResult)
        assert result.n_placed == 1

    def test_two_fragments(self):
        pls = [_place(0, x=0, y=0, w=20, h=20), _place(1, x=20, y=0, w=20, h=20)]
        result = build_canvas(pls)
        assert result.n_placed == 2

    def test_coverage_between_0_and_1(self):
        result = build_canvas([_place(0, x=0, y=0, w=20, h=20)], canvas_w=40, canvas_h=40)
        assert 0.0 <= result.coverage <= 1.0

    def test_large_canvas(self):
        result = build_canvas([_place(0, x=0, y=0, w=10, h=10)],
                               canvas_w=100, canvas_h=100)
        assert result.canvas_w == 100
        assert result.canvas_h == 100

    def test_canvas_ndim_3(self):
        result = build_canvas([_place(0, x=0, y=0)])
        assert result.canvas.ndim == 3

    def test_with_custom_cfg(self):
        cfg = CanvasConfig(bg_color=(0, 0, 0))
        result = build_canvas([_place(0, x=0, y=0)], cfg=cfg)
        assert result.canvas.dtype == np.uint8


# ─── TestCropToContentExtra ───────────────────────────────────────────────────

class TestCropToContentExtra:
    def test_returns_ndarray(self):
        pls = [_place(0, x=0, y=0, w=20, h=20, color=0)]
        result = build_canvas(pls, canvas_w=30, canvas_h=30,
                               cfg=CanvasConfig(bg_color=(255, 255, 255)))
        cropped = crop_to_content(result)
        assert isinstance(cropped, np.ndarray)

    def test_cropped_smaller_or_equal(self):
        canvas = np.full((50, 50, 3), 255, dtype=np.uint8)
        canvas[10:20, 10:20] = 0
        r = CanvasResult(canvas=canvas, coverage=0.04, n_placed=1,
                          canvas_w=50, canvas_h=50)
        cropped = crop_to_content(r, bg_color=(255, 255, 255))
        assert cropped.shape[0] <= 50
        assert cropped.shape[1] <= 50

    def test_full_content_no_change(self):
        canvas = np.zeros((20, 20, 3), dtype=np.uint8)
        r = CanvasResult(canvas=canvas, coverage=1.0, n_placed=1,
                          canvas_w=20, canvas_h=20)
        cropped = crop_to_content(r, bg_color=(255, 255, 255))
        assert cropped.shape[0] > 0 and cropped.shape[1] > 0

    def test_single_pixel_content(self):
        canvas = np.full((30, 30, 3), 255, dtype=np.uint8)
        canvas[15, 15] = [0, 0, 0]
        r = CanvasResult(canvas=canvas, coverage=0.001, n_placed=1,
                          canvas_w=30, canvas_h=30)
        cropped = crop_to_content(r, bg_color=(255, 255, 255))
        assert isinstance(cropped, np.ndarray)


# ─── TestBatchBuildCanvasesExtra ─────────────────────────────────────────────

class TestBatchBuildCanvasesExtra:
    def test_three_batches(self):
        batch = [
            [_place(0, x=0, y=0)],
            [_place(0, x=0, y=0), _place(1, x=20, y=0)],
            [_place(0, x=0, y=0)],
        ]
        result = batch_build_canvases(batch)
        assert len(result) == 3

    def test_all_canvas_results(self):
        batch = [[_place(0, x=0, y=0)], [_place(1, x=0, y=0)]]
        result = batch_build_canvases(batch)
        for r in result:
            assert isinstance(r, CanvasResult)

    def test_custom_cfg_applied(self):
        cfg = CanvasConfig(bg_color=(0, 0, 0))
        batch = [[_place(0, x=0, y=0)]]
        result = batch_build_canvases(batch, cfg=cfg)
        assert result[0].canvas.dtype == np.uint8

    def test_varying_sizes_in_batch(self):
        batch = [
            [_place(0, x=0, y=0, w=10, h=10)],
            [_place(0, x=0, y=0, w=20, h=20)],
            [_place(0, x=0, y=0, w=15, h=15)],
        ]
        result = batch_build_canvases(batch)
        assert len(result) == 3
