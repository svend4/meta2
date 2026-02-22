"""Тесты для puzzle_reconstruction.assembly.canvas_builder."""
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

def _gray_img(h=20, w=20, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr_img(h=20, w=20, val=(0, 128, 255)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = val
    return img


def _place(fid, x, y, h=20, w=20, color=128):
    return FragmentPlacement(fragment_id=fid, image=_gray_img(h, w, color), x=x, y=y)


def _simple_placements():
    return [
        _place(0, x=0,  y=0,  w=30, h=20),
        _place(1, x=30, y=0,  w=30, h=20),
        _place(2, x=0,  y=20, w=30, h=20),
    ]


# ─── TestCanvasConfig ─────────────────────────────────────────────────────────

class TestCanvasConfig:
    def test_defaults(self):
        cfg = CanvasConfig()
        assert cfg.bg_color == (255, 255, 255)
        assert cfg.blend_mode == "overwrite"
        assert cfg.padding == 0
        assert cfg.dtype == "uint8"

    def test_invalid_bg_color_channel(self):
        with pytest.raises(ValueError):
            CanvasConfig(bg_color=(256, 0, 0))

    def test_negative_bg_color_channel(self):
        with pytest.raises(ValueError):
            CanvasConfig(bg_color=(0, -1, 0))

    def test_invalid_blend_mode(self):
        with pytest.raises(ValueError):
            CanvasConfig(blend_mode="multiply")

    def test_negative_padding_raises(self):
        with pytest.raises(ValueError):
            CanvasConfig(padding=-1)

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError):
            CanvasConfig(dtype="float64")

    def test_valid_average_mode(self):
        cfg = CanvasConfig(blend_mode="average")
        assert cfg.blend_mode == "average"

    def test_valid_float32_dtype(self):
        cfg = CanvasConfig(dtype="float32")
        assert cfg.dtype == "float32"


# ─── TestFragmentPlacement ────────────────────────────────────────────────────

class TestFragmentPlacement:
    def test_basic_construction(self):
        fp = _place(0, x=5, y=10)
        assert fp.fragment_id == 0
        assert fp.x == 5
        assert fp.y == 10

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacement(fragment_id=-1, image=_gray_img(), x=0, y=0)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacement(fragment_id=0, image=_gray_img(), x=-1, y=0)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacement(fragment_id=0, image=_gray_img(), x=0, y=-1)

    def test_4d_image_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacement(
                fragment_id=0,
                image=np.zeros((2, 2, 3, 1), dtype=np.uint8),
                x=0, y=0,
            )

    def test_h_property(self):
        fp = FragmentPlacement(fragment_id=0, image=_gray_img(15, 25), x=0, y=0)
        assert fp.h == 15

    def test_w_property(self):
        fp = FragmentPlacement(fragment_id=0, image=_gray_img(15, 25), x=0, y=0)
        assert fp.w == 25

    def test_x2_property(self):
        fp = _place(0, x=10, y=0, w=20, h=10)
        assert fp.x2 == 30

    def test_y2_property(self):
        fp = _place(0, x=0, y=5, w=20, h=15)
        assert fp.y2 == 20

    def test_bgr_image_accepted(self):
        fp = FragmentPlacement(fragment_id=0, image=_bgr_img(), x=0, y=0)
        assert fp.image.ndim == 3


# ─── TestCanvasResult ─────────────────────────────────────────────────────────

class TestCanvasResult:
    def _make(self):
        return CanvasResult(
            canvas=np.zeros((50, 60, 3), dtype=np.uint8),
            coverage=0.5,
            n_placed=3,
            canvas_w=60,
            canvas_h=50,
        )

    def test_shape_prop(self):
        r = self._make()
        assert r.shape == (50, 60, 3)

    def test_negative_n_placed_raises(self):
        with pytest.raises(ValueError):
            CanvasResult(
                canvas=np.zeros((10, 10, 3), dtype=np.uint8),
                coverage=0.0,
                n_placed=-1,
                canvas_w=10,
                canvas_h=10,
            )

    def test_coverage_gt_1_raises(self):
        with pytest.raises(ValueError):
            CanvasResult(
                canvas=np.zeros((10, 10, 3), dtype=np.uint8),
                coverage=1.5,
                n_placed=0,
                canvas_w=10,
                canvas_h=10,
            )

    def test_canvas_w_lt_1_raises(self):
        with pytest.raises(ValueError):
            CanvasResult(
                canvas=np.zeros((10, 10, 3), dtype=np.uint8),
                coverage=0.0,
                n_placed=0,
                canvas_w=0,
                canvas_h=10,
            )

    def test_canvas_h_lt_1_raises(self):
        with pytest.raises(ValueError):
            CanvasResult(
                canvas=np.zeros((10, 10, 3), dtype=np.uint8),
                coverage=0.0,
                n_placed=0,
                canvas_w=10,
                canvas_h=0,
            )


# ─── TestComputeCanvasSize ────────────────────────────────────────────────────

class TestComputeCanvasSize:
    def test_basic(self):
        pls = _simple_placements()
        w, h = compute_canvas_size(pls)
        assert w == 60
        assert h == 40

    def test_with_padding(self):
        pls = [_place(0, x=0, y=0, w=10, h=10)]
        w, h = compute_canvas_size(pls, padding=5)
        assert w == 15
        assert h == 15

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_canvas_size([])

    def test_negative_padding_raises(self):
        with pytest.raises(ValueError):
            compute_canvas_size([_place(0, x=0, y=0)], padding=-1)

    def test_single_placement(self):
        pl = _place(0, x=5, y=3, w=20, h=15)
        w, h = compute_canvas_size([pl])
        assert w == 25
        assert h == 18


# ─── TestMakeEmptyCanvas ──────────────────────────────────────────────────────

class TestMakeEmptyCanvas:
    def test_shape(self):
        c = make_empty_canvas(100, 80)
        assert c.shape == (80, 100, 3)

    def test_bg_color_default(self):
        c = make_empty_canvas(10, 10)
        assert np.all(c == 255)

    def test_custom_bg_color(self):
        cfg = CanvasConfig(bg_color=(0, 0, 0))
        c = make_empty_canvas(10, 10, cfg)
        assert np.all(c == 0)

    def test_dtype_uint8(self):
        c = make_empty_canvas(10, 10)
        assert c.dtype == np.uint8

    def test_dtype_float32(self):
        cfg = CanvasConfig(dtype="float32")
        c = make_empty_canvas(10, 10, cfg)
        assert c.dtype == np.float32

    def test_width_zero_raises(self):
        with pytest.raises(ValueError):
            make_empty_canvas(0, 10)

    def test_height_zero_raises(self):
        with pytest.raises(ValueError):
            make_empty_canvas(10, 0)


# ─── TestPlaceFragment ────────────────────────────────────────────────────────

class TestPlaceFragment:
    def test_overwrites_region(self):
        canvas = make_empty_canvas(40, 40)
        fp = FragmentPlacement(fragment_id=0, image=_gray_img(10, 10, 0), x=5, y=5)
        place_fragment(canvas, fp)
        assert int(canvas[5, 5, 0]) == 0

    def test_returns_same_canvas(self):
        canvas = make_empty_canvas(40, 40)
        fp = _place(0, x=0, y=0)
        result = place_fragment(canvas, fp)
        assert result is canvas

    def test_invalid_blend_mode_raises(self):
        canvas = make_empty_canvas(40, 40)
        fp = _place(0, x=0, y=0)
        with pytest.raises(ValueError):
            place_fragment(canvas, fp, blend_mode="multiply")

    def test_average_blend(self):
        canvas = make_empty_canvas(40, 40, CanvasConfig(bg_color=(0, 0, 0)))
        fp = FragmentPlacement(fragment_id=0, image=_gray_img(10, 10, 200), x=0, y=0)
        place_fragment(canvas, fp, blend_mode="average")
        # avg(0, 200) = 100
        assert int(canvas[0, 0, 0]) == 100

    def test_clipped_outside_right(self):
        canvas = make_empty_canvas(20, 20)
        fp = FragmentPlacement(fragment_id=0, image=_gray_img(10, 30, 0), x=15, y=0)
        place_fragment(canvas, fp)  # Should not raise, clips to canvas width

    def test_gray_image_converted(self):
        canvas = make_empty_canvas(40, 40)
        fp = FragmentPlacement(fragment_id=0, image=_gray_img(10, 10, 50), x=0, y=0)
        place_fragment(canvas, fp)
        assert canvas.ndim == 3


# ─── TestBuildCanvas ──────────────────────────────────────────────────────────

class TestBuildCanvas:
    def test_returns_canvas_result(self):
        result = build_canvas(_simple_placements())
        assert isinstance(result, CanvasResult)

    def test_n_placed_matches_input(self):
        pls = _simple_placements()
        result = build_canvas(pls)
        assert result.n_placed == len(pls)

    def test_auto_size(self):
        pls = _simple_placements()
        result = build_canvas(pls)
        assert result.canvas_w == 60
        assert result.canvas_h == 40

    def test_explicit_size(self):
        result = build_canvas([_place(0, x=0, y=0)], canvas_w=100, canvas_h=80)
        assert result.canvas_w == 100
        assert result.canvas_h == 80

    def test_coverage_gt_0(self):
        result = build_canvas(_simple_placements(), canvas_w=60, canvas_h=40)
        assert result.coverage > 0.0

    def test_full_coverage(self):
        pls = [_place(0, x=0, y=0, w=50, h=50)]
        result = build_canvas(pls, canvas_w=50, canvas_h=50)
        assert abs(result.coverage - 1.0) < 1e-9

    def test_empty_placements_raises(self):
        with pytest.raises(ValueError):
            build_canvas([])

    def test_canvas_dtype_uint8(self):
        result = build_canvas(_simple_placements())
        assert result.canvas.dtype == np.uint8


# ─── TestCropToContent ────────────────────────────────────────────────────────

class TestCropToContent:
    def test_no_crop_when_full(self):
        pls = [_place(0, x=0, y=0, w=50, h=50, color=0)]
        result = build_canvas(pls, canvas_w=50, canvas_h=50,
                               cfg=CanvasConfig(bg_color=(255, 255, 255)))
        cropped = crop_to_content(result)
        assert cropped.shape[0] > 0

    def test_all_background_returns_original(self):
        result = CanvasResult(
            canvas=np.full((20, 20, 3), 255, dtype=np.uint8),
            coverage=0.0,
            n_placed=0,
            canvas_w=20,
            canvas_h=20,
        )
        cropped = crop_to_content(result)
        assert cropped.shape == result.canvas.shape

    def test_content_smaller_than_canvas(self):
        canvas = np.full((40, 40, 3), 255, dtype=np.uint8)
        canvas[10:20, 10:20] = 0  # black square in center
        result = CanvasResult(canvas=canvas, coverage=0.25,
                               n_placed=1, canvas_w=40, canvas_h=40)
        cropped = crop_to_content(result, bg_color=(255, 255, 255))
        assert cropped.shape[0] <= 40
        assert cropped.shape[1] <= 40


# ─── TestBatchBuildCanvases ───────────────────────────────────────────────────

class TestBatchBuildCanvases:
    def test_empty_batch_raises_or_empty(self):
        # batch_build_canvases of empty list should return []
        result = batch_build_canvases([])
        assert result == []

    def test_single_list(self):
        result = batch_build_canvases([_simple_placements()])
        assert len(result) == 1
        assert isinstance(result[0], CanvasResult)

    def test_multiple_lists(self):
        batch = [_simple_placements(), [_place(0, x=0, y=0)]]
        result = batch_build_canvases(batch)
        assert len(result) == 2

    def test_custom_cfg(self):
        cfg = CanvasConfig(bg_color=(0, 0, 0))
        result = batch_build_canvases([_simple_placements()], cfg=cfg)
        assert result[0].canvas.dtype == np.uint8
