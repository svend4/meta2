"""Tests for puzzle_reconstruction/assembly/canvas_builder.py"""
import pytest
import numpy as np

from puzzle_reconstruction.assembly.canvas_builder import (
    CanvasConfig,
    FragmentPlacement,
    CanvasResult,
    compute_canvas_size,
    make_empty_canvas,
    place_fragment,
    build_canvas,
    crop_to_content,
    batch_build_canvases,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _img(h=10, w=10, color=128, channels=3):
    if channels == 1:
        return np.full((h, w), color, dtype=np.uint8)
    return np.full((h, w, channels), color, dtype=np.uint8)


def _place(fid=0, img=None, x=0, y=0):
    if img is None:
        img = _img()
    return FragmentPlacement(fragment_id=fid, image=img, x=x, y=y)


# ─── TestCanvasConfig ──────────────────────────────────────────────────────────

class TestCanvasConfig:
    def test_defaults(self):
        cfg = CanvasConfig()
        assert cfg.bg_color == (255, 255, 255)
        assert cfg.blend_mode == "overwrite"
        assert cfg.padding == 0
        assert cfg.dtype == "uint8"

    def test_custom_bg_color(self):
        cfg = CanvasConfig(bg_color=(0, 0, 0))
        assert cfg.bg_color == (0, 0, 0)

    def test_bg_color_out_of_range_raises(self):
        with pytest.raises(ValueError):
            CanvasConfig(bg_color=(256, 0, 0))

    def test_bg_color_negative_raises(self):
        with pytest.raises(ValueError):
            CanvasConfig(bg_color=(0, -1, 0))

    def test_invalid_blend_mode_raises(self):
        with pytest.raises(ValueError):
            CanvasConfig(blend_mode="multiply")

    def test_average_blend_mode_ok(self):
        cfg = CanvasConfig(blend_mode="average")
        assert cfg.blend_mode == "average"

    def test_negative_padding_raises(self):
        with pytest.raises(ValueError):
            CanvasConfig(padding=-1)

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError):
            CanvasConfig(dtype="int32")

    def test_float32_dtype_ok(self):
        cfg = CanvasConfig(dtype="float32")
        assert cfg.dtype == "float32"


# ─── TestFragmentPlacement ────────────────────────────────────────────────────

class TestFragmentPlacement:
    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacement(fragment_id=-1, image=_img(), x=0, y=0)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacement(fragment_id=0, image=_img(), x=-1, y=0)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacement(fragment_id=0, image=_img(), x=0, y=-1)

    def test_1d_image_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacement(fragment_id=0, image=np.zeros(10, dtype=np.uint8), x=0, y=0)

    def test_valid_2d_image(self):
        p = FragmentPlacement(fragment_id=0, image=_img(channels=1), x=0, y=0)
        assert p.image.ndim == 2

    def test_h_w_properties(self):
        p = _place(img=_img(h=4, w=6))
        assert p.h == 4
        assert p.w == 6

    def test_x2_y2_properties(self):
        p = _place(img=_img(h=5, w=8), x=3, y=2)
        assert p.x2 == 11   # 3 + 8
        assert p.y2 == 7    # 2 + 5


# ─── TestCanvasResult ─────────────────────────────────────────────────────────

class TestCanvasResult:
    def _result(self, **kwargs):
        defaults = dict(
            canvas=np.zeros((10, 10, 3), dtype=np.uint8),
            coverage=0.5,
            n_placed=1,
            canvas_w=10,
            canvas_h=10,
        )
        defaults.update(kwargs)
        return CanvasResult(**defaults)

    def test_n_placed_negative_raises(self):
        with pytest.raises(ValueError):
            self._result(n_placed=-1)

    def test_coverage_above_one_raises(self):
        with pytest.raises(ValueError):
            self._result(coverage=1.5)

    def test_canvas_w_zero_raises(self):
        with pytest.raises(ValueError):
            self._result(canvas_w=0)

    def test_canvas_h_zero_raises(self):
        with pytest.raises(ValueError):
            self._result(canvas_h=0)

    def test_shape_property(self):
        r = self._result()
        assert r.shape == (10, 10, 3)

    def test_zero_coverage_ok(self):
        r = self._result(coverage=0.0, n_placed=0)
        assert r.coverage == 0.0


# ─── TestComputeCanvasSize ────────────────────────────────────────────────────

class TestComputeCanvasSize:
    def test_empty_placements_raises(self):
        with pytest.raises(ValueError):
            compute_canvas_size([])

    def test_negative_padding_raises(self):
        with pytest.raises(ValueError):
            compute_canvas_size([_place()], padding=-1)

    def test_single_placement(self):
        p = _place(img=_img(h=5, w=8), x=2, y=3)
        w, h = compute_canvas_size([p])
        assert w == 10   # 2 + 8
        assert h == 8    # 3 + 5

    def test_two_placements_takes_max(self):
        p1 = _place(img=_img(h=10, w=10), x=0, y=0)
        p2 = _place(img=_img(h=5, w=5), x=20, y=30)
        w, h = compute_canvas_size([p1, p2])
        assert w == 25
        assert h == 35

    def test_padding_adds_to_size(self):
        p = _place(img=_img(h=10, w=10), x=0, y=0)
        w, h = compute_canvas_size([p], padding=5)
        assert w == 15
        assert h == 15


# ─── TestMakeEmptyCanvas ─────────────────────────────────────────────────────

class TestMakeEmptyCanvas:
    def test_returns_ndarray(self):
        c = make_empty_canvas(10, 10)
        assert isinstance(c, np.ndarray)

    def test_shape_is_hwc(self):
        c = make_empty_canvas(30, 20)
        assert c.shape == (20, 30, 3)

    def test_width_zero_raises(self):
        with pytest.raises(ValueError):
            make_empty_canvas(0, 10)

    def test_height_zero_raises(self):
        with pytest.raises(ValueError):
            make_empty_canvas(10, 0)

    def test_default_bg_white(self):
        c = make_empty_canvas(5, 5)
        assert (c == 255).all()

    def test_custom_bg_color(self):
        cfg = CanvasConfig(bg_color=(0, 0, 0))
        c = make_empty_canvas(5, 5, cfg)
        assert (c == 0).all()

    def test_float32_dtype(self):
        cfg = CanvasConfig(dtype="float32")
        c = make_empty_canvas(5, 5, cfg)
        assert c.dtype == np.float32


# ─── TestPlaceFragment ────────────────────────────────────────────────────────

class TestPlaceFragment:
    def test_invalid_blend_mode_raises(self):
        canvas = make_empty_canvas(20, 20)
        p = _place(img=_img(h=5, w=5, color=0))
        with pytest.raises(ValueError):
            place_fragment(canvas, p, blend_mode="invalid")

    def test_overwrite_places_fragment(self):
        canvas = make_empty_canvas(20, 20)
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=img, x=0, y=0)
        place_fragment(canvas, p)
        assert (canvas[:5, :5] == 0).all()

    def test_returns_same_canvas(self):
        canvas = make_empty_canvas(20, 20)
        p = _place()
        result = place_fragment(canvas, p)
        assert result is canvas

    def test_average_blend_mode(self):
        canvas = np.full((10, 10, 3), 100, dtype=np.uint8)
        img = np.full((5, 5, 3), 200, dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=img, x=0, y=0)
        place_fragment(canvas, p, blend_mode="average")
        # average of 100 and 200 = 150
        assert canvas[0, 0, 0] == 150

    def test_grayscale_image_placed(self):
        canvas = make_empty_canvas(20, 20)
        gray = np.zeros((5, 5), dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=gray, x=2, y=2)
        place_fragment(canvas, p)
        assert (canvas[2:7, 2:7] == 0).all()

    def test_out_of_bounds_clipped(self):
        canvas = make_empty_canvas(10, 10)
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=img, x=8, y=8)
        result = place_fragment(canvas, p)
        assert result.shape == (10, 10, 3)


# ─── TestBuildCanvas ──────────────────────────────────────────────────────────

class TestBuildCanvas:
    def test_empty_placements_raises(self):
        with pytest.raises(ValueError):
            build_canvas([])

    def test_returns_canvas_result(self):
        result = build_canvas([_place()])
        assert isinstance(result, CanvasResult)

    def test_canvas_is_ndarray(self):
        result = build_canvas([_place()])
        assert isinstance(result.canvas, np.ndarray)

    def test_n_placed_matches(self):
        p1 = _place(fid=0, img=_img(h=5, w=5), x=0, y=0)
        p2 = _place(fid=1, img=_img(h=5, w=5), x=10, y=0)
        result = build_canvas([p1, p2])
        assert result.n_placed == 2

    def test_coverage_in_range(self):
        result = build_canvas([_place()])
        assert 0.0 <= result.coverage <= 1.0

    def test_coverage_full(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=img, x=0, y=0)
        result = build_canvas([p], canvas_w=10, canvas_h=10)
        assert result.coverage == pytest.approx(1.0)

    def test_explicit_canvas_size(self):
        p = _place(img=_img(h=5, w=5))
        result = build_canvas([p], canvas_w=20, canvas_h=20)
        assert result.canvas_w == 20
        assert result.canvas_h == 20


# ─── TestCropToContent ────────────────────────────────────────────────────────

class TestCropToContent:
    def test_returns_ndarray(self):
        p = FragmentPlacement(fragment_id=0, image=np.zeros((5, 5, 3), dtype=np.uint8), x=5, y=5)
        result = build_canvas([p], canvas_w=20, canvas_h=20)
        cropped = crop_to_content(result)
        assert isinstance(cropped, np.ndarray)

    def test_all_background_returns_original(self):
        result = build_canvas([_place(img=np.full((5, 5, 3), 255, dtype=np.uint8), x=0, y=0)])
        cropped = crop_to_content(result, bg_color=(255, 255, 255))
        assert cropped.shape == result.canvas.shape

    def test_cropped_smaller_than_canvas(self):
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=img, x=5, y=5)
        result = build_canvas([p], canvas_w=20, canvas_h=20)
        cropped = crop_to_content(result)
        assert cropped.shape[0] <= 20
        assert cropped.shape[1] <= 20


# ─── TestBatchBuildCanvases ───────────────────────────────────────────────────

class TestBatchBuildCanvases:
    def test_returns_list(self):
        groups = [[_place()], [_place(fid=1)]]
        results = batch_build_canvases(groups)
        assert isinstance(results, list)

    def test_length_matches(self):
        groups = [[_place()], [_place(fid=1)], [_place(fid=2)]]
        results = batch_build_canvases(groups)
        assert len(results) == 3

    def test_all_canvas_results(self):
        groups = [[_place()], [_place(fid=1)]]
        for r in batch_build_canvases(groups):
            assert isinstance(r, CanvasResult)
