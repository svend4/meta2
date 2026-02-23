"""Extra tests for puzzle_reconstruction.assembly.canvas_builder."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _img(h=10, w=10, color=128, channels=3):
    if channels == 1:
        return np.full((h, w), color, dtype=np.uint8)
    return np.full((h, w, channels), color, dtype=np.uint8)


def _place(fid=0, img=None, x=0, y=0):
    if img is None:
        img = _img()
    return FragmentPlacement(fragment_id=fid, image=img, x=x, y=y)


# ─── TestCanvasConfigExtra ───────────────────────────────────────────────────

class TestCanvasConfigExtra:
    def test_default_blend_mode(self):
        cfg = CanvasConfig()
        assert cfg.blend_mode == "overwrite"

    def test_default_padding_zero(self):
        cfg = CanvasConfig()
        assert cfg.padding == 0

    def test_black_bg_ok(self):
        cfg = CanvasConfig(bg_color=(0, 0, 0))
        assert cfg.bg_color == (0, 0, 0)

    def test_gray_bg_ok(self):
        cfg = CanvasConfig(bg_color=(128, 128, 128))
        assert cfg.bg_color == (128, 128, 128)

    def test_padding_10_ok(self):
        cfg = CanvasConfig(padding=10)
        assert cfg.padding == 10

    def test_uint8_dtype_ok(self):
        cfg = CanvasConfig(dtype="uint8")
        assert cfg.dtype == "uint8"

    def test_float32_dtype_ok(self):
        cfg = CanvasConfig(dtype="float32")
        assert cfg.dtype == "float32"

    def test_overwrite_blend_ok(self):
        cfg = CanvasConfig(blend_mode="overwrite")
        assert cfg.blend_mode == "overwrite"

    def test_average_blend_ok(self):
        cfg = CanvasConfig(blend_mode="average")
        assert cfg.blend_mode == "average"

    def test_full_255_bg_ok(self):
        cfg = CanvasConfig(bg_color=(255, 255, 255))
        assert cfg.bg_color == (255, 255, 255)


# ─── TestFragmentPlacementExtra ─────────────────────────────────────────────

class TestFragmentPlacementExtra:
    def test_large_coords_ok(self):
        p = _place(img=_img(h=5, w=5), x=1000, y=2000)
        assert p.x == 1000
        assert p.y == 2000

    def test_zero_coords_ok(self):
        p = _place(x=0, y=0)
        assert p.x == 0
        assert p.y == 0

    def test_h_w_props(self):
        p = _place(img=_img(h=7, w=13))
        assert p.h == 7
        assert p.w == 13

    def test_x2_y2_at_origin(self):
        p = _place(img=_img(h=3, w=4), x=0, y=0)
        assert p.x2 == 4
        assert p.y2 == 3

    def test_x2_y2_with_offset(self):
        p = _place(img=_img(h=6, w=9), x=10, y=5)
        assert p.x2 == 19   # 10 + 9
        assert p.y2 == 11   # 5 + 6

    def test_gray_image_ndim(self):
        p = _place(img=_img(h=4, w=4, channels=1))
        assert p.image.ndim == 2

    def test_color_image_ndim(self):
        p = _place(img=_img(h=4, w=4, channels=3))
        assert p.image.ndim == 3


# ─── TestCanvasResultExtra ──────────────────────────────────────────────────

class TestCanvasResultExtra:
    def _result(self, **kwargs):
        defaults = dict(
            canvas=np.zeros((10, 10, 3), dtype=np.uint8),
            coverage=0.5, n_placed=1,
            canvas_w=10, canvas_h=10,
        )
        defaults.update(kwargs)
        return CanvasResult(**defaults)

    def test_shape_property(self):
        r = self._result()
        assert r.shape == (10, 10, 3)

    def test_coverage_zero_ok(self):
        r = self._result(coverage=0.0, n_placed=0)
        assert r.coverage == pytest.approx(0.0)

    def test_coverage_one_ok(self):
        r = self._result(coverage=1.0)
        assert r.coverage == pytest.approx(1.0)

    def test_canvas_w_h_stored(self):
        r = self._result(canvas_w=30, canvas_h=20)
        assert r.canvas_w == 30
        assert r.canvas_h == 20

    def test_n_placed_zero_ok(self):
        r = self._result(n_placed=0, coverage=0.0)
        assert r.n_placed == 0

    def test_large_n_placed_ok(self):
        r = self._result(n_placed=1000)
        assert r.n_placed == 1000

    def test_canvas_stored(self):
        canvas = np.zeros((5, 5, 3), dtype=np.uint8)
        r = self._result(canvas=canvas, canvas_w=5, canvas_h=5)
        assert r.canvas.shape == (5, 5, 3)


# ─── TestComputeCanvasSizeExtra ─────────────────────────────────────────────

class TestComputeCanvasSizeExtra:
    def test_two_adjacent(self):
        p1 = _place(img=_img(h=10, w=10), x=0, y=0)
        p2 = _place(img=_img(h=10, w=10), x=10, y=0)
        w, h = compute_canvas_size([p1, p2])
        assert w == 20
        assert h == 10

    def test_padding_doubled_ok(self):
        p = _place(img=_img(h=8, w=8), x=0, y=0)
        w1, h1 = compute_canvas_size([p], padding=0)
        w2, h2 = compute_canvas_size([p], padding=4)
        assert w2 == w1 + 4
        assert h2 == h1 + 4

    def test_returns_tuple(self):
        result = compute_canvas_size([_place()])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_large_offset(self):
        p = _place(img=_img(h=5, w=5), x=100, y=200)
        w, h = compute_canvas_size([p])
        assert w == 105
        assert h == 205

    def test_three_placements(self):
        ps = [_place(img=_img(h=4, w=4), x=i * 5, y=0) for i in range(3)]
        w, h = compute_canvas_size(ps)
        assert w >= 14
        assert h >= 4


# ─── TestMakeEmptyCanvasExtra ───────────────────────────────────────────────

class TestMakeEmptyCanvasExtra:
    def test_gray_canvas_shape(self):
        # 3-channel by default
        c = make_empty_canvas(10, 10)
        assert c.shape == (10, 10, 3)

    def test_large_canvas(self):
        c = make_empty_canvas(1000, 500)
        assert c.shape == (500, 1000, 3)

    def test_rectangular_canvas(self):
        c = make_empty_canvas(60, 20)
        assert c.shape == (20, 60, 3)

    def test_bg_color_custom(self):
        cfg = CanvasConfig(bg_color=(10, 20, 30))
        c = make_empty_canvas(5, 5, cfg)
        assert c[0, 0, 0] == 10
        assert c[0, 0, 1] == 20
        assert c[0, 0, 2] == 30

    def test_dtype_uint8_default(self):
        c = make_empty_canvas(5, 5)
        assert c.dtype == np.uint8

    def test_dtype_float32(self):
        cfg = CanvasConfig(dtype="float32")
        c = make_empty_canvas(5, 5, cfg)
        assert c.dtype == np.float32


# ─── TestPlaceFragmentExtra ─────────────────────────────────────────────────

class TestPlaceFragmentExtra:
    def test_overwrite_at_offset(self):
        canvas = make_empty_canvas(20, 20)  # white
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=img, x=5, y=5)
        place_fragment(canvas, p)
        assert (canvas[5:10, 5:10] == 0).all()
        assert (canvas[0:5, 0:5] == 255).all()

    def test_average_blend_center(self):
        canvas = np.full((10, 10, 3), 100, dtype=np.uint8)
        img = np.full((4, 4, 3), 200, dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=img, x=3, y=3)
        place_fragment(canvas, p, blend_mode="average")
        assert canvas[3, 3, 0] == 150

    def test_returns_same_object(self):
        canvas = make_empty_canvas(10, 10)
        p = _place()
        result = place_fragment(canvas, p)
        assert result is canvas

    def test_grayscale_placed_full_region(self):
        canvas = make_empty_canvas(20, 20)
        gray = np.zeros((4, 4), dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=gray, x=0, y=0)
        place_fragment(canvas, p)
        assert (canvas[0:4, 0:4] == 0).all()

    def test_partial_out_of_bounds_no_error(self):
        canvas = make_empty_canvas(8, 8)
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=img, x=6, y=6)
        result = place_fragment(canvas, p)
        assert result.shape == (8, 8, 3)


# ─── TestBuildCanvasExtra ───────────────────────────────────────────────────

class TestBuildCanvasExtra:
    def test_single_placement(self):
        result = build_canvas([_place(img=_img(h=5, w=5, color=0))])
        assert isinstance(result, CanvasResult)

    def test_n_placed_two(self):
        p1 = _place(fid=0, img=_img(h=5, w=5), x=0, y=0)
        p2 = _place(fid=1, img=_img(h=5, w=5), x=10, y=10)
        result = build_canvas([p1, p2])
        assert result.n_placed == 2

    def test_explicit_large_canvas(self):
        p = _place(img=_img(h=5, w=5), x=0, y=0)
        result = build_canvas([p], canvas_w=100, canvas_h=100)
        assert result.canvas_w == 100
        assert result.canvas_h == 100

    def test_coverage_white_fragment_on_white_bg(self):
        img = np.full((5, 5, 3), 255, dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=img, x=0, y=0)
        result = build_canvas([p], canvas_w=10, canvas_h=10)
        # white on white — coverage may be 0
        assert 0.0 <= result.coverage <= 1.0

    def test_coverage_full_black_on_white(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=img, x=0, y=0)
        result = build_canvas([p], canvas_w=10, canvas_h=10)
        assert result.coverage == pytest.approx(1.0)

    def test_canvas_dtype_uint8(self):
        result = build_canvas([_place()])
        assert result.canvas.dtype == np.uint8


# ─── TestCropToContentExtra ─────────────────────────────────────────────────

class TestCropToContentExtra:
    def test_black_fragment_cropped(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=img, x=8, y=8)
        result = build_canvas([p], canvas_w=20, canvas_h=20)
        cropped = crop_to_content(result)
        assert cropped.shape[0] <= 20
        assert cropped.shape[1] <= 20

    def test_full_canvas_unchanged(self):
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=img, x=0, y=0)
        result = build_canvas([p], canvas_w=20, canvas_h=20)
        cropped = crop_to_content(result)
        assert cropped.shape == (20, 20, 3)

    def test_returns_ndarray(self):
        p = FragmentPlacement(fragment_id=0,
                              image=np.zeros((5, 5, 3), dtype=np.uint8),
                              x=3, y=3)
        result = build_canvas([p], canvas_w=15, canvas_h=15)
        cropped = crop_to_content(result)
        assert isinstance(cropped, np.ndarray)


# ─── TestBatchBuildCanvasesExtra ────────────────────────────────────────────

class TestBatchBuildCanvasesExtra:
    def test_single_group(self):
        results = batch_build_canvases([[_place()]])
        assert len(results) == 1

    def test_four_groups(self):
        groups = [[_place(fid=i)] for i in range(4)]
        results = batch_build_canvases(groups)
        assert len(results) == 4

    def test_all_canvas_results(self):
        groups = [[_place(fid=i)] for i in range(3)]
        for r in batch_build_canvases(groups):
            assert isinstance(r, CanvasResult)

    def test_each_n_placed_one(self):
        groups = [[_place(fid=i)] for i in range(3)]
        for r in batch_build_canvases(groups):
            assert r.n_placed == 1

    def test_two_placements_per_group(self):
        g = [_place(fid=0, img=_img(h=5, w=5), x=0, y=0),
             _place(fid=1, img=_img(h=5, w=5), x=10, y=0)]
        results = batch_build_canvases([g, g])
        for r in results:
            assert r.n_placed == 2
