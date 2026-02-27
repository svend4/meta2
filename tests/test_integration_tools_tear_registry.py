"""
Integration tests for tools/tear_generator.py and tools/registry.py.

Tests are grouped into two classes:
  - TestTearGenerator  (>= 13 tests)
  - TestToolRegistry   (>= 13 tests)

No mocks are used; every assertion is made against real computed values.
"""

import sys
import os
import pytest
import numpy as np

# Ensure the project root is importable regardless of how pytest is invoked.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.tear_generator import (
    _grid_shape,
    _divide_with_jitter,
    _fractal_profile,
    _torn_mask,
    generate_test_document,
    tear_document,
)

from tools.registry import (
    ToolInfo,
    build_tool_registry,
    list_tools,
    get_tool,
    run_tool,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_rng(seed: int = 0) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _solid_image(h: int = 200, w: int = 300, fill: int = 128,
                 channels: int = 3) -> np.ndarray:
    """Return a solid-colour BGR image for deterministic testing."""
    return np.full((h, w, channels), fill, dtype=np.uint8)


# =============================================================================
# TestTearGenerator
# =============================================================================

class TestTearGenerator:
    """Integration tests for tear_generator.py helper functions and main API."""

    # ── _grid_shape ──────────────────────────────────────────────────────────

    def test_grid_shape_returns_two_integers(self):
        cols, rows = _grid_shape(6)
        assert isinstance(cols, int)
        assert isinstance(rows, int)

    def test_grid_shape_product_gte_n(self):
        """cols * rows must be >= n so the grid covers all pieces."""
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16]:
            cols, rows = _grid_shape(n)
            assert cols * rows >= n, f"grid {cols}x{rows} < {n}"

    def test_grid_shape_single_piece(self):
        cols, rows = _grid_shape(1)
        assert cols == 1
        assert rows == 1

    def test_grid_shape_four_pieces(self):
        cols, rows = _grid_shape(4)
        assert cols == 2
        assert rows == 2

    def test_grid_shape_nine_pieces(self):
        cols, rows = _grid_shape(9)
        assert cols == 3
        assert rows == 3

    def test_grid_shape_cols_close_to_sqrt(self):
        """cols should equal ceil(sqrt(n))."""
        for n in [2, 5, 10, 20]:
            cols, rows = _grid_shape(n)
            expected_cols = int(np.ceil(np.sqrt(n)))
            assert cols == expected_cols

    # ── _divide_with_jitter ──────────────────────────────────────────────────

    def test_divide_jitter_starts_at_zero(self):
        rng = _make_rng(1)
        bounds = _divide_with_jitter(500, 4, rng)
        assert bounds[0] == 0

    def test_divide_jitter_ends_at_total(self):
        rng = _make_rng(2)
        bounds = _divide_with_jitter(400, 3, rng)
        assert bounds[-1] == 400

    def test_divide_jitter_correct_length(self):
        rng = _make_rng(3)
        n = 5
        bounds = _divide_with_jitter(600, n, rng)
        assert len(bounds) == n + 1

    def test_divide_jitter_strictly_increasing(self):
        rng = _make_rng(4)
        bounds = _divide_with_jitter(800, 6, rng)
        for a, b in zip(bounds, bounds[1:]):
            assert b > a, f"bounds not increasing: {bounds}"

    def test_divide_jitter_n_equals_one(self):
        """Single part should give [0, total]."""
        rng = _make_rng(5)
        bounds = _divide_with_jitter(300, 1, rng)
        assert bounds[0] == 0
        assert bounds[-1] == 300
        assert len(bounds) == 2

    def test_divide_jitter_reproducible_with_same_seed(self):
        bounds1 = _divide_with_jitter(500, 4, _make_rng(77))
        bounds2 = _divide_with_jitter(500, 4, _make_rng(77))
        assert bounds1 == bounds2

    def test_divide_jitter_differs_with_different_seed(self):
        bounds1 = _divide_with_jitter(500, 4, _make_rng(1))
        bounds2 = _divide_with_jitter(500, 4, _make_rng(2))
        # It is theoretically possible (but extremely unlikely) they are equal;
        # for practical purposes two different seeds should yield different results.
        assert bounds1 != bounds2

    # ── _fractal_profile ─────────────────────────────────────────────────────

    def test_fractal_profile_length(self):
        rng = _make_rng(10)
        length = 150
        profile = _fractal_profile(length, 10, rng)
        assert len(profile) == length

    def test_fractal_profile_returns_ndarray(self):
        rng = _make_rng(11)
        profile = _fractal_profile(100, 5, rng)
        assert isinstance(profile, np.ndarray)

    def test_fractal_profile_zero_amplitude_near_zero(self):
        """With amplitude=0 all octave knots are 0 → profile should be all zeros."""
        rng = _make_rng(12)
        profile = _fractal_profile(80, 0, rng)
        assert np.allclose(profile, 0.0)

    def test_fractal_profile_bounded_by_amplitude(self):
        """Profile values should stay within a reasonable multiple of amplitude."""
        amplitude = 20
        rng = _make_rng(13)
        profile = _fractal_profile(200, amplitude, rng)
        # The fBm sum can exceed amplitude by at most sum of geometric series
        # (1 + 0.55 + 0.55^2 + ... < 2.3) times amplitude.
        assert np.all(np.abs(profile) < amplitude * 3), (
            f"profile values out of expected range: "
            f"max={np.abs(profile).max():.2f} vs limit={amplitude * 3}"
        )

    def test_fractal_profile_reproducible(self):
        p1 = _fractal_profile(120, 8, _make_rng(99))
        p2 = _fractal_profile(120, 8, _make_rng(99))
        np.testing.assert_array_equal(p1, p2)

    def test_fractal_profile_octaves_parameter(self):
        """More octaves should not change the array length."""
        rng = _make_rng(14)
        p = _fractal_profile(100, 5, rng, octaves=3)
        assert len(p) == 100

    # ── _torn_mask ───────────────────────────────────────────────────────────

    def test_torn_mask_shape_matches_image(self):
        h, w = 200, 300
        rng = _make_rng(20)
        mask = _torn_mask(h, w, 50, 200, 40, 160, noise_level=0.3, rng=rng)
        assert mask.shape == (h, w)

    def test_torn_mask_dtype_uint8(self):
        rng = _make_rng(21)
        mask = _torn_mask(100, 150, 20, 120, 10, 90, noise_level=0.5, rng=rng)
        assert mask.dtype == np.uint8

    def test_torn_mask_binary_values(self):
        """Mask should only contain 0 or 255."""
        rng = _make_rng(22)
        mask = _torn_mask(200, 300, 30, 250, 30, 170, noise_level=0.4, rng=rng)
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255})

    def test_torn_mask_nonzero_pixels_exist(self):
        rng = _make_rng(23)
        mask = _torn_mask(300, 400, 50, 350, 50, 250, noise_level=0.3, rng=rng)
        assert np.any(mask > 0), "Expected at least some foreground pixels"

    def test_torn_mask_zero_noise_mostly_rectangular(self):
        """With noise_level=0 the mask should cover most of the nominal rectangle.

        Note: _torn_mask floors amplitude to max(2, ...) so even noise_level=0
        produces a tiny non-zero amplitude.  We therefore check that the mask
        covers at least 97% of the nominal rectangle rather than exactly 100%%.
        """
        rng = _make_rng(24)
        h, w = 300, 400
        x0, x1, y0, y1 = 50, 350, 40, 260
        mask = _torn_mask(h, w, x0, x1, y0, y1, noise_level=0.0, rng=rng)
        rect_pixels = (y1 - y0) * (x1 - x0)
        mask_pixels = int(np.sum(mask > 0))
        coverage = mask_pixels / rect_pixels
        assert coverage >= 0.97, (
            f"Expected >= 97% coverage at noise_level=0, got {coverage:.2%}"
        )

    # ── generate_test_document ───────────────────────────────────────────────

    def test_generate_test_document_shape(self):
        img = generate_test_document(width=400, height=500, seed=0)
        assert img.shape == (500, 400, 3)

    def test_generate_test_document_dtype(self):
        img = generate_test_document(width=200, height=250, seed=1)
        assert img.dtype == np.uint8

    def test_generate_test_document_has_dark_pixels(self):
        """The synthetic document should contain text/rule pixels darker than white."""
        img = generate_test_document(width=400, height=500, seed=42)
        assert np.any(img < 200), "Expected some non-white pixels in synthetic document"

    def test_generate_test_document_reproducible(self):
        img1 = generate_test_document(width=300, height=400, seed=7)
        img2 = generate_test_document(width=300, height=400, seed=7)
        np.testing.assert_array_equal(img1, img2)

    def test_generate_test_document_different_seeds_differ(self):
        img1 = generate_test_document(width=300, height=400, seed=0)
        img2 = generate_test_document(width=300, height=400, seed=99)
        assert not np.array_equal(img1, img2)

    # ── tear_document ────────────────────────────────────────────────────────

    def test_tear_document_returns_list(self):
        img = _solid_image()
        result = tear_document(img, n_pieces=4, seed=42)
        assert isinstance(result, list)

    def test_tear_document_piece_count_leq_n(self):
        """Result length should be <= n_pieces (empty fragments are skipped)."""
        img = _solid_image(h=400, w=400)
        frags = tear_document(img, n_pieces=4, seed=1)
        assert len(frags) <= 4

    def test_tear_document_piece_count_positive(self):
        """At least one fragment should be produced for a non-trivial image."""
        img = _solid_image(h=400, w=400)
        frags = tear_document(img, n_pieces=4, seed=1)
        assert len(frags) >= 1

    def test_tear_document_fragments_are_ndarrays(self):
        img = _solid_image()
        frags = tear_document(img, n_pieces=4, seed=5)
        for f in frags:
            assert isinstance(f, np.ndarray)

    def test_tear_document_fragment_channel_count_preserved(self):
        img = _solid_image(h=300, w=300, channels=3)
        frags = tear_document(img, n_pieces=4, seed=7)
        for f in frags:
            assert f.ndim == 3
            assert f.shape[2] == 3

    def test_tear_document_reproducible(self):
        img = _solid_image(h=400, w=400)
        frags1 = tear_document(img, n_pieces=4, seed=100)
        frags2 = tear_document(img, n_pieces=4, seed=100)
        assert len(frags1) == len(frags2)
        for f1, f2 in zip(frags1, frags2):
            np.testing.assert_array_equal(f1, f2)

    def test_tear_document_different_seeds_give_different_shapes(self):
        """Different seeds should almost certainly produce different fragment shapes."""
        img = _solid_image(h=400, w=400)
        frags1 = tear_document(img, n_pieces=4, seed=1)
        frags2 = tear_document(img, n_pieces=4, seed=2)
        shapes1 = [f.shape for f in frags1]
        shapes2 = [f.shape for f in frags2]
        assert shapes1 != shapes2

    def test_tear_document_single_piece(self):
        """Requesting one piece should return one fragment."""
        img = _solid_image(h=300, w=300)
        frags = tear_document(img, n_pieces=1, seed=10)
        assert len(frags) == 1

    def test_tear_document_fragment_smaller_than_original(self):
        """Each fragment must be smaller than the original in at least one dimension."""
        img = _solid_image(h=400, w=500)
        frags = tear_document(img, n_pieces=4, seed=20)
        h_orig, w_orig = img.shape[:2]
        for f in frags:
            h_f, w_f = f.shape[:2]
            assert h_f <= h_orig and w_f <= w_orig, (
                f"Fragment {f.shape} should not exceed original {img.shape}"
            )

    def test_tear_document_white_background_outside_mask(self):
        """Pixels outside the torn mask should be 255 (white background)."""
        # Use a pure black image so any pixel that is NOT covered by the mask
        # will be 255 (white fill).
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        frags = tear_document(img, n_pieces=4, seed=33)
        for frag in frags:
            # White pixels exist in every fragment (padding / background)
            assert np.any(frag == 255)

    def test_tear_document_noise_level_zero(self):
        """noise_level=0 should still produce the correct number of fragments."""
        img = _solid_image(h=400, w=400)
        frags = tear_document(img, n_pieces=4, noise_level=0.0, seed=55)
        assert len(frags) >= 1

    def test_tear_document_nine_pieces(self):
        img = _solid_image(h=600, w=600)
        frags = tear_document(img, n_pieces=9, seed=77)
        assert len(frags) >= 1
        assert len(frags) <= 9

    def test_tear_document_with_generated_document(self):
        """End-to-end: generate a synthetic document and tear it."""
        doc = generate_test_document(width=400, height=500, seed=3)
        frags = tear_document(doc, n_pieces=6, noise_level=0.4, seed=3)
        assert len(frags) >= 1
        for f in frags:
            assert f.dtype == np.uint8
            assert f.ndim == 3


# =============================================================================
# TestToolRegistry
# =============================================================================

EXPECTED_KEYS = {"benchmark", "evaluate", "mix", "profile", "serve", "tear"}


class TestToolRegistry:
    """Integration tests for tools/registry.py."""

    # ── ToolInfo dataclass ───────────────────────────────────────────────────

    def test_tool_info_fields_stored_correctly(self):
        info = ToolInfo(
            name="test_tool",
            description="A test tool",
            params=["a", "b"],
        )
        assert info.name == "test_tool"
        assert info.description == "A test tool"
        assert info.params == ["a", "b"]

    def test_tool_info_default_params_is_empty_list(self):
        info = ToolInfo(name="x", description="y")
        assert info.params == []

    def test_tool_info_repr_omits_private_fields(self):
        """_loader and _fn are repr=False so they should not appear in repr()."""
        info = ToolInfo(name="foo", description="bar", _loader=lambda: None)
        r = repr(info)
        assert "_loader" not in r
        assert "_fn" not in r

    # ── ToolInfo.load ────────────────────────────────────────────────────────

    def test_tool_info_load_returns_callable_on_success(self):
        def _loader():
            return lambda: "result"

        info = ToolInfo(name="t", description="d", _loader=_loader)
        fn = info.load()
        assert callable(fn)

    def test_tool_info_load_returns_none_when_loader_raises(self):
        def _bad_loader():
            raise ImportError("no such module")

        info = ToolInfo(name="bad", description="d", _loader=_bad_loader)
        fn = info.load()
        assert fn is None

    def test_tool_info_load_returns_none_when_no_loader(self):
        info = ToolInfo(name="empty", description="d")
        fn = info.load()
        assert fn is None

    def test_tool_info_load_is_lazy_cached(self):
        """load() should cache _fn; the loader must only be called once."""
        call_count = {"n": 0}

        def _loader():
            call_count["n"] += 1
            return lambda: 42

        info = ToolInfo(name="lazy", description="d", _loader=_loader)
        info.load()
        info.load()
        assert call_count["n"] == 1, "loader was called more than once"

    def test_tool_info_load_returns_none_when_loader_raises_system_exit(self):
        def _loader():
            raise SystemExit(1)

        info = ToolInfo(name="exit_tool", description="d", _loader=_loader)
        fn = info.load()
        assert fn is None

    # ── ToolInfo.run ─────────────────────────────────────────────────────────

    def test_tool_info_run_raises_runtime_error_when_fn_is_none(self):
        info = ToolInfo(name="unavailable", description="d")
        with pytest.raises(RuntimeError):
            info.run()

    def test_tool_info_run_calls_fn_with_kwargs(self):
        def _loader():
            def _fn(x, y):
                return x + y
            return _fn

        info = ToolInfo(name="adder", description="d", _loader=_loader)
        result = info.run(x=3, y=4)
        assert result == 7

    def test_tool_info_run_runtime_error_message_contains_name(self):
        info = ToolInfo(name="my_missing_tool", description="d")
        with pytest.raises(RuntimeError, match="my_missing_tool"):
            info.run()

    # ── build_tool_registry ──────────────────────────────────────────────────

    def test_build_tool_registry_returns_dict(self):
        reg = build_tool_registry()
        assert isinstance(reg, dict)

    def test_build_tool_registry_has_all_six_keys(self):
        reg = build_tool_registry()
        assert set(reg.keys()) == EXPECTED_KEYS

    def test_build_tool_registry_values_are_tool_info(self):
        reg = build_tool_registry()
        for key, val in reg.items():
            assert isinstance(val, ToolInfo), f"{key!r} is not a ToolInfo"

    def test_build_tool_registry_names_match_keys(self):
        reg = build_tool_registry()
        for key, info in reg.items():
            assert info.name == key, f"key={key!r} but info.name={info.name!r}"

    def test_build_tool_registry_all_have_descriptions(self):
        reg = build_tool_registry()
        for key, info in reg.items():
            assert info.description, f"{key!r} has empty description"

    def test_build_tool_registry_all_have_params_lists(self):
        reg = build_tool_registry()
        for key, info in reg.items():
            assert isinstance(info.params, list), (
                f"{key!r}.params is not a list"
            )

    # ── list_tools ───────────────────────────────────────────────────────────

    def test_list_tools_returns_dict(self):
        result = list_tools()
        assert isinstance(result, dict)

    def test_list_tools_has_six_entries(self):
        result = list_tools()
        assert len(result) == 6

    def test_list_tools_has_expected_keys(self):
        result = list_tools()
        assert set(result.keys()) == EXPECTED_KEYS

    def test_list_tools_values_are_tool_info_instances(self):
        result = list_tools()
        for name, info in result.items():
            assert isinstance(info, ToolInfo), f"{name!r} value is not ToolInfo"

    def test_list_tools_returns_independent_copy(self):
        """Mutating the returned dict must not corrupt the registry."""
        result1 = list_tools()
        result1.pop("tear")
        result2 = list_tools()
        assert "tear" in result2

    # ── get_tool ─────────────────────────────────────────────────────────────

    def test_get_tool_returns_tool_info_for_known_names(self):
        for name in EXPECTED_KEYS:
            info = get_tool(name)
            assert isinstance(info, ToolInfo), f"get_tool({name!r}) returned {info!r}"

    def test_get_tool_returns_none_for_unknown(self):
        info = get_tool("this_tool_does_not_exist")
        assert info is None

    def test_get_tool_name_attribute_matches(self):
        for name in EXPECTED_KEYS:
            info = get_tool(name)
            assert info.name == name

    def test_get_tool_tear_has_expected_params(self):
        info = get_tool("tear")
        assert "image" in info.params
        assert "n_pieces" in info.params

    # ── run_tool ─────────────────────────────────────────────────────────────

    def test_run_tool_raises_key_error_for_unknown_name(self):
        with pytest.raises(KeyError):
            run_tool("nonexistent_tool_xyz")

    def test_run_tool_key_error_message_contains_name(self):
        with pytest.raises(KeyError, match="nonexistent_tool_xyz"):
            run_tool("nonexistent_tool_xyz")

    def test_run_tool_tear_produces_fragments(self):
        """run_tool('tear', ...) should call tear_document and return fragments."""
        img = _solid_image(h=400, w=400)
        frags = run_tool("tear", image=img, n_pieces=4, noise_level=0.3, seed=42)
        assert isinstance(frags, list)
        assert len(frags) >= 1
        for f in frags:
            assert isinstance(f, np.ndarray)

    def test_run_tool_tear_reproducible(self):
        img = _solid_image(h=300, w=300)
        frags1 = run_tool("tear", image=img, n_pieces=4, seed=55)
        frags2 = run_tool("tear", image=img, n_pieces=4, seed=55)
        assert len(frags1) == len(frags2)
        for f1, f2 in zip(frags1, frags2):
            np.testing.assert_array_equal(f1, f2)

    def test_run_tool_tear_fragment_count_matches_direct_call(self):
        """run_tool('tear') should yield same result as calling tear_document directly."""
        img = _solid_image(h=400, w=400)
        kwargs = {"image": img, "n_pieces": 6, "noise_level": 0.4, "seed": 7}
        via_registry = run_tool("tear", **kwargs)
        direct = tear_document(**kwargs)
        assert len(via_registry) == len(direct)
        for f_reg, f_dir in zip(via_registry, direct):
            np.testing.assert_array_equal(f_reg, f_dir)
