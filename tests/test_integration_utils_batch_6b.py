"""Integration tests for utils batch 6b:
threshold_utils, tile_utils, topology_utils,
tracker_utils, voting_utils, window_utils.
"""
import numpy as np
import pytest

from puzzle_reconstruction.utils.threshold_utils import (
    apply_threshold, binarize, adaptive_threshold, soft_threshold,
    threshold_matrix, hysteresis_threshold, otsu_threshold,
    count_above, fraction_above, batch_threshold,
)
from puzzle_reconstruction.utils.tile_utils import (
    TileConfig, tile_image, reassemble_tiles, tile_overlap_ratio,
    filter_tiles_by_content, compute_tile_grid, batch_tile_images,
)
from puzzle_reconstruction.utils.topology_utils import (
    compute_euler_number, count_holes, compute_solidity, compute_extent,
    compute_convexity, compute_compactness, is_simply_connected,
    shape_complexity, batch_topology,
)
from puzzle_reconstruction.utils.tracker_utils import (
    TrackerConfig, create_iter_tracker, record_step, get_values, get_steps,
    get_best_record, get_worst_record, compute_delta, is_improving,
    find_plateau_start, smooth_values, tracker_stats, compare_trackers,
)
from puzzle_reconstruction.utils.voting_utils import (
    VoteConfig, cast_pair_votes, aggregate_pair_votes, cast_position_votes,
    majority_vote, weighted_vote, rank_fusion, batch_vote,
)
from puzzle_reconstruction.utils.window_utils import (
    WindowConfig, apply_window_function, rolling_mean, rolling_std,
    rolling_max, rolling_min, compute_overlap, split_into_windows,
    merge_windows, batch_rolling,
)

rng = np.random.default_rng(42)


# ─── threshold_utils ──────────────────────────────────────────────────────────

class TestThresholdUtils:
    def test_apply_threshold_basic_and_invert(self):
        arr = np.array([0.1, 0.5, 0.9])
        assert list(apply_threshold(arr, 0.4)) == [False, True, True]
        assert list(apply_threshold(arr, 0.6, invert=True)) == [True, True, False]

    def test_apply_threshold_empty_raises(self):
        with pytest.raises(ValueError):
            apply_threshold(np.array([]), 0.5)

    def test_binarize_values_and_invert(self):
        arr = np.array([0.2, 0.8, 0.5])
        result = binarize(arr, 0.5)
        assert result.dtype == np.float64
        assert list(result) == [0.0, 1.0, 1.0]
        assert list(binarize(np.array([0.2, 0.8]), 0.5, invert=True)) == [1.0, 0.0]

    def test_adaptive_threshold_flat(self):
        assert all(adaptive_threshold(np.ones(10) * 0.5, window=3))

    def test_adaptive_threshold_non_1d_raises(self):
        with pytest.raises(ValueError):
            adaptive_threshold(np.ones((3, 3)), window=3)

    def test_soft_threshold_shrinkage(self):
        result = soft_threshold(np.array([-2.0, -0.5, 0.0, 0.5, 2.0]), 1.0)
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.0, 0.0, 1.0])

    def test_threshold_matrix_fill(self):
        m = np.array([[0.1, 0.8], [0.3, 0.6]])
        r = threshold_matrix(m, 0.5, fill=-1.0)
        assert r[0, 0] == -1.0 and r[0, 1] == 0.8

    def test_hysteresis_threshold_propagation(self):
        result = hysteresis_threshold(np.array([0.9, 0.4, 0.4, 0.1]), low=0.3, high=0.8)
        assert result[0] and result[1] and not result[3]

    def test_otsu_threshold_bimodal(self):
        low = rng.uniform(0.0, 0.3, 50)
        high = rng.uniform(0.7, 1.0, 50)
        t = otsu_threshold(np.concatenate([low, high]))
        assert float(low.max()) < t < float(high.min()) + 0.1

    def test_count_above_and_fraction(self):
        arr = np.array([0.1, 0.5, 0.9, 0.7])
        assert count_above(arr, 0.5) == 3
        assert fraction_above(arr, 0.6) == pytest.approx(0.5)

    def test_batch_threshold(self):
        masks = batch_threshold([np.array([0.1, 0.9]), np.array([0.4, 0.6])], 0.5)
        assert len(masks) == 2 and list(masks[0]) == [False, True]


# ─── tile_utils ───────────────────────────────────────────────────────────────

class TestTileUtils:
    def _img(self, h=64, w=64):
        return rng.integers(0, 256, (h, w), dtype=np.uint8)

    def test_tile_image_no_overlap(self):
        tiles = tile_image(self._img(64, 64), TileConfig(tile_h=32, tile_w=32))
        assert len(tiles) == 4
        assert all(t.data.shape == (32, 32) for t in tiles)

    def test_tile_image_padding(self):
        tiles = tile_image(self._img(70, 70), TileConfig(tile_h=64, tile_w=64, pad_value=128))
        assert len(tiles) == 4 and all(t.data.shape == (64, 64) for t in tiles)

    def test_tile_image_color(self):
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        assert all(t.data.shape == (32, 32, 3) for t in tile_image(img, TileConfig(32, 32)))

    def test_reassemble_tiles_roundtrip(self):
        img = self._img()
        cfg = TileConfig(tile_h=32, tile_w=32)
        np.testing.assert_array_equal(reassemble_tiles(tile_image(img, cfg), img.shape[:2]), img)

    def test_compute_tile_grid_count(self):
        assert len(compute_tile_grid(32, 32, TileConfig(tile_h=16, tile_w=16))) == 4

    def test_tile_overlap_ratio(self):
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(self._img(), cfg)
        assert tile_overlap_ratio(tiles[0], tiles[0]) == pytest.approx(1.0)
        assert tile_overlap_ratio(tiles[0], tiles[3]) == pytest.approx(0.0)

    def test_filter_tiles_by_content(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[32:, 32:] = 200
        filtered = filter_tiles_by_content(tile_image(img, TileConfig(32, 32)), min_foreground=0.3)
        assert len(filtered) == 1

    def test_tile_config_validation(self):
        with pytest.raises(ValueError):
            TileConfig(tile_h=-1)
        with pytest.raises(ValueError):
            TileConfig(pad_value=300)

    def test_batch_tile_images(self):
        result = batch_tile_images([self._img(32, 32) for _ in range(3)], TileConfig(16, 16))
        assert len(result) == 3 and all(len(ts) == 4 for ts in result)

    def test_tile_metadata(self):
        t = tile_image(self._img(), TileConfig(32, 32))[0]
        assert t.row == 0 and t.col == 0 and t.source_h == 64


# ─── topology_utils ───────────────────────────────────────────────────────────

class TestTopologyUtils:
    _sq_pts = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)

    def _solid(self, s=20):
        m = np.zeros((s, s), dtype=bool); m[2:-2, 2:-2] = True; return m

    def _holed(self):
        m = self._solid(); m[7:13, 7:13] = False; return m

    def test_euler_number(self):
        assert compute_euler_number(self._solid()) == 1
        assert compute_euler_number(self._holed()) == 0

    def test_count_holes(self):
        assert count_holes(self._solid()) == 0
        assert count_holes(self._holed()) == 1

    def test_compute_solidity_square(self):
        assert compute_solidity(self._sq_pts) == pytest.approx(1.0, abs=0.05)

    def test_compute_extent_square(self):
        assert compute_extent(self._sq_pts) == pytest.approx(1.0, abs=0.05)

    def test_compute_convexity_convex(self):
        assert compute_convexity(self._sq_pts) == pytest.approx(1.0, abs=0.05)

    def test_compute_compactness_circle(self):
        ang = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        pts = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        assert 0.8 <= compute_compactness(pts) <= 1.1

    def test_is_simply_connected(self):
        assert is_simply_connected(self._solid()) is True
        assert is_simply_connected(self._holed()) is False

    def test_shape_complexity(self):
        r = shape_complexity(self._sq_pts)
        assert isinstance(r, float) and 0.0 <= r <= 1.0

    def test_batch_topology(self):
        tri = np.array([[0, 0], [6, 0], [3, 5]], dtype=float)
        results = batch_topology([self._sq_pts, tri])
        assert len(results) == 2 and "solidity" in results[0]

    def test_contour_invalid_raises(self):
        with pytest.raises(ValueError):
            compute_solidity(np.array([[0, 0], [1, 1]], dtype=float))  # < 3 pts


# ─── tracker_utils ────────────────────────────────────────────────────────────

class TestTrackerUtils:
    def _tracker(self, n=10):
        t = create_iter_tracker()
        for i in range(n):
            record_step(t, i, float(i))
        return t

    def test_create_and_record(self):
        t = create_iter_tracker()
        assert len(t.records) == 0
        record_step(t, 0, 1.5)
        assert t.records[0].value == pytest.approx(1.5)

    def test_get_values_and_steps(self):
        t = self._tracker(5)
        np.testing.assert_array_equal(get_values(t), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(get_steps(t), [0, 1, 2, 3, 4])

    def test_best_worst_record(self):
        t = self._tracker(5)
        assert get_best_record(t).value == pytest.approx(4.0)
        assert get_worst_record(t).value == pytest.approx(0.0)

    def test_compute_delta(self):
        np.testing.assert_array_equal(compute_delta(self._tracker(5), lag=1), [1, 1, 1, 1])

    def test_is_improving(self):
        assert is_improving(self._tracker(10), window=5) is True

    def test_find_plateau_start(self):
        t = self._tracker(5)
        for i in range(5, 12):
            record_step(t, i, 5.0)
        start = find_plateau_start(t, window=3, tol=1e-9)
        assert start is not None and start >= 5

    def test_smooth_values(self):
        smoothed = smooth_values(np.array([1.0, 3.0, 1.0, 3.0]), window=2)
        assert len(smoothed) == 4

    def test_tracker_stats(self):
        s = tracker_stats(self._tracker(10))
        assert s["n"] == 10 and s["min"] == pytest.approx(0.0) and s["max"] == pytest.approx(9.0)

    def test_compare_trackers(self):
        t2 = create_iter_tracker()
        for i in range(5):
            record_step(t2, i, float(i) * 2)
        assert "winner" in compare_trackers(self._tracker(5), t2)

    def test_keep_history_false(self):
        cfg = TrackerConfig(keep_history=False)
        t = create_iter_tracker(cfg)
        record_step(t, 0, 1.0)
        record_step(t, 1, 2.0)
        assert len(t.records) == 1 and t.records[0].value == pytest.approx(2.0)


# ─── voting_utils ─────────────────────────────────────────────────────────────

class TestVotingUtils:
    def test_cast_pair_votes_basic(self):
        votes = cast_pair_votes([[(0, 1), (1, 2)], [(1, 0), (2, 3)]])
        assert votes[(0, 1)] == pytest.approx(2.0)

    def test_cast_pair_votes_weighted(self):
        votes = cast_pair_votes([[(0, 1)], [(0, 1)]], weights=[1.0, 2.0])
        assert votes[(0, 1)] == pytest.approx(3.0)

    def test_cast_pair_votes_weight_mismatch_raises(self):
        with pytest.raises(ValueError):
            cast_pair_votes([[(0, 1)]], weights=[1.0, 2.0])

    def test_aggregate_pair_votes(self):
        result = aggregate_pair_votes({(0, 1): 3.0, (1, 2): 1.0})
        assert result[0][0] == (0, 1) and result[0][1] == pytest.approx(1.0)

    def test_aggregate_min_votes_filter(self):
        result = aggregate_pair_votes({(0, 1): 3.0, (1, 2): 1.0}, cfg=VoteConfig(min_votes=2))
        assert len(result) == 1

    def test_cast_position_votes(self):
        r = cast_position_votes([{0: 0.9, 1: 0.5}, {0: 0.8, 2: 0.3}])
        assert r[0] == pytest.approx(1.7) and r[2] == pytest.approx(0.3)

    def test_majority_vote(self):
        assert majority_vote(["a", "b", "a", "c", "a"]) == "a"
        assert majority_vote([]) is None

    def test_weighted_vote(self):
        assert weighted_vote([1.0, 3.0]) == pytest.approx(2.0)
        assert weighted_vote([1.0, 3.0], weights=[3.0, 1.0]) == pytest.approx(1.5)

    def test_rank_fusion_order(self):
        result = rank_fusion([["a", "b", "c"]])
        assert result[0][0] == "a"

    def test_batch_vote(self):
        results = batch_vote([[[(0, 1), (1, 2)], [(0, 1)]], [[(2, 3)]]])
        assert len(results) == 2


# ─── window_utils ─────────────────────────────────────────────────────────────

class TestWindowUtils:
    def _sig(self, n=20):
        return rng.random(n)

    def test_apply_window_rect_unchanged(self):
        s = self._sig(10)
        np.testing.assert_array_equal(apply_window_function(s, WindowConfig(size=10, func="rect")), s)

    def test_apply_window_hann_tapers(self):
        result = apply_window_function(np.ones(8), WindowConfig(size=8, func="hann"))
        assert result[0] < 0.1 and result[4] > 0.9

    def test_rolling_mean_lengths(self):
        s = self._sig(20)
        assert len(rolling_mean(s, WindowConfig(size=4, padding="same"))) == 20
        assert len(rolling_mean(s, WindowConfig(size=4, step=1, padding="valid"))) == 17

    def test_rolling_std_non_negative(self):
        assert np.all(rolling_std(self._sig(20), WindowConfig(size=3, padding="same")) >= 0)

    def test_rolling_max_geq_min(self):
        s = self._sig(20)
        cfg = WindowConfig(size=4, padding="same")
        assert np.all(rolling_max(s, cfg) >= rolling_min(s, cfg) - 1e-9)

    def test_compute_overlap(self):
        assert compute_overlap(0, 10, 0, 10) == pytest.approx(1.0)
        assert compute_overlap(0, 5, 10, 15) == pytest.approx(0.0)
        assert 0.0 < compute_overlap(0, 10, 5, 15) < 1.0

    def test_split_into_windows_count(self):
        cfg = WindowConfig(size=5, step=5, padding="valid")
        windows = split_into_windows(np.arange(20, dtype=float), cfg)
        assert len(windows) == 4 and all(len(w) == 5 for w in windows)

    def test_merge_windows_roundtrip(self):
        cfg = WindowConfig(size=5, step=5, padding="valid")
        windows = split_into_windows(np.ones(20, dtype=float), cfg)
        merged = merge_windows(windows, 20, cfg)
        assert len(merged) == 20
        np.testing.assert_allclose(merged, 1.0, atol=1e-10)

    def test_batch_rolling(self):
        cfg = WindowConfig(size=3, padding="same")
        results = batch_rolling([self._sig(15) for _ in range(4)], stat="mean", cfg=cfg)
        assert len(results) == 4 and all(len(r) == 15 for r in results)

    def test_window_config_invalid_raises(self):
        with pytest.raises(ValueError):
            WindowConfig(size=0)
        with pytest.raises(ValueError):
            WindowConfig(func="boxcar")
