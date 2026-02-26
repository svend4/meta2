"""Integration tests for utils batch 6b:
threshold_utils, tile_utils, topology_utils,
tracker_utils, voting_utils, window_utils.
"""
import numpy as np
import pytest

from puzzle_reconstruction.utils.threshold_utils import (
    ThresholdConfig,
    apply_threshold,
    binarize,
    adaptive_threshold,
    soft_threshold,
    threshold_matrix,
    hysteresis_threshold,
    otsu_threshold,
    count_above,
    fraction_above,
    batch_threshold,
)
from puzzle_reconstruction.utils.tile_utils import (
    TileConfig,
    Tile,
    tile_image,
    reassemble_tiles,
    tile_overlap_ratio,
    filter_tiles_by_content,
    compute_tile_grid,
    batch_tile_images,
)
from puzzle_reconstruction.utils.topology_utils import (
    TopologyConfig,
    compute_euler_number,
    count_holes,
    compute_solidity,
    compute_extent,
    compute_convexity,
    compute_compactness,
    is_simply_connected,
    shape_complexity,
    batch_topology,
)
from puzzle_reconstruction.utils.tracker_utils import (
    TrackerConfig,
    StepRecord,
    IterTracker,
    create_iter_tracker,
    record_step,
    get_values,
    get_steps,
    get_best_record,
    get_worst_record,
    compute_delta,
    is_improving,
    find_plateau_start,
    smooth_values,
    tracker_stats,
    compare_trackers,
)
from puzzle_reconstruction.utils.voting_utils import (
    VoteConfig,
    cast_pair_votes,
    aggregate_pair_votes,
    cast_position_votes,
    majority_vote,
    weighted_vote,
    rank_fusion,
    batch_vote,
)
from puzzle_reconstruction.utils.window_utils import (
    WindowConfig,
    apply_window_function,
    rolling_mean,
    rolling_std,
    rolling_max,
    rolling_min,
    compute_overlap,
    split_into_windows,
    merge_windows,
    batch_rolling,
)

rng = np.random.default_rng(42)


# ─── threshold_utils ──────────────────────────────────────────────────────────

class TestThresholdUtils:
    def test_apply_threshold_basic(self):
        arr = np.array([0.1, 0.5, 0.9])
        mask = apply_threshold(arr, 0.4)
        assert mask.dtype == bool
        assert list(mask) == [False, True, True]

    def test_apply_threshold_invert(self):
        arr = np.array([0.1, 0.5, 0.9])
        mask = apply_threshold(arr, 0.6, invert=True)
        assert list(mask) == [True, True, False]

    def test_apply_threshold_empty_raises(self):
        with pytest.raises(ValueError):
            apply_threshold(np.array([]), 0.5)

    def test_binarize_values(self):
        arr = np.array([0.2, 0.8, 0.5])
        result = binarize(arr, 0.5)
        assert result.dtype == np.float64
        assert list(result) == [0.0, 1.0, 1.0]

    def test_binarize_invert(self):
        arr = np.array([0.2, 0.8])
        result = binarize(arr, 0.5, invert=True)
        assert list(result) == [1.0, 0.0]

    def test_adaptive_threshold_flat_signal(self):
        arr = np.ones(10) * 0.5
        result = adaptive_threshold(arr, window=3, offset=0.0)
        # all elements equal local mean: >= local_mean => True
        assert all(result)

    def test_adaptive_threshold_non_1d_raises(self):
        with pytest.raises(ValueError):
            adaptive_threshold(np.ones((3, 3)), window=3)

    def test_soft_threshold_shrinkage(self):
        arr = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = soft_threshold(arr, 1.0)
        expected = np.array([-1.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_threshold_matrix_fill(self):
        m = np.array([[0.1, 0.8], [0.3, 0.6]])
        result = threshold_matrix(m, 0.5, fill=-1.0)
        assert result[0, 0] == -1.0
        assert result[0, 1] == 0.8

    def test_hysteresis_threshold_propagation(self):
        arr = np.array([0.9, 0.4, 0.4, 0.1])
        result = hysteresis_threshold(arr, low=0.3, high=0.8)
        assert result[0] is np.bool_(True)
        assert result[1] is np.bool_(True)   # weak but adjacent to strong
        assert result[3] is np.bool_(False)  # below low

    def test_otsu_threshold_bimodal(self):
        low = rng.uniform(0.0, 0.3, 50)
        high = rng.uniform(0.7, 1.0, 50)
        arr = np.concatenate([low, high])
        t = otsu_threshold(arr)
        # Threshold should lie between the two clusters
        assert float(low.max()) < t < float(high.min()) + 0.1

    def test_count_above_and_fraction(self):
        arr = np.array([0.1, 0.5, 0.9, 0.7])
        assert count_above(arr, 0.5) == 3   # 0.5, 0.9, 0.7 >= 0.5
        assert fraction_above(arr, 0.6) == pytest.approx(0.5)

    def test_batch_threshold(self):
        arrays = [np.array([0.1, 0.9]), np.array([0.4, 0.6])]
        masks = batch_threshold(arrays, 0.5)
        assert len(masks) == 2
        assert list(masks[0]) == [False, True]


# ─── tile_utils ───────────────────────────────────────────────────────────────

class TestTileUtils:
    def _gray_image(self, h=128, w=128):
        return rng.integers(0, 256, (h, w), dtype=np.uint8)

    def test_tile_image_no_overlap(self):
        img = self._gray_image(64, 64)
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(img, cfg)
        assert len(tiles) == 4
        for t in tiles:
            assert t.data.shape == (32, 32)

    def test_tile_image_padding_applied(self):
        img = self._gray_image(70, 70)
        cfg = TileConfig(tile_h=64, tile_w=64, pad_value=128)
        tiles = tile_image(img, cfg)
        # Should have 4 tiles (2x2 grid), edge tiles padded
        assert len(tiles) == 4
        for t in tiles:
            assert t.data.shape == (64, 64)

    def test_tile_image_color(self):
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        tiles = tile_image(img, TileConfig(tile_h=32, tile_w=32))
        assert all(t.data.shape == (32, 32, 3) for t in tiles)

    def test_reassemble_tiles_roundtrip(self):
        img = self._gray_image(64, 64)
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(img, cfg)
        restored = reassemble_tiles(tiles, img.shape[:2], cfg)
        np.testing.assert_array_equal(restored, img)

    def test_compute_tile_grid_count(self):
        cfg = TileConfig(tile_h=16, tile_w=16)
        grid = compute_tile_grid(32, 32, cfg)
        assert len(grid) == 4  # 2x2

    def test_tile_overlap_ratio_identical(self):
        cfg = TileConfig(tile_h=32, tile_w=32)
        img = self._gray_image(64, 64)
        tiles = tile_image(img, cfg)
        ratio = tile_overlap_ratio(tiles[0], tiles[0])
        assert ratio == pytest.approx(1.0)

    def test_tile_overlap_ratio_no_overlap(self):
        cfg = TileConfig(tile_h=32, tile_w=32)
        img = self._gray_image(64, 64)
        tiles = tile_image(img, cfg)
        # tiles[0] is top-left, tiles[3] is bottom-right: no overlap
        ratio = tile_overlap_ratio(tiles[0], tiles[3])
        assert ratio == pytest.approx(0.0)

    def test_filter_tiles_by_content(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[32:, 32:] = 200
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(img, cfg)
        filtered = filter_tiles_by_content(tiles, min_fraction=0.3)
        # Only one quadrant has sufficient content
        assert len(filtered) == 1

    def test_tile_config_validation(self):
        with pytest.raises(ValueError):
            TileConfig(tile_h=-1)
        with pytest.raises(ValueError):
            TileConfig(pad_value=300)

    def test_batch_tile_images(self):
        images = [self._gray_image(32, 32) for _ in range(3)]
        cfg = TileConfig(tile_h=16, tile_w=16)
        result = batch_tile_images(images, cfg)
        assert len(result) == 3
        assert all(len(ts) == 4 for ts in result)

    def test_tile_metadata(self):
        img = self._gray_image(64, 64)
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(img, cfg)
        t = tiles[0]
        assert t.row == 0 and t.col == 0
        assert t.y == 0 and t.x == 0
        assert t.source_h == 64 and t.source_w == 64


# ─── topology_utils ───────────────────────────────────────────────────────────

class TestTopologyUtils:
    def _solid_square(self, size=20):
        m = np.zeros((size, size), dtype=bool)
        m[2:-2, 2:-2] = True
        return m

    def _square_with_hole(self):
        m = np.zeros((20, 20), dtype=bool)
        m[2:-2, 2:-2] = True
        m[7:13, 7:13] = False
        return m

    def test_euler_number_solid(self):
        mask = self._solid_square()
        assert compute_euler_number(mask) == 1

    def test_euler_number_with_hole(self):
        mask = self._square_with_hole()
        euler = compute_euler_number(mask)
        assert euler == 0  # 1 component - 1 hole

    def test_count_holes_none(self):
        mask = self._solid_square()
        assert count_holes(mask) == 0

    def test_count_holes_one(self):
        mask = self._square_with_hole()
        assert count_holes(mask) == 1

    def test_compute_solidity_square(self):
        pts = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
        sol = compute_solidity(pts)
        assert sol == pytest.approx(1.0, abs=0.05)

    def test_compute_extent_square(self):
        pts = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
        ext = compute_extent(pts)
        assert ext == pytest.approx(1.0, abs=0.05)

    def test_compute_convexity_convex(self):
        pts = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
        conv = compute_convexity(pts)
        assert conv == pytest.approx(1.0, abs=0.05)

    def test_compute_compactness_circle_approx(self):
        angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        c = compute_compactness(pts)
        assert 0.8 <= c <= 1.1  # circle is ~1.0

    def test_is_simply_connected_solid(self):
        mask = self._solid_square()
        assert is_simply_connected(mask) is True

    def test_is_simply_connected_with_hole(self):
        mask = self._square_with_hole()
        assert is_simply_connected(mask) is False

    def test_shape_complexity_returns_dict(self):
        pts = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
        result = shape_complexity(pts)
        assert isinstance(result, dict)

    def test_batch_topology(self):
        masks = [self._solid_square(), self._square_with_hole()]
        results = batch_topology(masks)
        assert len(results) == 2


# ─── tracker_utils ────────────────────────────────────────────────────────────

class TestTrackerUtils:
    def _filled_tracker(self, n=10):
        t = create_iter_tracker()
        for i in range(n):
            record_step(t, step=i, value=float(i))
        return t

    def test_create_tracker_empty(self):
        t = create_iter_tracker()
        assert len(t.records) == 0

    def test_record_step_appends(self):
        t = create_iter_tracker()
        record_step(t, 0, 1.5)
        assert len(t.records) == 1
        assert t.records[0].value == pytest.approx(1.5)

    def test_get_values(self):
        t = self._filled_tracker(5)
        vals = get_values(t)
        np.testing.assert_array_equal(vals, [0, 1, 2, 3, 4])

    def test_get_steps(self):
        t = self._filled_tracker(5)
        steps = get_steps(t)
        np.testing.assert_array_equal(steps, [0, 1, 2, 3, 4])

    def test_get_best_record(self):
        t = self._filled_tracker(5)
        best = get_best_record(t)
        assert best.value == pytest.approx(4.0)

    def test_get_worst_record(self):
        t = self._filled_tracker(5)
        worst = get_worst_record(t)
        assert worst.value == pytest.approx(0.0)

    def test_compute_delta(self):
        t = self._filled_tracker(5)
        deltas = compute_delta(t, lag=1)
        np.testing.assert_array_equal(deltas, [1, 1, 1, 1])

    def test_is_improving_true(self):
        t = self._filled_tracker(10)
        assert is_improving(t, window=5) is True

    def test_find_plateau_start(self):
        t = create_iter_tracker()
        for i in range(5):
            record_step(t, i, float(i))
        for i in range(5, 12):
            record_step(t, i, 5.0)
        start = find_plateau_start(t, window=3, tol=1e-9)
        assert start is not None
        assert start >= 5

    def test_smooth_values(self):
        vals = np.array([1.0, 3.0, 1.0, 3.0], dtype=np.float64)
        smoothed = smooth_values(vals, window=2)
        assert len(smoothed) == len(vals)

    def test_tracker_stats(self):
        t = self._filled_tracker(10)
        stats = tracker_stats(t)
        assert stats["n"] == 10
        assert stats["min"] == pytest.approx(0.0)
        assert stats["max"] == pytest.approx(9.0)

    def test_compare_trackers(self):
        t1 = self._filled_tracker(5)
        t2 = create_iter_tracker()
        for i in range(5):
            record_step(t2, i, float(i) * 2)
        cmp = compare_trackers(t1, t2)
        assert "winner" in cmp


# ─── voting_utils ─────────────────────────────────────────────────────────────

class TestVotingUtils:
    def test_cast_pair_votes_basic(self):
        lists = [[(0, 1), (1, 2)], [(1, 0), (2, 3)]]
        votes = cast_pair_votes(lists)
        assert votes[(0, 1)] == pytest.approx(2.0)
        assert votes[(1, 2)] == pytest.approx(1.0)

    def test_cast_pair_votes_weighted(self):
        lists = [[(0, 1)], [(0, 1)]]
        votes = cast_pair_votes(lists, weights=[1.0, 2.0])
        assert votes[(0, 1)] == pytest.approx(3.0)

    def test_cast_pair_votes_weight_mismatch(self):
        with pytest.raises(ValueError):
            cast_pair_votes([[(0, 1)]], weights=[1.0, 2.0])

    def test_aggregate_pair_votes_sorted(self):
        votes = {(0, 1): 3.0, (1, 2): 1.0}
        result = aggregate_pair_votes(votes)
        assert result[0][0] == (0, 1)
        assert result[0][1] == pytest.approx(1.0)  # normalized

    def test_aggregate_pair_votes_min_filter(self):
        votes = {(0, 1): 3.0, (1, 2): 1.0}
        cfg = VoteConfig(min_votes=2)
        result = aggregate_pair_votes(votes, cfg=cfg)
        assert len(result) == 1

    def test_cast_position_votes(self):
        plists = [{0: 0.9, 1: 0.5}, {0: 0.8, 2: 0.3}]
        result = cast_position_votes(plists)
        assert result[0] == pytest.approx(1.7)
        assert result[2] == pytest.approx(0.3)

    def test_majority_vote_simple(self):
        assert majority_vote(["a", "b", "a", "c", "a"]) == "a"

    def test_majority_vote_empty(self):
        assert majority_vote([]) is None

    def test_weighted_vote_equal_weights(self):
        result = weighted_vote([1.0, 3.0])
        assert result == pytest.approx(2.0)

    def test_weighted_vote_custom_weights(self):
        result = weighted_vote([1.0, 3.0], weights=[3.0, 1.0])
        assert result == pytest.approx(1.5)

    def test_rank_fusion_single_list(self):
        result = rank_fusion([["a", "b", "c"]])
        items = [r[0] for r in result]
        assert items[0] == "a"

    def test_batch_vote(self):
        batch = [[[(0, 1), (1, 2)], [(0, 1)]], [[(2, 3)]]]
        results = batch_vote(batch)
        assert len(results) == 2


# ─── window_utils ─────────────────────────────────────────────────────────────

class TestWindowUtils:
    def _signal(self, n=20):
        return rng.random(n)

    def test_apply_window_function_rect(self):
        s = self._signal(10)
        cfg = WindowConfig(size=10, func="rect")
        result = apply_window_function(s, cfg)
        np.testing.assert_array_equal(result, s)

    def test_apply_window_function_hann(self):
        s = np.ones(8)
        cfg = WindowConfig(size=8, func="hann")
        result = apply_window_function(s, cfg)
        assert result[0] < 0.1
        assert result[4] > 0.9

    def test_rolling_mean_same_length(self):
        s = self._signal(20)
        result = rolling_mean(s, WindowConfig(size=4, padding="same"))
        assert len(result) == 20

    def test_rolling_mean_valid_length(self):
        s = self._signal(20)
        cfg = WindowConfig(size=4, step=1, padding="valid")
        result = rolling_mean(s, cfg)
        assert len(result) == 20 - 4 + 1

    def test_rolling_std_non_negative(self):
        s = self._signal(20)
        result = rolling_std(s, WindowConfig(size=3, padding="same"))
        assert np.all(result >= 0)

    def test_rolling_max_geq_mean(self):
        s = self._signal(20)
        cfg = WindowConfig(size=4, padding="same")
        assert np.all(rolling_max(s, cfg) >= rolling_mean(s, cfg) - 1e-9)

    def test_rolling_min_leq_mean(self):
        s = self._signal(20)
        cfg = WindowConfig(size=4, padding="same")
        assert np.all(rolling_min(s, cfg) <= rolling_mean(s, cfg) + 1e-9)

    def test_compute_overlap_full(self):
        assert compute_overlap(0, 10, 0, 10) == pytest.approx(1.0)

    def test_compute_overlap_none(self):
        assert compute_overlap(0, 5, 10, 15) == pytest.approx(0.0)

    def test_compute_overlap_partial(self):
        ratio = compute_overlap(0, 10, 5, 15)
        assert 0.0 < ratio < 1.0

    def test_split_into_windows_count(self):
        s = np.arange(20, dtype=float)
        cfg = WindowConfig(size=5, step=5, padding="valid")
        windows = split_into_windows(s, cfg)
        assert len(windows) == 4
        assert all(len(w) == 5 for w in windows)

    def test_merge_windows_roundtrip(self):
        s = np.ones(20, dtype=float)
        cfg = WindowConfig(size=5, step=5, padding="valid")
        windows = split_into_windows(s, cfg)
        merged = merge_windows(windows, len(s), cfg)
        assert len(merged) == len(s)
        np.testing.assert_allclose(merged, 1.0, atol=1e-10)

    def test_batch_rolling(self):
        signals = [self._signal(15) for _ in range(4)]
        cfg = WindowConfig(size=3, padding="same")
        results = batch_rolling(signals, cfg, stat="mean")
        assert len(results) == 4
        assert all(len(r) == 15 for r in results)
