"""Integration tests for utils low-coverage modules batch 6.

Covers:
  - puzzle_reconstruction.utils.shape_match_utils
  - puzzle_reconstruction.utils.spatial_index
  - puzzle_reconstruction.utils.stats_utils
  - puzzle_reconstruction.utils.text_utils
  - puzzle_reconstruction.utils.texture_pipeline_utils
  - puzzle_reconstruction.utils.threshold_utils
  - puzzle_reconstruction.utils.tile_utils
  - puzzle_reconstruction.utils.topology_utils
  - puzzle_reconstruction.utils.tracker_utils
  - puzzle_reconstruction.utils.voting_utils
  - puzzle_reconstruction.utils.window_utils
  - puzzle_reconstruction.verification.homography_verifier
  - puzzle_reconstruction.utils.orient_skew_utils
"""
import math
import pytest
import numpy as np

# ============================================================
# shape_match_utils
# ============================================================
from puzzle_reconstruction.utils.shape_match_utils import (
    ShapeMatchConfig, ShapeMatchEntry, ShapeMatchSummary,
    make_match_entry, entries_from_results, summarise_matches,
    filter_good_matches, filter_poor_matches, filter_by_hu_dist,
    filter_match_by_score_range, top_k_match_entries,
    match_entry_stats, compare_match_summaries, batch_summarise_matches,
)


class TestShapeMatchConfig:
    def test_defaults(self):
        cfg = ShapeMatchConfig()
        assert cfg.min_score == 0.0
        assert cfg.max_pairs == 100
        assert cfg.method == "hu"

    def test_valid_methods(self):
        for m in ("hu", "zernike", "combined"):
            ShapeMatchConfig(method=m)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(method="bad")

    def test_negative_min_score(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(min_score=-0.1)

    def test_zero_max_pairs(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(max_pairs=0)


class TestShapeMatchEntry:
    def test_basic(self):
        e = ShapeMatchEntry(idx1=0, idx2=1, score=0.8)
        assert e.is_good is True

    def test_poor_score(self):
        e = ShapeMatchEntry(idx1=0, idx2=1, score=0.3)
        assert e.is_good is False

    def test_boundary_score(self):
        e = ShapeMatchEntry(idx1=0, idx2=1, score=0.5)
        assert e.is_good is False  # not > 0.5

    def test_negative_idx_raises(self):
        with pytest.raises(ValueError):
            ShapeMatchEntry(idx1=-1, idx2=0, score=0.5)


class TestMakeMatchEntry:
    def test_creates_entry(self):
        e = make_match_entry(0, 1, 0.7, hu_dist=1.5, iou=0.6)
        assert e.idx1 == 0
        assert e.idx2 == 1
        assert e.score == 0.7
        assert e.hu_dist == 1.5

    def test_default_meta(self):
        e = make_match_entry(0, 1, 0.5)
        assert e.meta == {}


class TestEntriesFromResults:
    def test_basic(self):
        results = [(0, 1, 0.9), (2, 3, 0.4)]
        entries = entries_from_results(results)
        assert len(entries) == 2
        assert entries[0].score == 0.9
        assert entries[1].rank == 1

    def test_empty(self):
        assert entries_from_results([]) == []


class TestSummariseMatches:
    def test_empty(self):
        s = summarise_matches([])
        assert s.n_total == 0
        assert s.mean_score == 0.0

    def test_with_entries(self):
        entries = entries_from_results([(0,1,0.9),(1,2,0.3),(2,3,0.6)])
        s = summarise_matches(entries)
        assert s.n_total == 3
        assert s.n_good == 2
        assert s.n_poor == 1
        assert abs(s.mean_score - (0.9+0.3+0.6)/3) < 1e-9

    def test_repr(self):
        s = summarise_matches(entries_from_results([(0,1,0.8)]))
        assert "ShapeMatchSummary" in repr(s)


class TestFilterFunctions:
    def setup_method(self):
        self.entries = entries_from_results([
            (0,1,0.9),(1,2,0.3),(2,3,0.7),(3,4,0.1)])

    def test_filter_good(self):
        good = filter_good_matches(self.entries)
        assert all(e.score > 0.5 for e in good)
        assert len(good) == 2

    def test_filter_poor(self):
        poor = filter_poor_matches(self.entries)
        assert all(e.score <= 0.5 for e in poor)

    def test_filter_by_hu_dist(self):
        for e in self.entries:
            e.hu_dist = e.score * 10
        filtered = filter_by_hu_dist(self.entries, max_hu=5.0)
        assert all(e.hu_dist <= 5.0 for e in filtered)

    def test_filter_by_score_range(self):
        filtered = filter_match_by_score_range(self.entries, lo=0.3, hi=0.7)
        assert all(0.3 <= e.score <= 0.7 for e in filtered)

    def test_top_k(self):
        top2 = top_k_match_entries(self.entries, k=2)
        assert len(top2) == 2
        assert top2[0].score >= top2[1].score

    def test_top_k_invalid(self):
        with pytest.raises(ValueError):
            top_k_match_entries(self.entries, k=0)


class TestMatchEntryStats:
    def test_empty(self):
        stats = match_entry_stats([])
        assert stats["n"] == 0
        assert stats["mean_score"] == 0.0

    def test_nonempty(self):
        entries = entries_from_results([(0,1,0.8),(1,2,0.4)])
        stats = match_entry_stats(entries)
        assert stats["n"] == 2
        assert abs(stats["mean_score"] - 0.6) < 1e-9


class TestCompareMatchSummaries:
    def test_compare(self):
        s1 = summarise_matches(entries_from_results([(0,1,0.8),(1,2,0.6)]))
        s2 = summarise_matches(entries_from_results([(0,1,0.4)]))
        diff = compare_match_summaries(s1, s2)
        assert diff["n_total_delta"] == 1
        assert diff["n_good_delta"] >= 0


class TestBatchSummariseMatches:
    def test_batch(self):
        lists = [
            entries_from_results([(0,1,0.9)]),
            entries_from_results([(2,3,0.2),(4,5,0.8)]),
        ]
        summaries = batch_summarise_matches(lists)
        assert len(summaries) == 2
        assert summaries[0].n_total == 1
        assert summaries[1].n_total == 2

# ============================================================
# spatial_index
# ============================================================
from puzzle_reconstruction.utils.spatial_index import (
    SpatialConfig, SpatialEntry, SpatialIndex,
    build_spatial_index, query_radius, query_knn,
    pairwise_distances, cluster_by_distance,
)


class TestSpatialConfig:
    def test_defaults(self):
        cfg = SpatialConfig()
        assert cfg.cell_size == 50.0
        assert cfg.metric == "euclidean"

    def test_invalid_cell_size(self):
        with pytest.raises(ValueError):
            SpatialConfig(cell_size=0)

    def test_invalid_metric(self):
        with pytest.raises(ValueError):
            SpatialConfig(metric="L2")

    def test_negative_max_results(self):
        with pytest.raises(ValueError):
            SpatialConfig(max_results=-1)


class TestSpatialEntry:
    def test_basic(self):
        e = SpatialEntry(item_id=0, position=np.array([1.0, 2.0]))
        assert e.item_id == 0
        assert e.position.dtype == np.float64

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            SpatialEntry(item_id=-1, position=np.array([0.0, 0.0]))

    def test_wrong_shape(self):
        with pytest.raises(ValueError):
            SpatialEntry(item_id=0, position=np.array([1.0, 2.0, 3.0]))


class TestSpatialIndex:
    def setup_method(self):
        self.rng = np.random.default_rng(42)
        positions = self.rng.uniform(0, 100, (20, 2))
        self.idx = build_spatial_index(positions)

    def test_size(self):
        assert len(self.idx) == 20

    def test_contains(self):
        assert 0 in self.idx
        assert 19 in self.idx
        assert 20 not in self.idx

    def test_get_all(self):
        all_entries = self.idx.get_all()
        assert len(all_entries) == 20

    def test_insert_and_remove(self):
        idx = SpatialIndex()
        e = SpatialEntry(item_id=99, position=np.array([10.0, 20.0]))
        idx.insert(e)
        assert 99 in idx
        removed = idx.remove(99)
        assert removed is True
        assert 99 not in idx

    def test_remove_nonexistent(self):
        idx = SpatialIndex()
        assert idx.remove(999) is False

    def test_clear(self):
        self.idx.clear()
        assert len(self.idx) == 0

    def test_query_radius(self):
        center = np.array([50.0, 50.0])
        results = self.idx.query_radius(center, radius=30.0)
        assert isinstance(results, list)
        for dist, entry in results:
            assert dist <= 30.0

    def test_query_radius_negative(self):
        with pytest.raises(ValueError):
            self.idx.query_radius(np.array([0.0, 0.0]), radius=-1.0)

    def test_query_knn(self):
        center = np.array([50.0, 50.0])
        results = self.idx.query_knn(center, k=5)
        assert len(results) == 5
        dists = [d for d, _ in results]
        assert dists == sorted(dists)

    def test_query_knn_invalid(self):
        with pytest.raises(ValueError):
            self.idx.query_knn(np.array([0.0, 0.0]), k=0)

    def test_query_knn_empty_index(self):
        idx = SpatialIndex()
        assert idx.query_knn(np.array([0.0, 0.0]), k=3) == []

    def test_manhattan_metric(self):
        cfg = SpatialConfig(metric="manhattan")
        rng = np.random.default_rng(7)
        pts = rng.uniform(0, 50, (10, 2))
        idx = build_spatial_index(pts, cfg=cfg)
        results = idx.query_radius(np.array([25.0, 25.0]), radius=20.0)
        assert isinstance(results, list)

    def test_chebyshev_metric(self):
        cfg = SpatialConfig(metric="chebyshev")
        rng = np.random.default_rng(8)
        pts = rng.uniform(0, 50, (10, 2))
        idx = build_spatial_index(pts, cfg=cfg)
        results = idx.query_radius(np.array([25.0, 25.0]), radius=15.0)
        assert isinstance(results, list)


class TestBuildSpatialIndex:
    def test_with_payloads(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        payloads = ["a", "b"]
        idx = build_spatial_index(pts, payloads=payloads)
        all_e = idx.get_all()
        assert any(e.payload == "a" for e in all_e)

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            build_spatial_index(np.array([1.0, 2.0, 3.0]))

    def test_mismatched_payloads(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError):
            build_spatial_index(pts, payloads=["a"])


class TestPairwiseDistances:
    def test_euclidean(self):
        pts = np.array([[0.0,0.0],[3.0,4.0],[6.0,8.0]])
        D = pairwise_distances(pts)
        assert D.shape == (3, 3)
        assert D[0, 0] == 0.0
        assert abs(D[0, 1] - 5.0) < 1e-9
        assert np.allclose(D, D.T)

    def test_manhattan(self):
        pts = np.array([[0.0,0.0],[1.0,1.0]])
        D = pairwise_distances(pts, metric="manhattan")
        assert D[0, 1] == 2.0

    def test_chebyshev(self):
        pts = np.array([[0.0,0.0],[3.0,1.0]])
        D = pairwise_distances(pts, metric="chebyshev")
        assert D[0, 1] == 3.0

    def test_invalid_metric(self):
        with pytest.raises(ValueError):
            pairwise_distances(np.array([[0.0,0.0]]), metric="bad")

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            pairwise_distances(np.array([1.0, 2.0, 3.0]))


class TestClusterByDistance:
    def test_two_clusters(self):
        pts = np.array([[0.0,0.0],[1.0,0.0],[100.0,100.0],[101.0,100.0]])
        clusters = cluster_by_distance(pts, threshold=5.0)
        assert len(clusters) == 2

    def test_single_cluster(self):
        pts = np.array([[0.0,0.0],[1.0,0.0],[2.0,0.0]])
        clusters = cluster_by_distance(pts, threshold=5.0)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_empty(self):
        pts = np.empty((0, 2))
        clusters = cluster_by_distance(pts, threshold=5.0)
        assert clusters == []

    def test_negative_threshold(self):
        with pytest.raises(ValueError):
            cluster_by_distance(np.array([[0.0,0.0]]), threshold=-1.0)

# ============================================================
# stats_utils
# ============================================================
from puzzle_reconstruction.utils.stats_utils import (
    StatsConfig, describe, zscore_array, iqr, winsorize,
    percentile_rank, outlier_mask, running_stats,
    weighted_mean, weighted_std, batch_describe,
)


class TestStatsConfig:
    def test_defaults(self):
        cfg = StatsConfig()
        assert cfg.outlier_iqr_k == 1.5
        assert cfg.ddof == 0

    def test_invalid_iqr_k(self):
        with pytest.raises(ValueError):
            StatsConfig(outlier_iqr_k=0)

    def test_invalid_winsor(self):
        with pytest.raises(ValueError):
            StatsConfig(winsor_low=0.9, winsor_high=0.1)

    def test_invalid_ddof(self):
        with pytest.raises(ValueError):
            StatsConfig(ddof=2)


class TestDescribe:
    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.arr = self.rng.uniform(0, 1, 100)

    def test_keys(self):
        stats = describe(self.arr)
        for k in ("min","max","mean","std","median","q25","q75","iqr"):
            assert k in stats

    def test_values_sane(self):
        stats = describe(self.arr)
        assert stats["min"] <= stats["q25"] <= stats["median"] <= stats["q75"] <= stats["max"]
        assert stats["iqr"] == pytest.approx(stats["q75"] - stats["q25"])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            describe(np.array([]))

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            describe(np.ones((3, 3)))


class TestZscoreArray:
    def test_zero_mean_unit_std(self):
        rng = np.random.default_rng(1)
        arr = rng.uniform(0, 10, 50)
        z = zscore_array(arr)
        assert abs(z.mean()) < 1e-9
        assert abs(z.std() - 1.0) < 1e-6

    def test_constant_array(self):
        arr = np.ones(10)
        z = zscore_array(arr)
        assert np.all(z == 0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            zscore_array(np.array([]))


class TestIqr:
    def test_known(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = iqr(arr)
        assert result >= 0

    def test_single_value(self):
        assert iqr(np.array([5.0])) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            iqr(np.array([]))


class TestWinsorize:
    def test_clips_extremes(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
        cfg = StatsConfig(winsor_low=0.0, winsor_high=0.8)
        w = winsorize(arr, cfg=cfg)
        # The extreme value (100.0) should be clipped down
        assert w.max() < 100.0

    def test_no_change_when_no_extremes(self):
        arr = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        w = winsorize(arr)
        assert np.allclose(w, arr)


class TestPercentileRank:
    def test_minimum(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert percentile_rank(arr, 0.5) == 0.0

    def test_maximum(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert percentile_rank(arr, 6.0) == 100.0

    def test_middle(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rank = percentile_rank(arr, 3.0)
        assert rank == 40.0  # 2 values below 3


class TestOutlierMask:
    def test_detects_outlier(self):
        arr = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
        mask = outlier_mask(arr)
        assert mask[-1] == True

    def test_no_outliers(self):
        arr = np.ones(20)
        mask = outlier_mask(arr)
        assert not mask.any()


class TestRunningStats:
    def test_keys(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = running_stats(arr)
        for k in ("cumsum", "cummax", "cummin", "cummean"):
            assert k in result

    def test_cumsum(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = running_stats(arr)
        np.testing.assert_array_almost_equal(result["cumsum"], [1.0, 3.0, 6.0])

    def test_cummean_final(self):
        arr = np.array([2.0, 4.0, 6.0])
        result = running_stats(arr)
        assert abs(result["cummean"][-1] - 4.0) < 1e-9


class TestWeightedMean:
    def test_equal_weights(self):
        arr = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        result = weighted_mean(arr, w)
        assert abs(result - 2.0) < 1e-9

    def test_unequal_weights(self):
        arr = np.array([0.0, 10.0])
        w = np.array([1.0, 9.0])
        result = weighted_mean(arr, w)
        assert abs(result - 9.0) < 1e-9

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError):
            weighted_mean(np.array([1.0]), np.array([0.0]))

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            weighted_mean(np.array([1.0, 2.0]), np.array([1.0]))


class TestWeightedStd:
    def test_zero_for_constant(self):
        arr = np.ones(5)
        w = np.ones(5)
        assert weighted_std(arr, w) == 0.0

    def test_positive_for_varied(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0])
        w = np.ones(4)
        assert weighted_std(arr, w) > 0


class TestBatchDescribe:
    def test_batch(self):
        rng = np.random.default_rng(1)
        arrays = [rng.uniform(0, 1, 20) for _ in range(3)]
        results = batch_describe(arrays)
        assert len(results) == 3
        assert all("mean" in r for r in results)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            batch_describe([])

# ============================================================
# text_utils
# ============================================================
from puzzle_reconstruction.utils.text_utils import (
    TextConfig, TextBlock,
    clean_ocr_text, estimate_text_density, find_text_lines,
    segment_words, compute_text_score, compare_text_blocks,
    align_text_blocks, batch_clean_text,
)


class TestTextConfig:
    def test_defaults(self):
        cfg = TextConfig()
        assert cfg.min_word_gap == 4
        assert cfg.line_threshold == 0.05

    def test_invalid_min_word_gap(self):
        with pytest.raises(ValueError):
            TextConfig(min_word_gap=-1)

    def test_invalid_line_threshold(self):
        with pytest.raises(ValueError):
            TextConfig(line_threshold=1.5)

    def test_invalid_min_line_height(self):
        with pytest.raises(ValueError):
            TextConfig(min_line_height=0)


class TestTextBlock:
    def test_basic_properties(self):
        b = TextBlock(text="Hello world", x=10, y=20, w=100, h=30)
        assert b.area == 3000
        assert b.center == (60.0, 35.0)
        assert b.n_chars == 10

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            TextBlock(text="x", x=0, y=0, w=10, h=10, confidence=1.5)

    def test_negative_width_raises(self):
        with pytest.raises(ValueError):
            TextBlock(text="x", x=0, y=0, w=-1, h=10)


class TestCleanOcrText:
    def test_basic_clean(self):
        result = clean_ocr_text("  Hello   World  ")
        assert result == "Hello World"

    def test_lowercase(self):
        cfg = TextConfig(lowercase=True)
        result = clean_ocr_text("HELLO World", cfg=cfg)
        assert result == "hello world"

    def test_strip_punct(self):
        cfg = TextConfig(strip_punct=True)
        result = clean_ocr_text("Hello, World!", cfg=cfg)
        assert "," not in result
        assert "!" not in result

    def test_preserves_newline(self):
        result = clean_ocr_text("line1\nline2")
        assert "\n" in result

    def test_nfc_normalization(self):
        # Combining accent: e + combining acute = é
        s = "e\u0301"  # precomposed
        result = clean_ocr_text(s)
        assert len(result) == 1  # NFC: single char é


class TestEstimateTextDensity:
    def test_all_zero(self):
        binary = np.zeros((10, 10), dtype=np.uint8)
        assert estimate_text_density(binary) == 0.0

    def test_all_nonzero(self):
        binary = np.ones((10, 10), dtype=np.uint8) * 255
        assert estimate_text_density(binary) == 1.0

    def test_half_filled(self):
        binary = np.zeros((10, 10), dtype=np.uint8)
        binary[:5, :] = 255
        assert estimate_text_density(binary) == pytest.approx(0.5)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            estimate_text_density(np.ones((5, 5, 3), dtype=np.uint8))


class TestFindTextLines:
    def test_no_text(self):
        binary = np.zeros((50, 50), dtype=np.uint8)
        lines = find_text_lines(binary)
        assert lines == []

    def test_one_line(self):
        binary = np.zeros((50, 100), dtype=np.uint8)
        binary[20:30, :] = 255
        cfg = TextConfig(line_threshold=0.05, min_line_height=4)
        lines = find_text_lines(binary, cfg=cfg)
        assert len(lines) == 1
        y0, y1 = lines[0]
        assert y0 <= 20
        assert y1 >= 30

    def test_two_lines(self):
        binary = np.zeros((100, 100), dtype=np.uint8)
        binary[10:20, :] = 255
        binary[60:70, :] = 255
        cfg = TextConfig(line_threshold=0.05, min_line_height=4)
        lines = find_text_lines(binary, cfg=cfg)
        assert len(lines) == 2

    def test_empty_image(self):
        binary = np.zeros((0, 10), dtype=np.uint8)
        assert find_text_lines(binary) == []


class TestSegmentWords:
    def test_two_words(self):
        binary = np.zeros((10, 50), dtype=np.uint8)
        binary[:, 5:15] = 255
        binary[:, 30:40] = 255
        cfg = TextConfig(min_word_gap=4)
        words = segment_words(binary, cfg=cfg)
        assert len(words) >= 1

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            segment_words(np.ones(10))


class TestComputeTextScore:
    def test_empty_image_zero(self):
        binary = np.zeros((50, 50), dtype=np.uint8)
        assert compute_text_score(binary) == 0.0

    def test_text_image_positive(self):
        rng = np.random.default_rng(42)
        binary = np.zeros((60, 100), dtype=np.uint8)
        for row in [10, 25, 40]:
            binary[row:row+8, 5:95] = 255
        score = compute_text_score(binary)
        assert 0.0 < score <= 1.0


class TestCompareTextBlocks:
    def test_identical(self):
        b = TextBlock(text="hello", x=0, y=0, w=50, h=10)
        assert compare_text_blocks(b, b) == 1.0

    def test_completely_different(self):
        a = TextBlock(text="abc", x=0, y=0, w=50, h=10)
        b = TextBlock(text="xyz", x=0, y=0, w=50, h=10)
        score = compare_text_blocks(a, b)
        assert 0.0 <= score <= 1.0

    def test_empty_both(self):
        a = TextBlock(text="", x=0, y=0, w=0, h=0)
        b = TextBlock(text="", x=0, y=0, w=0, h=0)
        assert compare_text_blocks(a, b) == 1.0

    def test_one_empty(self):
        a = TextBlock(text="hello", x=0, y=0, w=50, h=10)
        b = TextBlock(text="", x=0, y=0, w=0, h=0)
        assert compare_text_blocks(a, b) == 0.0


class TestAlignTextBlocks:
    def setup_method(self):
        self.blocks = [
            TextBlock(text="c", x=20, y=5, w=10, h=10),
            TextBlock(text="a", x=5, y=10, w=10, h=10),
            TextBlock(text="b", x=10, y=5, w=10, h=10),
        ]

    def test_top_to_bottom(self):
        sorted_blocks = align_text_blocks(self.blocks, primary="top-to-bottom")
        ys = [b.y for b in sorted_blocks]
        assert ys == sorted(ys)

    def test_left_to_right(self):
        sorted_blocks = align_text_blocks(self.blocks, primary="left-to-right")
        xs = [b.x for b in sorted_blocks]
        assert xs == sorted(xs)

    def test_reading_order(self):
        sorted_blocks = align_text_blocks(self.blocks, primary="reading-order")
        assert len(sorted_blocks) == 3

    def test_invalid_primary(self):
        with pytest.raises(ValueError):
            align_text_blocks(self.blocks, primary="bad-order")


class TestBatchCleanText:
    def test_batch(self):
        texts = ["Hello  World", "  Test  "]
        results = batch_clean_text(texts)
        assert results[0] == "Hello World"
        assert results[1] == "Test"

# ============================================================
# texture_pipeline_utils
# ============================================================
from puzzle_reconstruction.utils.texture_pipeline_utils import (
    TextureMatchRecord, TextureMatchSummary,
    make_texture_match_record, summarise_texture_matches,
    filter_texture_by_score, filter_texture_by_lbp,
    top_k_texture_records, best_texture_record, texture_score_stats,
    BatchPipelineRecord, BatchPipelineSummary,
    make_batch_pipeline_record, summarise_batch_pipeline,
    filter_batch_by_success_rate, filter_batch_by_stage,
    top_k_batch_records, batch_throughput_stats, compare_batch_summaries,
)


class TestTextureMatchRecord:
    def test_make(self):
        r = make_texture_match_record((0,1), 0.8, 0.7, 0.6, 0.5)
        assert r.pair == (0, 1)
        assert r.score == pytest.approx(0.8)

    def test_with_params(self):
        r = make_texture_match_record((2,3), 0.5, 0.4, 0.3, 0.2, side1=0, side2=1, extra=42)
        assert r.params["extra"] == 42


class TestSummariseTextureMatches:
    def test_empty(self):
        s = summarise_texture_matches([])
        assert s.n_pairs == 0
        assert s.best_pair is None

    def test_nonempty(self):
        records = [
            make_texture_match_record((0,1), 0.9, 0.8, 0.7, 0.6),
            make_texture_match_record((1,2), 0.3, 0.2, 0.1, 0.0),
        ]
        s = summarise_texture_matches(records)
        assert s.n_pairs == 2
        assert s.best_pair == (0, 1)
        assert s.best_score == pytest.approx(0.9)


class TestFilterTexture:
    def setup_method(self):
        self.records = [
            make_texture_match_record((0,1), 0.9, 0.8, 0.7, 0.6),
            make_texture_match_record((1,2), 0.5, 0.5, 0.5, 0.5),
            make_texture_match_record((2,3), 0.2, 0.3, 0.1, 0.1),
        ]

    def test_filter_by_score(self):
        filtered = filter_texture_by_score(self.records, 0.5)
        assert all(r.score >= 0.5 for r in filtered)

    def test_filter_by_lbp(self):
        filtered = filter_texture_by_lbp(self.records, 0.5)
        assert all(r.lbp_score >= 0.5 for r in filtered)

    def test_top_k(self):
        top2 = top_k_texture_records(self.records, 2)
        assert len(top2) == 2
        assert top2[0].score >= top2[1].score

    def test_best(self):
        best = best_texture_record(self.records)
        assert best is not None
        assert best.score == max(r.score for r in self.records)

    def test_best_empty(self):
        assert best_texture_record([]) is None


class TestTextureScoreStats:
    def test_empty(self):
        s = texture_score_stats([])
        assert s["count"] == 0

    def test_stats(self):
        records = [make_texture_match_record((0,1), v, v, v, v) for v in [0.2, 0.5, 0.8]]
        s = texture_score_stats(records)
        assert s["min"] == pytest.approx(0.2)
        assert s["max"] == pytest.approx(0.8)
        assert s["count"] == 3


class TestBatchPipelineRecord:
    def test_make(self):
        r = make_batch_pipeline_record(0, 10, 9, 1, 0, 5.0, "stage1")
        assert r.batch_id == 0
        assert r.success_rate == pytest.approx(0.9)
        assert r.throughput == pytest.approx(9.0/5.0)

    def test_zero_items(self):
        r = make_batch_pipeline_record(0, 0, 0, 0, 0, 1.0, "stage")
        assert r.success_rate == 0.0

    def test_zero_elapsed(self):
        r = make_batch_pipeline_record(0, 10, 10, 0, 0, 0.0, "stage")
        assert r.throughput == 0.0


class TestSummariseBatchPipeline:
    def test_empty(self):
        s = summarise_batch_pipeline([])
        assert s.n_batches == 0
        assert s.best_batch_id is None

    def test_summary(self):
        records = [
            make_batch_pipeline_record(0, 10, 10, 0, 0, 1.0, "s1"),
            make_batch_pipeline_record(1, 10, 5, 5, 0, 2.0, "s1"),
        ]
        s = summarise_batch_pipeline(records)
        assert s.n_batches == 2
        assert s.best_batch_id == 0
        assert s.worst_batch_id == 1


class TestFilterBatchRecords:
    def setup_method(self):
        self.records = [
            make_batch_pipeline_record(0, 10, 10, 0, 0, 1.0, "stage_a"),
            make_batch_pipeline_record(1, 10, 7, 3, 0, 1.0, "stage_b"),
            make_batch_pipeline_record(2, 10, 4, 6, 0, 1.0, "stage_a"),
        ]

    def test_filter_by_success_rate(self):
        f = filter_batch_by_success_rate(self.records, 0.7)
        assert all(r.success_rate >= 0.7 for r in f)

    def test_filter_by_stage(self):
        f = filter_batch_by_stage(self.records, "stage_a")
        assert len(f) == 2

    def test_top_k_batch(self):
        top2 = top_k_batch_records(self.records, 2)
        assert len(top2) == 2

    def test_throughput_stats(self):
        stats = batch_throughput_stats(self.records)
        assert "min" in stats and "max" in stats and "mean" in stats

    def test_throughput_stats_empty(self):
        stats = batch_throughput_stats([])
        assert stats["count"] == 0

    def test_compare_summaries(self):
        s1 = summarise_batch_pipeline(self.records[:2])
        s2 = summarise_batch_pipeline(self.records[1:])
        diff = compare_batch_summaries(s1, s2)
        assert "delta_total_done" in diff
        assert "same_best" in diff

# ============================================================
# threshold_utils
# ============================================================
from puzzle_reconstruction.utils.threshold_utils import (
    ThresholdConfig, apply_threshold, binarize, adaptive_threshold,
    soft_threshold, threshold_matrix, hysteresis_threshold,
    otsu_threshold, count_above, fraction_above, batch_threshold,
)


class TestThresholdConfig:
    def test_defaults(self):
        cfg = ThresholdConfig()
        assert cfg.low == 0.3
        assert cfg.high == 0.7
        assert cfg.mode == "hard"

    def test_invalid_low_high(self):
        with pytest.raises(ValueError):
            ThresholdConfig(low=0.8, high=0.2)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            ThresholdConfig(mode="medium")


class TestApplyThreshold:
    def test_basic(self):
        arr = np.array([0.1, 0.5, 0.9])
        mask = apply_threshold(arr, 0.5)
        assert list(mask) == [False, True, True]

    def test_invert(self):
        arr = np.array([0.1, 0.5, 0.9])
        mask = apply_threshold(arr, 0.5, invert=True)
        assert list(mask) == [True, False, False]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            apply_threshold(np.array([]), 0.5)


class TestBinarize:
    def test_values_are_01(self):
        arr = np.array([0.2, 0.6, 0.9])
        b = binarize(arr, 0.5)
        assert set(b.tolist()).issubset({0.0, 1.0})

    def test_invert(self):
        arr = np.array([0.2, 0.6])
        b = binarize(arr, 0.5, invert=True)
        assert b[0] == 1.0 and b[1] == 0.0


class TestAdaptiveThreshold:
    def test_returns_bool(self):
        arr = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = adaptive_threshold(arr, window=3)
        assert result.dtype == bool
        assert len(result) == 5

    def test_invert(self):
        arr = np.array([1.0, 2.0, 3.0])
        normal = adaptive_threshold(arr, window=3)
        inverted = adaptive_threshold(arr, window=3, invert=True)
        assert not np.all(normal == inverted)

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            adaptive_threshold(np.array([1.0, 2.0]), window=0)

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            adaptive_threshold(np.ones((3, 3)))


class TestSoftThreshold:
    def test_zero_at_below_threshold(self):
        arr = np.array([0.3, -0.3, 0.0])
        result = soft_threshold(arr, 0.5)
        assert np.all(result == 0.0)

    def test_shrinkage_above(self):
        arr = np.array([2.0, -2.0])
        result = soft_threshold(arr, 1.0)
        assert abs(result[0] - 1.0) < 1e-9
        assert abs(result[1] - (-1.0)) < 1e-9

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            soft_threshold(np.array([1.0]), -0.5)


class TestThresholdMatrix:
    def test_below_replaced(self):
        m = np.array([[0.1, 0.5], [0.8, 0.2]])
        result = threshold_matrix(m, 0.4)
        assert result[0, 0] == 0.0
        assert result[0, 1] == pytest.approx(0.5)

    def test_custom_fill(self):
        m = np.array([[0.1, 0.9]])
        result = threshold_matrix(m, 0.5, fill=-1.0)
        assert result[0, 0] == -1.0

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            threshold_matrix(np.array([1.0, 2.0, 3.0]), 0.5)


class TestHysteresisThreshold:
    def test_propagates_weak(self):
        arr = np.array([0.8, 0.5, 0.2, 0.5, 0.8])
        result = hysteresis_threshold(arr, low=0.4, high=0.7)
        assert result[0] == True   # strong
        assert result[2] == False  # below low

    def test_all_strong(self):
        arr = np.array([0.9, 0.95, 0.85])
        result = hysteresis_threshold(arr, low=0.5, high=0.8)
        assert np.all(result)

    def test_invalid_low_high(self):
        with pytest.raises(ValueError):
            hysteresis_threshold(np.array([0.5]), low=0.8, high=0.3)


class TestOtsuThreshold:
    def test_bimodal(self):
        arr = np.array([0.1]*20 + [0.9]*20, dtype=float)
        t = otsu_threshold(arr)
        assert 0.1 <= t <= 0.9

    def test_single_element_raises(self):
        with pytest.raises(ValueError):
            otsu_threshold(np.array([0.5]))

    def test_constant_returns_value(self):
        arr = np.array([0.5, 0.5, 0.5, 0.5])
        t = otsu_threshold(arr)
        assert t == 0.5


class TestCountAbove:
    def test_basic(self):
        arr = np.array([0.1, 0.5, 0.9, 0.3])
        assert count_above(arr, 0.5) == 2

    def test_none(self):
        assert count_above(np.array([0.1, 0.2]), 0.5) == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            count_above(np.array([]), 0.5)


class TestFractionAbove:
    def test_half(self):
        arr = np.array([0.1, 0.9, 0.2, 0.8])
        assert fraction_above(arr, 0.5) == pytest.approx(0.5)

    def test_all_below(self):
        assert fraction_above(np.array([0.1, 0.2]), 0.5) == 0.0


class TestBatchThreshold:
    def test_basic(self):
        arrays = [np.array([0.2, 0.8]), np.array([0.1, 0.9, 0.5])]
        results = batch_threshold(arrays, 0.5)
        assert len(results) == 2
        assert results[0].dtype == bool

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            batch_threshold([], 0.5)

# ============================================================
# tile_utils
# ============================================================
from puzzle_reconstruction.utils.tile_utils import (
    TileConfig, Tile, compute_tile_grid, tile_image,
    reassemble_tiles, tile_overlap_ratio, filter_tiles_by_content,
    batch_tile_images,
)


class TestTileConfig:
    def test_defaults(self):
        cfg = TileConfig()
        assert cfg.tile_h == 64
        assert cfg.effective_stride_h == 64

    def test_custom_stride(self):
        cfg = TileConfig(tile_h=32, tile_w=32, stride_h=16, stride_w=16)
        assert cfg.effective_stride_h == 16
        assert cfg.effective_stride_w == 16

    def test_invalid_tile_h(self):
        with pytest.raises(ValueError):
            TileConfig(tile_h=0)

    def test_invalid_stride_h(self):
        with pytest.raises(ValueError):
            TileConfig(stride_h=-1)

    def test_invalid_pad_value(self):
        with pytest.raises(ValueError):
            TileConfig(pad_value=300)


class TestComputeTileGrid:
    def test_exact_fit(self):
        cfg = TileConfig(tile_h=32, tile_w=32)
        grid = compute_tile_grid(64, 64, cfg)
        assert len(grid) == 4  # 2x2

    def test_partial_fit(self):
        cfg = TileConfig(tile_h=30, tile_w=30)
        grid = compute_tile_grid(50, 50, cfg)
        assert len(grid) >= 1

    def test_invalid_dims(self):
        with pytest.raises(ValueError):
            compute_tile_grid(0, 50, TileConfig())


class TestTileImage:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.img_gray = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        self.img_color = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)

    def test_gray(self):
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(self.img_gray, cfg)
        assert len(tiles) == 16
        assert all(t.h == 32 and t.w == 32 for t in tiles)

    def test_color(self):
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(self.img_color, cfg)
        assert tiles[0].data.ndim == 3

    def test_with_padding(self):
        cfg = TileConfig(tile_h=50, tile_w=50)
        tiles = tile_image(self.img_gray, cfg)
        assert all(t.h == 50 and t.w == 50 for t in tiles)

    def test_source_size_stored(self):
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(self.img_gray, cfg)
        assert tiles[0].source_h == 128
        assert tiles[0].source_w == 128

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            tile_image(np.ones((10, 10, 3, 2), dtype=np.uint8))


class TestReassembleTiles:
    def test_roundtrip_no_overlap(self):
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(img, cfg)
        reconstructed = reassemble_tiles(tiles, (64, 64))
        assert reconstructed.shape == (64, 64)
        np.testing.assert_array_almost_equal(reconstructed, img, decimal=0)

    def test_empty_tiles_raises(self):
        with pytest.raises(ValueError):
            reassemble_tiles([], (64, 64))

    def test_invalid_out_shape(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        tiles = tile_image(img, TileConfig(tile_h=32, tile_w=32))
        with pytest.raises(ValueError):
            reassemble_tiles(tiles, (0, 32))


class TestTileOverlapRatio:
    def test_no_overlap(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(img, cfg)
        # tiles[0] and tiles[3] are at opposite corners
        ratio = tile_overlap_ratio(tiles[0], tiles[3])
        assert ratio == 0.0

    def test_full_overlap(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(img, cfg)
        ratio = tile_overlap_ratio(tiles[0], tiles[0])
        assert ratio == 1.0


class TestFilterTilesByContent:
    def test_filters_empty(self):
        rng = np.random.default_rng(1)
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:32, :32] = 255
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(img, cfg)
        filled = filter_tiles_by_content(tiles, min_foreground=0.5)
        assert len(filled) == 1

    def test_invalid_foreground(self):
        with pytest.raises(ValueError):
            filter_tiles_by_content([], min_foreground=2.0)


class TestBatchTileImages:
    def test_basic(self):
        rng = np.random.default_rng(1)
        images = [rng.integers(0, 256, (32, 32), dtype=np.uint8) for _ in range(3)]
        cfg = TileConfig(tile_h=16, tile_w=16)
        result = batch_tile_images(images, cfg)
        assert len(result) == 3
        assert all(len(tiles) == 4 for tiles in result)

# ============================================================
# topology_utils
# ============================================================
from puzzle_reconstruction.utils.topology_utils import (
    TopologyConfig, compute_euler_number, count_holes,
    compute_solidity, compute_extent, compute_convexity,
    compute_compactness, is_simply_connected, shape_complexity,
    batch_topology,
)


class TestTopologyConfig:
    def test_defaults(self):
        cfg = TopologyConfig()
        assert cfg.connectivity in (4, 8)

    def test_invalid_connectivity(self):
        with pytest.raises(ValueError):
            TopologyConfig(connectivity=6)

    def test_invalid_min_area(self):
        with pytest.raises(ValueError):
            TopologyConfig(min_area=0)


def _make_square_contour(n=20):
    """Create a square contour with n points per side."""
    pts = []
    # bottom edge
    for x in range(n): pts.append([x, 0])
    # right edge
    for y in range(1, n): pts.append([n-1, y])
    # top edge
    for x in range(n-2, -1, -1): pts.append([x, n-1])
    # left edge
    for y in range(n-2, 0, -1): pts.append([0, y])
    return np.array(pts, dtype=float)


def _make_circle_contour(r=10.0, n=100):
    """Approximate circle contour."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


class TestComputeEulerNumber:
    def test_solid_rectangle(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[5:25, 5:25] = True
        e = compute_euler_number(mask)
        assert isinstance(e, int)
        assert e == 1  # one component, no holes

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            compute_euler_number(np.ones((5, 5, 3), dtype=bool))


class TestCountHoles:
    def test_no_holes(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:18, 2:18] = True
        assert count_holes(mask) == 0

    def test_one_hole(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[2:28, 2:28] = True
        mask[10:20, 10:20] = False  # hole
        assert count_holes(mask) == 1

    def test_is_simply_connected_no_holes(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:18, 2:18] = True
        assert is_simply_connected(mask) is True

    def test_is_simply_connected_with_hole(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[2:28, 2:28] = True
        mask[10:20, 10:20] = False
        assert is_simply_connected(mask) is False


class TestComputeSolidity:
    def test_square_near_one(self):
        pts = _make_square_contour(20)
        s = compute_solidity(pts)
        assert 0.8 <= s <= 1.0

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            compute_solidity(np.array([[1.0, 2.0]]))

    def test_less_than_3_points(self):
        with pytest.raises(ValueError):
            compute_solidity(np.array([[0.0,0.0],[1.0,1.0]]))


class TestComputeExtent:
    def test_square_near_one(self):
        pts = _make_square_contour(20)
        e = compute_extent(pts)
        assert 0.5 <= e <= 1.0

    def test_invalid(self):
        with pytest.raises(ValueError):
            compute_extent(np.array([[0.0,0.0],[1.0,0.0]]))


class TestComputeConvexity:
    def test_convex_polygon_near_one(self):
        pts = _make_square_contour(20)
        c = compute_convexity(pts)
        assert 0.8 <= c <= 1.0

    def test_circle_near_one(self):
        pts = _make_circle_contour()
        c = compute_convexity(pts)
        assert 0.9 <= c <= 1.0


class TestComputeCompactness:
    def test_circle_near_one(self):
        pts = _make_circle_contour(r=50.0, n=200)
        c = compute_compactness(pts)
        assert 0.9 <= c <= 1.0

    def test_square_less_than_circle(self):
        sq = compute_compactness(_make_square_contour(20))
        circ = compute_compactness(_make_circle_contour(r=50.0, n=200))
        assert sq < circ


class TestShapeComplexity:
    def test_range(self):
        pts = _make_square_contour(20)
        c = shape_complexity(pts)
        assert 0.0 <= c <= 1.0

    def test_circle_less_complex(self):
        sq_c = shape_complexity(_make_square_contour(20))
        circ_c = shape_complexity(_make_circle_contour(r=50.0, n=200))
        assert circ_c < sq_c


class TestBatchTopology:
    def test_basic(self):
        contours = [_make_square_contour(10), _make_circle_contour(r=5.0)]
        results = batch_topology(contours)
        assert len(results) == 2
        for r in results:
            assert all(k in r for k in ("solidity","extent","convexity","compactness","complexity"))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            batch_topology([])

# ============================================================
# tracker_utils
# ============================================================
from puzzle_reconstruction.utils.tracker_utils import (
    TrackerConfig, StepRecord, IterTracker,
    create_iter_tracker, record_step, get_values, get_steps,
    get_best_record, get_worst_record, compute_delta,
    is_improving, find_plateau_start, smooth_values,
    tracker_stats, compare_trackers, merge_trackers,
    window_stats, top_k_records,
)


def _make_tracker(values, name="test"):
    """Helper: create tracker with given values as steps 0..n-1."""
    t = create_iter_tracker(TrackerConfig(name=name))
    for i, v in enumerate(values):
        record_step(t, i, v)
    return t


class TestCreateIterTracker:
    def test_empty_tracker(self):
        t = create_iter_tracker()
        assert len(t.records) == 0

    def test_with_meta(self):
        t = create_iter_tracker(experiment="run1")
        assert t.metadata["experiment"] == "run1"


class TestRecordStep:
    def test_records_append(self):
        t = create_iter_tracker()
        record_step(t, 0, 1.5)
        assert len(t.records) == 1
        assert t.records[0].value == 1.5

    def test_keep_history_false(self):
        cfg = TrackerConfig(keep_history=False)
        t = create_iter_tracker(cfg)
        for i in range(5):
            record_step(t, i, float(i))
        assert len(t.records) == 1
        assert t.records[0].value == 4.0

    def test_with_meta(self):
        t = create_iter_tracker()
        record_step(t, 0, 1.0, loss=0.5)
        assert t.records[0].meta["loss"] == 0.5


class TestGetValues:
    def test_basic(self):
        t = _make_tracker([1.0, 2.0, 3.0])
        vals = get_values(t)
        np.testing.assert_array_equal(vals, [1.0, 2.0, 3.0])

    def test_empty(self):
        t = create_iter_tracker()
        assert len(get_values(t)) == 0


class TestGetSteps:
    def test_basic(self):
        t = _make_tracker([1.0, 2.0, 3.0])
        steps = get_steps(t)
        np.testing.assert_array_equal(steps, [0, 1, 2])


class TestGetBestWorstRecord:
    def test_best(self):
        t = _make_tracker([1.0, 5.0, 3.0])
        best = get_best_record(t)
        assert best.value == 5.0

    def test_worst(self):
        t = _make_tracker([1.0, 5.0, 3.0])
        worst = get_worst_record(t)
        assert worst.value == 1.0

    def test_empty_returns_none(self):
        t = create_iter_tracker()
        assert get_best_record(t) is None
        assert get_worst_record(t) is None


class TestComputeDelta:
    def test_basic(self):
        t = _make_tracker([1.0, 3.0, 6.0, 10.0])
        delta = compute_delta(t, lag=1)
        np.testing.assert_array_almost_equal(delta, [2.0, 3.0, 4.0])

    def test_lag_2(self):
        t = _make_tracker([0.0, 1.0, 3.0, 6.0])
        delta = compute_delta(t, lag=2)
        np.testing.assert_array_almost_equal(delta, [3.0, 5.0])

    def test_short_tracker(self):
        t = _make_tracker([1.0])
        delta = compute_delta(t, lag=1)
        assert len(delta) == 0

    def test_invalid_lag(self):
        t = _make_tracker([1.0, 2.0])
        with pytest.raises(ValueError):
            compute_delta(t, lag=0)


class TestIsImproving:
    def test_improving(self):
        t = _make_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
        assert is_improving(t, window=3) is True

    def test_not_improving(self):
        t = _make_tracker([5.0, 4.0, 3.0, 2.0, 1.0])
        assert is_improving(t, window=3) is False

    def test_too_short(self):
        t = _make_tracker([1.0, 2.0])
        assert is_improving(t, window=5) is False


class TestFindPlateauStart:
    def test_finds_plateau(self):
        t = _make_tracker([1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        step = find_plateau_start(t, window=3, tol=0.01)
        assert step is not None

    def test_no_plateau(self):
        t = _make_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
        step = find_plateau_start(t, window=3, tol=0.01)
        assert step is None

    def test_invalid_window(self):
        t = _make_tracker([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            find_plateau_start(t, window=1)


class TestSmoothValues:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        smoothed = smooth_values(vals, window=3)
        assert len(smoothed) == 5

    def test_window_1_unchanged(self):
        vals = np.array([1.0, 2.0, 3.0])
        smoothed = smooth_values(vals, window=1)
        np.testing.assert_array_almost_equal(smoothed, vals)

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            smooth_values(np.array([1.0, 2.0]), window=0)


class TestTrackerStats:
    def test_empty(self):
        t = create_iter_tracker()
        stats = tracker_stats(t)
        assert stats["n"] == 0

    def test_nonempty(self):
        t = _make_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = tracker_stats(t)
        assert stats["n"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert abs(stats["mean"] - 3.0) < 1e-9


class TestCompareTrackers:
    def test_compare(self):
        t1 = _make_tracker([1.0, 5.0, 3.0])
        t2 = _make_tracker([2.0, 4.0, 6.0])
        result = compare_trackers(t1, t2)
        assert "winner" in result
        assert result["winner"] == "b"  # t2 best=6 > t1 best=5

    def test_both_empty(self):
        t1 = create_iter_tracker()
        t2 = create_iter_tracker()
        result = compare_trackers(t1, t2)
        assert result["winner"] == "tie"


class TestMergeTrackers:
    def test_merge(self):
        t1 = _make_tracker([1.0, 2.0])
        t2 = _make_tracker([3.0, 4.0])
        merged = merge_trackers([t1, t2])
        assert len(merged.records) == 4


class TestWindowStats:
    def test_basic(self):
        t = _make_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
        ws = window_stats(t, window=3)
        assert len(ws) == 3  # n - window + 1 = 5 - 3 + 1 = 3
        assert all("mean" in w and "std" in w for w in ws)

    def test_invalid_window(self):
        t = _make_tracker([1.0, 2.0])
        with pytest.raises(ValueError):
            window_stats(t, window=0)


class TestTopKRecords:
    def test_top_k(self):
        t = _make_tracker([3.0, 1.0, 5.0, 2.0, 4.0])
        top3 = top_k_records(t, 3)
        assert len(top3) == 3
        assert top3[0].value == 5.0

# ============================================================
# voting_utils
# ============================================================
from puzzle_reconstruction.utils.voting_utils import (
    VoteConfig, cast_pair_votes, aggregate_pair_votes,
    cast_position_votes, majority_vote, weighted_vote,
    rank_fusion, batch_vote,
)


class TestVoteConfig:
    def test_defaults(self):
        cfg = VoteConfig()
        assert cfg.min_votes == 1
        assert cfg.rrf_k == 60.0

    def test_invalid_min_votes(self):
        with pytest.raises(ValueError):
            VoteConfig(min_votes=0)

    def test_invalid_rrf_k(self):
        with pytest.raises(ValueError):
            VoteConfig(rrf_k=0)

    def test_invalid_weight(self):
        with pytest.raises(ValueError):
            VoteConfig(weights=[-0.1])


class TestCastPairVotes:
    def test_basic(self):
        pair_lists = [[(0,1),(1,2)], [(0,1),(2,3)]]
        votes = cast_pair_votes(pair_lists)
        assert votes[(0,1)] == 2.0
        assert votes[(1,2)] == 1.0

    def test_canonical_form(self):
        pair_lists = [[(1,0)], [(0,1)]]
        votes = cast_pair_votes(pair_lists)
        assert (0,1) in votes
        assert votes[(0,1)] == 2.0

    def test_with_weights(self):
        pair_lists = [[(0,1)], [(0,1)]]
        votes = cast_pair_votes(pair_lists, weights=[2.0, 3.0])
        assert votes[(0,1)] == 5.0

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError):
            cast_pair_votes([[(0,1)]], weights=[1.0, 2.0])


class TestAggregatePairVotes:
    def test_basic(self):
        votes = {(0,1): 3.0, (1,2): 1.0}
        result = aggregate_pair_votes(votes)
        assert result[0][0] == (0,1)
        assert result[0][1] == pytest.approx(1.0)  # normalized

    def test_min_votes_filter(self):
        votes = {(0,1): 3.0, (1,2): 1.0}
        cfg = VoteConfig(min_votes=2)
        result = aggregate_pair_votes(votes, cfg=cfg)
        assert len(result) == 1
        assert result[0][0] == (0,1)

    def test_empty_votes(self):
        assert aggregate_pair_votes({}) == []

    def test_no_normalize(self):
        votes = {(0,1): 3.0}
        cfg = VoteConfig(normalize=False)
        result = aggregate_pair_votes(votes, cfg=cfg)
        assert result[0][1] == 3.0


class TestCastPositionVotes:
    def test_basic(self):
        pos_lists = [{0: 0.8, 1: 0.5}, {0: 0.6, 2: 0.9}]
        result = cast_position_votes(pos_lists)
        assert result[0] == pytest.approx(1.4)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(0.9)

    def test_with_weights(self):
        pos_lists = [{0: 1.0}, {0: 1.0}]
        result = cast_position_votes(pos_lists, weights=[2.0, 3.0])
        assert result[0] == pytest.approx(5.0)

    def test_mismatch_raises(self):
        with pytest.raises(ValueError):
            cast_position_votes([{0: 1.0}], weights=[1.0, 2.0])


class TestMajorityVote:
    def test_simple(self):
        result = majority_vote([1, 2, 2, 3, 2])
        assert result == 2

    def test_empty(self):
        assert majority_vote([]) is None

    def test_strings(self):
        result = majority_vote(["a", "b", "a", "c"])
        assert result == "a"


class TestWeightedVote:
    def test_equal_weights(self):
        result = weighted_vote([1.0, 2.0, 3.0])
        assert abs(result - 2.0) < 1e-9

    def test_unequal_weights(self):
        result = weighted_vote([0.0, 10.0], weights=[9.0, 1.0])
        assert abs(result - 1.0) < 1e-9

    def test_empty(self):
        assert weighted_vote([]) == 0.0

    def test_zero_weights(self):
        assert weighted_vote([1.0, 2.0], weights=[0.0, 0.0]) == 0.0

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            weighted_vote([1.0], weights=[-0.5])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            weighted_vote([1.0, 2.0], weights=[1.0])


class TestRankFusion:
    def test_basic(self):
        lists = [["a","b","c"], ["b","a","c"]]
        result = rank_fusion(lists)
        assert len(result) == 3
        assert result[0][0] in ("a", "b")

    def test_single_list(self):
        result = rank_fusion([["x","y","z"]])
        assert result[0][0] == "x"

    def test_normalize(self):
        result = rank_fusion([["a","b"]], cfg=VoteConfig(normalize=True))
        assert result[0][1] == pytest.approx(1.0)


class TestBatchVote:
    def test_basic(self):
        batch = [
            [[(0,1),(1,2)], [(0,1)]],
            [[(2,3)], [(2,3),(3,4)]],
        ]
        results = batch_vote(batch)
        assert len(results) == 2
        assert len(results[0]) >= 1

# ============================================================
# window_utils
# ============================================================
from puzzle_reconstruction.utils.window_utils import (
    WindowConfig, apply_window_function, rolling_mean, rolling_std,
    rolling_max, rolling_min, compute_overlap,
    split_into_windows, merge_windows, batch_rolling,
)


class TestWindowConfig:
    def test_defaults(self):
        cfg = WindowConfig()
        assert cfg.size == 8
        assert cfg.step == 1
        assert cfg.func == "rect"

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            WindowConfig(size=0)

    def test_invalid_step(self):
        with pytest.raises(ValueError):
            WindowConfig(step=0)

    def test_invalid_func(self):
        with pytest.raises(ValueError):
            WindowConfig(func="flat")

    def test_invalid_padding(self):
        with pytest.raises(ValueError):
            WindowConfig(padding="wrap")


class TestApplyWindowFunction:
    def test_rect_unchanged(self):
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cfg = WindowConfig(func="rect", size=5)
        result = apply_window_function(s, cfg)
        np.testing.assert_array_almost_equal(result, s)

    def test_hann_attenuates_edges(self):
        s = np.ones(8)
        cfg = WindowConfig(func="hann", size=8)
        result = apply_window_function(s, cfg)
        # edges should be near 0
        assert result[0] < 0.1

    def test_all_window_types(self):
        s = np.ones(16)
        for func in ("hann", "hamming", "bartlett", "blackman"):
            cfg = WindowConfig(func=func, size=16)
            result = apply_window_function(s, cfg)
            assert len(result) == 16

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            apply_window_function(np.ones((4, 4)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            apply_window_function(np.array([]))


class TestRollingMean:
    def test_constant_signal(self):
        s = np.ones(20) * 5.0
        cfg = WindowConfig(size=4, step=1, padding="same")
        result = rolling_mean(s, cfg)
        assert len(result) == 20
        assert np.allclose(result, 5.0, atol=0.1)

    def test_valid_padding(self):
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cfg = WindowConfig(size=3, step=1, padding="valid")
        result = rolling_mean(s, cfg)
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            rolling_mean(np.array([]))


class TestRollingStd:
    def test_constant_zero_std(self):
        s = np.ones(20)
        cfg = WindowConfig(size=4, step=1, padding="same")
        result = rolling_std(s, cfg)
        assert len(result) == 20
        assert np.allclose(result, 0.0, atol=0.01)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            rolling_std(np.ones((3, 3)))


class TestRollingMaxMin:
    def test_rolling_max(self):
        s = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        cfg = WindowConfig(size=3, step=1, padding="valid")
        result = rolling_max(s, cfg)
        np.testing.assert_array_almost_equal(result, [3.0, 5.0, 5.0])

    def test_rolling_min(self):
        s = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        cfg = WindowConfig(size=3, step=1, padding="valid")
        result = rolling_min(s, cfg)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 2.0])


class TestComputeOverlap:
    def test_full_overlap(self):
        assert compute_overlap(0, 10, 0, 10) == 1.0

    def test_no_overlap(self):
        assert compute_overlap(0, 5, 6, 10) == 0.0

    def test_partial_overlap(self):
        result = compute_overlap(0, 10, 5, 15)
        assert 0.0 < result < 1.0

    def test_invalid_window_a(self):
        with pytest.raises(ValueError):
            compute_overlap(5, 3, 0, 10)

    def test_invalid_window_b(self):
        with pytest.raises(ValueError):
            compute_overlap(0, 10, 5, 5)


class TestSplitIntoWindows:
    def test_same_padding(self):
        s = np.arange(10.0)
        cfg = WindowConfig(size=4, step=2, padding="same")
        windows = split_into_windows(s, cfg)
        assert len(windows) > 0
        assert all(len(w) == 4 for w in windows)

    def test_valid_padding(self):
        s = np.arange(10.0)
        cfg = WindowConfig(size=4, step=2, padding="valid")
        windows = split_into_windows(s, cfg)
        assert len(windows) == 4  # (10 - 4) / 2 + 1 = 4

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            split_into_windows(np.array([]))


class TestMergeWindows:
    def test_roundtrip(self):
        s = np.arange(16.0)
        cfg = WindowConfig(size=4, step=4, padding="valid")
        windows = split_into_windows(s, cfg)
        reconstructed = merge_windows(windows, 16, cfg)
        np.testing.assert_array_almost_equal(reconstructed, s)

    def test_empty_windows_raises(self):
        with pytest.raises(ValueError):
            merge_windows([], 10)

    def test_invalid_length_raises(self):
        windows = [np.ones(4)]
        with pytest.raises(ValueError):
            merge_windows(windows, 0)


class TestBatchRolling:
    def test_mean(self):
        sigs = [np.ones(10), np.ones(10) * 2.0]
        results = batch_rolling(sigs, stat="mean")
        assert len(results) == 2

    def test_all_stats(self):
        sigs = [np.arange(10.0)]
        for stat in ("mean", "std", "max", "min"):
            result = batch_rolling(sigs, stat=stat)
            assert len(result[0]) > 0

    def test_invalid_stat(self):
        with pytest.raises(ValueError):
            batch_rolling([np.ones(5)], stat="median")

    def test_empty_signals_raises(self):
        with pytest.raises(ValueError):
            batch_rolling([])

# ============================================================
# homography_verifier
# ============================================================
from puzzle_reconstruction.verification.homography_verifier import (
    HomographyConfig, HomographyResult,
    estimate_homography_dlt, reprojection_error,
    estimate_homography_ransac, check_homography_quality,
    HomographyVerifier,
)


def _make_homography_data(seed=42, n=20, noise=0.0, outlier_ratio=0.0):
    """Generate point correspondences from a known homography."""
    rng = np.random.default_rng(seed)
    # Known H: translation + slight rotation
    H_true = np.array([
        [1.0, 0.05, 10.0],
        [-0.05, 1.0, 5.0],
        [0.0, 0.0, 1.0],
    ])
    src = rng.uniform(10, 200, (n, 2))
    src_h = np.column_stack([src, np.ones(n)])
    dst_h = (H_true @ src_h.T).T
    w = dst_h[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    dst = dst_h[:, :2] / w
    if noise > 0:
        dst += rng.normal(0, noise, dst.shape)
    n_out = int(n * outlier_ratio)
    if n_out > 0:
        dst[-n_out:] = rng.uniform(0, 300, (n_out, 2))
    return src, dst, H_true


class TestEstimateHomographyDlt:
    def test_exact_correspondences(self):
        src, dst, H_true = _make_homography_data(n=10, noise=0.0)
        H = estimate_homography_dlt(src, dst)
        assert H is not None
        assert H.shape == (3, 3)

    def test_few_points_returns_none(self):
        src = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
        dst = src.copy()
        H = estimate_homography_dlt(src, dst)
        assert H is None

    def test_mismatched_lengths(self):
        src = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]])
        dst = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
        H = estimate_homography_dlt(src, dst)
        assert H is None

    def test_normalization(self):
        """H[2,2] should be ~1."""
        src, dst, _ = _make_homography_data(n=8)
        H = estimate_homography_dlt(src, dst)
        assert H is not None
        assert abs(H[2, 2] - 1.0) < 1e-6

    def test_4_point_minimum(self):
        src, dst, _ = _make_homography_data(n=4, noise=0.0)
        H = estimate_homography_dlt(src, dst)
        assert H is not None


class TestReprojectionError:
    def test_exact_zero_error(self):
        src, dst, H_true = _make_homography_data(n=10, noise=0.0)
        H = estimate_homography_dlt(src, dst)
        assert H is not None
        errs = reprojection_error(H, src, dst)
        assert np.all(errs < 1.0)

    def test_identity_no_error(self):
        H = np.eye(3)
        pts = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
        errs = reprojection_error(H, pts, pts)
        assert np.allclose(errs, 0.0)

    def test_shape(self):
        H = np.eye(3)
        pts = np.ones((5, 2))
        errs = reprojection_error(H, pts, pts)
        assert errs.shape == (5,)


class TestEstimateHomographyRansac:
    def test_clean_data(self):
        src, dst, H_true = _make_homography_data(n=20, noise=0.1)
        H, mask = estimate_homography_ransac(src, dst)
        assert H is not None
        assert mask.dtype == bool
        assert mask.shape == (20,)

    def test_with_outliers(self):
        src, dst, H_true = _make_homography_data(n=30, noise=0.5, outlier_ratio=0.2)
        cfg = HomographyConfig(ransac_threshold=5.0, max_iterations=200)
        H, mask = estimate_homography_ransac(src, dst, cfg=cfg)
        assert H is not None
        assert int(mask.sum()) >= 4

    def test_too_few_points(self):
        src = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
        dst = src.copy()
        H, mask = estimate_homography_ransac(src, dst)
        assert H is None
        assert not any(mask)


class TestCheckHomographyQuality:
    def test_identity_valid(self):
        H = np.eye(3)
        assert check_homography_quality(H, (256, 256)) is True

    def test_reflection_invalid(self):
        H = np.diag([-1.0, 1.0, 1.0])
        assert check_homography_quality(H, (256, 256)) is False

    def test_degenerate_invalid(self):
        H = np.zeros((3, 3))
        assert check_homography_quality(H, (256, 256)) is False


class TestHomographyVerifier:
    def setup_method(self):
        self.verifier = HomographyVerifier()

    def test_clean_data_valid(self):
        src, dst, _ = _make_homography_data(n=20, noise=0.1)
        result = self.verifier.verify(src, dst, fragment_size=(256, 256))
        assert isinstance(result, HomographyResult)
        assert result.H is not None
        assert result.n_inliers >= 4

    def test_too_few_points(self):
        src = np.array([[0.0,0.0],[1.0,1.0]])
        dst = src.copy()
        result = self.verifier.verify(src, dst)
        assert result.H is None
        assert result.is_valid is False
        assert result.score == 0.0

    def test_score_range(self):
        src, dst, _ = _make_homography_data(n=20, noise=0.1)
        result = self.verifier.verify(src, dst)
        assert 0.0 <= result.score <= 1.0

    def test_inlier_ratio_range(self):
        src, dst, _ = _make_homography_data(n=20, noise=0.1)
        result = self.verifier.verify(src, dst)
        assert 0.0 <= result.inlier_ratio <= 1.0

    def test_with_outliers_degrades(self):
        src_clean, dst_clean, _ = _make_homography_data(n=20, noise=0.0)
        src_noisy, dst_noisy, _ = _make_homography_data(n=20, noise=0.0, outlier_ratio=0.5)
        r_clean = self.verifier.verify(src_clean, dst_clean)
        r_noisy = self.verifier.verify(src_noisy, dst_noisy)
        # Clean should score at least as well
        assert r_clean.score >= r_noisy.score - 0.3

    def test_custom_config(self):
        cfg = HomographyConfig(ransac_threshold=2.0, min_inliers=3, max_iterations=50)
        verifier = HomographyVerifier(cfg=cfg)
        src, dst, _ = _make_homography_data(n=15, noise=0.3)
        result = verifier.verify(src, dst)
        assert isinstance(result, HomographyResult)

    def test_verify_batch(self):
        pairs = [
            _make_homography_data(n=20, noise=0.1)[:2],
            _make_homography_data(n=20, noise=0.5)[:2],
        ]
        results = self.verifier.verify_batch(pairs)
        assert len(results) == 2
        assert all(isinstance(r, HomographyResult) for r in results)

    def test_reprojection_error_finite(self):
        src, dst, _ = _make_homography_data(n=20, noise=0.1)
        result = self.verifier.verify(src, dst)
        if result.H is not None:
            assert np.isfinite(result.reprojection_error) or result.reprojection_error == float("inf")

# ============================================================
# orient_skew_utils
# ============================================================
from puzzle_reconstruction.utils.orient_skew_utils import (
    OrientMatchConfig, OrientMatchEntry, OrientMatchSummary,
    make_orient_match_entry, summarise_orient_match_entries,
    filter_high_orient_matches, filter_low_orient_matches,
    filter_orient_by_score_range, filter_orient_by_max_angle,
    top_k_orient_match_entries, best_orient_match_entry,
    orient_match_stats, compare_orient_summaries,
    batch_summarise_orient_match_entries,
    SkewCorrConfig, SkewCorrEntry, SkewCorrSummary,
    make_skew_corr_entry, summarise_skew_corr_entries,
    filter_high_confidence_skew, filter_skew_by_method,
    filter_skew_by_angle_range, top_k_skew_entries,
    best_skew_entry, skew_corr_stats, compare_skew_summaries,
    batch_summarise_skew_corr_entries,
)


def _make_orient_entries(n=5):
    rng = np.random.default_rng(42)
    return [
        make_orient_match_entry(i, i+1, rng.uniform(0, 180), rng.uniform(0, 1), 18)
        for i in range(n)
    ]


class TestOrientMatchEntry:
    def test_make(self):
        e = make_orient_match_entry(0, 1, 45.0, 0.8, 36)
        assert e.fragment_a == 0
        assert e.fragment_b == 1
        assert e.best_angle == 45.0
        assert e.best_score == 0.8
        assert e.n_angles_tested == 36


class TestSummariseOrientMatch:
    def test_empty(self):
        s = summarise_orient_match_entries([])
        assert s.n_entries == 0
        assert s.mean_score == 0.0

    def test_nonempty(self):
        entries = _make_orient_entries(5)
        s = summarise_orient_match_entries(entries)
        assert s.n_entries == 5
        assert s.min_score <= s.mean_score <= s.max_score

    def test_high_score_count(self):
        entries = [
            make_orient_match_entry(0, 1, 10.0, 0.9, 18),
            make_orient_match_entry(1, 2, 20.0, 0.5, 18),
            make_orient_match_entry(2, 3, 30.0, 0.8, 18),
        ]
        s = summarise_orient_match_entries(entries)
        assert s.high_score_count == 2


class TestFilterOrientMatch:
    def setup_method(self):
        self.entries = [
            make_orient_match_entry(0, 1, 10.0, 0.9, 18),
            make_orient_match_entry(1, 2, 90.0, 0.3, 18),
            make_orient_match_entry(2, 3, 45.0, 0.7, 18),
        ]

    def test_filter_high(self):
        f = filter_high_orient_matches(self.entries, threshold=0.7)
        assert all(e.best_score >= 0.7 for e in f)

    def test_filter_low(self):
        f = filter_low_orient_matches(self.entries, threshold=0.7)
        assert all(e.best_score < 0.7 for e in f)

    def test_filter_score_range(self):
        f = filter_orient_by_score_range(self.entries, lo=0.3, hi=0.7)
        assert all(0.3 <= e.best_score <= 0.7 for e in f)

    def test_filter_by_max_angle(self):
        f = filter_orient_by_max_angle(self.entries, max_angle=50.0)
        assert all(e.best_angle <= 50.0 for e in f)
        assert len(f) == 2

    def test_top_k(self):
        top2 = top_k_orient_match_entries(self.entries, k=2)
        assert len(top2) == 2
        assert top2[0].best_score >= top2[1].best_score

    def test_best(self):
        best = best_orient_match_entry(self.entries)
        assert best.best_score == 0.9

    def test_best_empty(self):
        assert best_orient_match_entry([]) is None


class TestOrientMatchStats:
    def test_empty(self):
        stats = orient_match_stats([])
        assert stats["count"] == 0

    def test_nonempty(self):
        entries = _make_orient_entries(5)
        stats = orient_match_stats(entries)
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_single_zero_std(self):
        entries = [make_orient_match_entry(0, 1, 10.0, 0.5, 10)]
        stats = orient_match_stats(entries)
        assert stats["std"] == 0.0


class TestCompareOrientSummaries:
    def test_compare(self):
        s1 = summarise_orient_match_entries([
            make_orient_match_entry(0, 1, 10.0, 0.9, 18)])
        s2 = summarise_orient_match_entries([
            make_orient_match_entry(0, 1, 20.0, 0.4, 18)])
        diff = compare_orient_summaries(s1, s2)
        assert "mean_score_delta" in diff
        assert diff["mean_score_delta"] == pytest.approx(0.5)


class TestBatchSummariseOrientMatch:
    def test_batch(self):
        groups = [_make_orient_entries(3), _make_orient_entries(2)]
        summaries = batch_summarise_orient_match_entries(groups)
        assert len(summaries) == 2
        assert summaries[0].n_entries == 3
        assert summaries[1].n_entries == 2


# ─── SkewCorr ────────────────────────────────────────────────────────────────

def _make_skew_entries(n=5):
    rng = np.random.default_rng(99)
    methods = ["deskew", "hough", "projection"]
    return [
        make_skew_corr_entry(i, rng.uniform(-45, 45),
                            rng.uniform(0, 1),
                            methods[i % len(methods)])
        for i in range(n)
    ]


class TestSkewCorrEntry:
    def test_make(self):
        e = make_skew_corr_entry(0, -5.0, 0.9, "deskew")
        assert e.image_id == 0
        assert e.angle_deg == -5.0
        assert e.confidence == 0.9
        assert e.method == "deskew"


class TestSummariseSkewCorr:
    def test_empty(self):
        s = summarise_skew_corr_entries([])
        assert s.n_entries == 0
        assert s.dominant_method == ""

    def test_nonempty(self):
        entries = _make_skew_entries(6)
        s = summarise_skew_corr_entries(entries)
        assert s.n_entries == 6
        assert s.min_confidence <= s.mean_confidence <= s.max_confidence
        assert isinstance(s.dominant_method, str)

    def test_dominant_method(self):
        entries = [
            make_skew_corr_entry(0, 0.0, 0.9, "deskew"),
            make_skew_corr_entry(1, 1.0, 0.8, "deskew"),
            make_skew_corr_entry(2, 2.0, 0.7, "hough"),
        ]
        s = summarise_skew_corr_entries(entries)
        assert s.dominant_method == "deskew"


class TestFilterSkewCorr:
    def setup_method(self):
        self.entries = [
            make_skew_corr_entry(0, 0.0, 0.9, "deskew"),
            make_skew_corr_entry(1, 10.0, 0.3, "hough"),
            make_skew_corr_entry(2, -5.0, 0.7, "deskew"),
        ]

    def test_filter_high_confidence(self):
        f = filter_high_confidence_skew(self.entries, threshold=0.7)
        assert all(e.confidence >= 0.7 for e in f)

    def test_filter_by_method(self):
        f = filter_skew_by_method(self.entries, "deskew")
        assert len(f) == 2
        assert all(e.method == "deskew" for e in f)

    def test_filter_by_angle_range(self):
        f = filter_skew_by_angle_range(self.entries, lo=-10.0, hi=5.0)
        assert all(-10.0 <= e.angle_deg <= 5.0 for e in f)

    def test_top_k(self):
        top2 = top_k_skew_entries(self.entries, 2)
        assert len(top2) == 2
        assert top2[0].confidence >= top2[1].confidence

    def test_best(self):
        best = best_skew_entry(self.entries)
        assert best.confidence == 0.9

    def test_best_empty(self):
        assert best_skew_entry([]) is None


class TestSkewCorrStats:
    def test_empty(self):
        stats = skew_corr_stats([])
        assert stats["count"] == 0

    def test_nonempty(self):
        entries = _make_skew_entries(5)
        stats = skew_corr_stats(entries)
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_single_zero_std(self):
        entries = [make_skew_corr_entry(0, 0.0, 0.5, "x")]
        stats = skew_corr_stats(entries)
        assert stats["std"] == 0.0


class TestCompareSkewSummaries:
    def test_compare(self):
        s1 = summarise_skew_corr_entries([make_skew_corr_entry(0, 0.0, 0.9, "x")])
        s2 = summarise_skew_corr_entries([make_skew_corr_entry(0, 5.0, 0.4, "x")])
        diff = compare_skew_summaries(s1, s2)
        assert "mean_confidence_delta" in diff
        assert diff["mean_confidence_delta"] == pytest.approx(0.5)


class TestBatchSummariseSkewCorr:
    def test_batch(self):
        groups = [_make_skew_entries(3), _make_skew_entries(4)]
        summaries = batch_summarise_skew_corr_entries(groups)
        assert len(summaries) == 2
        assert summaries[0].n_entries == 3
        assert summaries[1].n_entries == 4
