"""Integration tests for utils batch 6a.

Covers:
    1. puzzle_reconstruction.utils.shape_match_utils
    2. puzzle_reconstruction.utils.spatial_index
    3. puzzle_reconstruction.utils.stats_utils
    4. puzzle_reconstruction.utils.text_utils
    5. puzzle_reconstruction.utils.texture_pipeline_utils
"""
import math
import pytest
import numpy as np

rng = np.random.default_rng(42)

# ─── imports ──────────────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.shape_match_utils import (
    ShapeMatchConfig, ShapeMatchEntry, ShapeMatchSummary,
    make_match_entry, entries_from_results, summarise_matches,
    filter_good_matches, filter_poor_matches, filter_by_hu_dist,
    filter_match_by_score_range, top_k_match_entries,
    match_entry_stats, compare_match_summaries,
)
from puzzle_reconstruction.utils.spatial_index import (
    SpatialConfig, SpatialEntry, SpatialIndex,
    build_spatial_index, query_radius, query_knn, pairwise_distances,
    cluster_by_distance,
)
from puzzle_reconstruction.utils.stats_utils import (
    StatsConfig, describe, zscore_array, iqr, winsorize,
    percentile_rank, outlier_mask, running_stats,
    weighted_mean, weighted_std, batch_describe,
)
from puzzle_reconstruction.utils.text_utils import (
    TextConfig, TextBlock,
    clean_ocr_text, estimate_text_density, find_text_lines,
    segment_words, compute_text_score, compare_text_blocks,
    align_text_blocks, batch_clean_text,
)
from puzzle_reconstruction.utils.texture_pipeline_utils import (
    TextureMatchRecord, TextureMatchSummary,
    make_texture_match_record, summarise_texture_matches,
    filter_texture_by_score, filter_texture_by_lbp,
    top_k_texture_records, best_texture_record, texture_score_stats,
    BatchPipelineRecord, BatchPipelineSummary,
    make_batch_pipeline_record, summarise_batch_pipeline,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. shape_match_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestShapeMatchUtils:

    def _make_entries(self, scores):
        return [make_match_entry(i, i + 1, s) for i, s in enumerate(scores)]

    def test_config_defaults(self):
        cfg = ShapeMatchConfig()
        assert cfg.min_score == 0.0
        assert cfg.max_pairs == 100
        assert cfg.method == "hu"

    def test_config_invalid_min_score(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(min_score=-0.1)

    def test_config_invalid_method(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(method="unknown")

    def test_config_invalid_max_pairs(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(max_pairs=0)

    def test_make_match_entry_basic(self):
        e = make_match_entry(0, 1, 0.8, hu_dist=1.2, iou=0.6, chamfer=0.3, rank=2)
        assert e.idx1 == 0 and e.idx2 == 1
        assert e.score == pytest.approx(0.8)
        assert e.hu_dist == pytest.approx(1.2)
        assert e.rank == 2

    def test_entry_is_good_property(self):
        assert make_match_entry(0, 1, 0.9).is_good is True
        assert make_match_entry(0, 1, 0.5).is_good is False
        assert make_match_entry(0, 1, 0.0).is_good is False

    def test_entries_from_results(self):
        results = [(0, 1, 0.7), (2, 3, 0.3)]
        entries = entries_from_results(results)
        assert len(entries) == 2
        assert entries[0].rank == 0
        assert entries[1].rank == 1

    def test_summarise_matches_populated(self):
        entries = self._make_entries([0.8, 0.2, 0.6, 0.4])
        s = summarise_matches(entries)
        assert s.n_total == 4
        assert s.n_good == 2
        assert s.n_poor == 2
        assert s.mean_score == pytest.approx(0.5)
        assert s.max_score == pytest.approx(0.8)
        assert s.min_score == pytest.approx(0.2)

    def test_summarise_matches_empty(self):
        s = summarise_matches([])
        assert s.n_total == 0
        assert s.mean_score == 0.0

    def test_filter_good_and_poor_matches(self):
        entries = self._make_entries([0.9, 0.1, 0.7, 0.4])
        good = filter_good_matches(entries)
        poor = filter_poor_matches(entries)
        assert len(good) == 2
        assert len(poor) == 2
        assert all(e.is_good for e in good)
        assert all(not e.is_good for e in poor)

    def test_filter_by_hu_dist(self):
        e1 = make_match_entry(0, 1, 0.8, hu_dist=5.0)
        e2 = make_match_entry(1, 2, 0.7, hu_dist=15.0)
        result = filter_by_hu_dist([e1, e2], max_hu=10.0)
        assert len(result) == 1
        assert result[0].hu_dist == pytest.approx(5.0)

    def test_filter_match_by_score_range(self):
        entries = self._make_entries([0.1, 0.4, 0.7, 0.95])
        result = filter_match_by_score_range(entries, lo=0.3, hi=0.8)
        assert all(0.3 <= e.score <= 0.8 for e in result)
        assert len(result) == 2

    def test_top_k_match_entries(self):
        entries = self._make_entries([0.3, 0.9, 0.5, 0.8, 0.1])
        top3 = top_k_match_entries(entries, k=3)
        assert len(top3) == 3
        assert top3[0].score == pytest.approx(0.9)
        assert top3[1].score == pytest.approx(0.8)

    def test_match_entry_stats(self):
        entries = [
            make_match_entry(0, 1, 0.6, hu_dist=2.0, iou=0.5, chamfer=1.0),
            make_match_entry(1, 2, 0.4, hu_dist=4.0, iou=0.3, chamfer=3.0),
        ]
        stats = match_entry_stats(entries)
        assert stats["n"] == 2
        assert stats["mean_score"] == pytest.approx(0.5)
        assert stats["mean_hu_dist"] == pytest.approx(3.0)

    def test_compare_match_summaries(self):
        ea = self._make_entries([0.8, 0.9])
        eb = self._make_entries([0.3, 0.4])
        sa, sb = summarise_matches(ea), summarise_matches(eb)
        delta = compare_match_summaries(sa, sb)
        assert delta["n_total_delta"] == 0
        assert delta["mean_score_delta"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. spatial_index
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpatialIndex:

    def _build(self, n=10, scale=200.0):
        pts = rng.random((n, 2)) * scale
        return build_spatial_index(pts), pts

    def test_spatial_config_defaults(self):
        cfg = SpatialConfig()
        assert cfg.cell_size == 50.0
        assert cfg.metric == "euclidean"

    def test_spatial_config_invalid_cell_size(self):
        with pytest.raises(ValueError):
            SpatialConfig(cell_size=0.0)

    def test_spatial_config_invalid_metric(self):
        with pytest.raises(ValueError):
            SpatialConfig(metric="minkowski")

    def test_build_spatial_index_size(self):
        idx, _ = self._build(8)
        assert len(idx) == 8

    def test_insert_and_contains(self):
        idx = SpatialIndex()
        entry = SpatialEntry(item_id=5, position=np.array([10.0, 20.0]))
        idx.insert(entry)
        assert 5 in idx
        assert 99 not in idx

    def test_remove_entry(self):
        idx = SpatialIndex()
        idx.insert(SpatialEntry(item_id=0, position=np.array([0.0, 0.0])))
        removed = idx.remove(0)
        assert removed is True
        assert 0 not in idx
        assert idx.size == 0

    def test_query_radius_returns_within_radius(self):
        idx, pts = self._build(20)
        center = np.array([100.0, 100.0])
        results = idx.query_radius(center, radius=80.0)
        for dist, entry in results:
            assert dist <= 80.0 + 1e-9

    def test_query_radius_sorted(self):
        idx, pts = self._build(15)
        center = np.array([50.0, 50.0])
        results = idx.query_radius(center, radius=200.0)
        dists = [d for d, _ in results]
        assert dists == sorted(dists)

    def test_query_knn_count(self):
        idx, _ = self._build(10)
        results = idx.query_knn(np.array([100.0, 100.0]), k=4)
        assert len(results) == 4

    def test_pairwise_distances_shape(self):
        pts = rng.random((5, 2)) * 100
        D = pairwise_distances(pts)
        assert D.shape == (5, 5)
        assert np.allclose(D, D.T)
        assert np.allclose(np.diag(D), 0.0)

    def test_cluster_by_distance(self):
        # Two tight clusters far apart
        cluster_a = rng.random((5, 2)) * 5
        cluster_b = rng.random((5, 2)) * 5 + 500
        pts = np.vstack([cluster_a, cluster_b])
        # Returns list of clusters (list of lists of indices)
        clusters = cluster_by_distance(pts, threshold=20.0)
        assert len(clusters) == 2
        # Each cluster should have 5 points
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [5, 5]
        # The two clusters should not share any indices
        all_indices = [i for cluster in clusters for i in cluster]
        assert sorted(all_indices) == list(range(10))

    def test_query_radius_invalid_radius(self):
        idx = SpatialIndex()
        with pytest.raises(ValueError):
            idx.query_radius(np.array([0.0, 0.0]), radius=-1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. stats_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatsUtils:

    def _arr(self, n=20):
        return rng.random(n).astype(np.float64)

    def test_stats_config_validation(self):
        with pytest.raises(ValueError):
            StatsConfig(outlier_iqr_k=-1.0)
        with pytest.raises(ValueError):
            StatsConfig(winsor_low=0.9, winsor_high=0.1)
        with pytest.raises(ValueError):
            StatsConfig(ddof=2)

    def test_describe_keys(self):
        d = describe(self._arr())
        for key in ("min", "max", "mean", "std", "median", "q25", "q75", "iqr"):
            assert key in d

    def test_describe_values_consistent(self):
        a = self._arr(50)
        d = describe(a)
        assert d["min"] <= d["q25"] <= d["median"] <= d["q75"] <= d["max"]
        assert d["iqr"] == pytest.approx(d["q75"] - d["q25"])

    def test_describe_raises_on_empty(self):
        with pytest.raises(ValueError):
            describe(np.array([]))

    def test_zscore_mean_zero_std_one(self):
        a = self._arr(30)
        z = zscore_array(a)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std() - 1.0) < 1e-10

    def test_zscore_constant_array(self):
        a = np.ones(10)
        z = zscore_array(a)
        assert np.all(z == 0.0)

    def test_iqr_known_value(self):
        a = np.arange(1, 101, dtype=float)
        val = iqr(a)
        assert val == pytest.approx(50.0, abs=1.0)

    def test_winsorize_clips_extremes(self):
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 100.0])
        cfg = StatsConfig(winsor_low=0.0, winsor_high=0.8)
        w = winsorize(a, cfg)
        assert w.max() <= np.percentile(a, 80) + 1e-9

    def test_percentile_rank_bounds(self):
        a = self._arr(100)
        rank = percentile_rank(a, a.min() - 1)
        assert rank == pytest.approx(0.0)
        rank2 = percentile_rank(a, a.max() + 1)
        assert rank2 == pytest.approx(100.0)

    def test_outlier_mask_detects_outliers(self):
        a = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0])
        mask = outlier_mask(a)
        assert mask[-1] is np.bool_(True)
        assert mask[0] is np.bool_(False) or not mask[:-1].any()

    def test_running_stats_cumsum(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        rs = running_stats(a)
        assert np.allclose(rs["cumsum"], [1, 3, 6, 10])
        assert np.allclose(rs["cummean"], [1, 1.5, 2, 2.5])

    def test_weighted_mean_uniform(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        assert weighted_mean(a, w) == pytest.approx(2.5)

    def test_batch_describe_length(self):
        arrays = [self._arr(10), self._arr(20), self._arr(15)]
        results = batch_describe(arrays)
        assert len(results) == 3
        for r in results:
            assert "mean" in r


# ═══════════════════════════════════════════════════════════════════════════════
# 4. text_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestTextUtils:

    def _binary(self, h=40, w=80, density=0.3):
        arr = (rng.random((h, w)) < density).astype(np.uint8)
        return arr

    def test_text_config_defaults(self):
        cfg = TextConfig()
        assert cfg.min_word_gap == 4
        assert cfg.line_threshold == 0.05
        assert cfg.strip_punct is False

    def test_text_config_invalid(self):
        with pytest.raises(ValueError):
            TextConfig(min_word_gap=-1)
        with pytest.raises(ValueError):
            TextConfig(line_threshold=1.5)
        with pytest.raises(ValueError):
            TextConfig(min_line_height=0)

    def test_textblock_area_and_center(self):
        tb = TextBlock(text="hello", x=10, y=20, w=40, h=20)
        assert tb.area == 800
        assert tb.center == (30.0, 30.0)

    def test_textblock_n_chars(self):
        tb = TextBlock(text="hello world", x=0, y=0, w=10, h=10)
        assert tb.n_chars == 10

    def test_clean_ocr_text_strips_extra_spaces(self):
        result = clean_ocr_text("  hello   world  ")
        assert result == "hello world"

    def test_clean_ocr_text_lowercase(self):
        cfg = TextConfig(lowercase=True)
        result = clean_ocr_text("Hello WORLD", cfg)
        assert result == "hello world"

    def test_clean_ocr_text_strip_punct(self):
        cfg = TextConfig(strip_punct=True)
        result = clean_ocr_text("Hello, World!", cfg)
        assert "," not in result and "!" not in result

    def test_estimate_text_density_zero(self):
        binary = np.zeros((20, 30), dtype=np.uint8)
        assert estimate_text_density(binary) == pytest.approx(0.0)

    def test_estimate_text_density_full(self):
        binary = np.ones((20, 30), dtype=np.uint8)
        assert estimate_text_density(binary) == pytest.approx(1.0)

    def test_estimate_text_density_invalid(self):
        with pytest.raises(ValueError):
            estimate_text_density(np.ones((10, 10, 3), dtype=np.uint8))

    def test_find_text_lines_returns_list(self):
        binary = self._binary(h=60, w=100, density=0.4)
        lines = find_text_lines(binary)
        assert isinstance(lines, list)
        for y0, y1 in lines:
            assert y0 < y1

    def test_segment_words_returns_list(self):
        binary_line = self._binary(h=10, w=80, density=0.5)
        words = segment_words(binary_line)
        assert isinstance(words, list)
        for x0, x1 in words:
            assert x0 < x1

    def test_compare_text_blocks_identical(self):
        tb = TextBlock(text="hello", x=0, y=0, w=10, h=10)
        score = compare_text_blocks(tb, tb)
        assert score == pytest.approx(1.0)

    def test_align_text_blocks_order(self):
        blocks = [
            TextBlock(text="c", x=50, y=10, w=5, h=5),
            TextBlock(text="a", x=10, y=0, w=5, h=5),
            TextBlock(text="b", x=10, y=10, w=5, h=5),
        ]
        aligned = align_text_blocks(blocks)
        assert aligned[0].text == "a"

    def test_batch_clean_text_length(self):
        texts = ["  hello  ", "WORLD!", "test  text"]
        result = batch_clean_text(texts)
        assert len(result) == len(texts)
        assert result[0] == "hello"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. texture_pipeline_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestTexturePipelineUtils:

    def _make_record(self, score=0.7, pair=(0, 1)):
        return make_texture_match_record(
            pair=pair, score=score,
            lbp_score=score * 0.9,
            gabor_score=score * 1.1 if score * 1.1 <= 1 else 1.0,
            gradient_score=score,
        )

    def _make_batch_record(self, batch_id=0, n=10, done=8, failed=1, skipped=1):
        return make_batch_pipeline_record(
            batch_id=batch_id, n_items=n, n_done=done,
            n_failed=failed, n_skipped=skipped,
            elapsed=1.0, stage="texture",
        )

    def test_make_texture_match_record_fields(self):
        r = self._make_record(score=0.8, pair=(2, 5))
        assert r.pair == (2, 5)
        assert r.score == pytest.approx(0.8)
        assert r.lbp_score == pytest.approx(0.72)

    def test_summarise_texture_matches_empty(self):
        s = summarise_texture_matches([])
        assert s.n_pairs == 0
        assert s.best_pair is None

    def test_summarise_texture_matches_populated(self):
        records = [self._make_record(s) for s in [0.6, 0.8, 0.4]]
        s = summarise_texture_matches(records)
        assert s.n_pairs == 3
        assert s.best_score == pytest.approx(0.8)
        assert s.mean_score == pytest.approx((0.6 + 0.8 + 0.4) / 3)

    def test_filter_texture_by_score(self):
        records = [self._make_record(s) for s in [0.3, 0.6, 0.9]]
        filtered = filter_texture_by_score(records, threshold=0.5)
        assert len(filtered) == 2
        assert all(r.score >= 0.5 for r in filtered)

    def test_filter_texture_by_lbp(self):
        records = [self._make_record(s) for s in [0.3, 0.7, 0.9]]
        filtered = filter_texture_by_lbp(records, threshold=0.5)
        for r in filtered:
            assert r.lbp_score >= 0.5

    def test_top_k_texture_records(self):
        records = [self._make_record(s) for s in [0.2, 0.8, 0.5, 0.9, 0.1]]
        top2 = top_k_texture_records(records, k=2)
        assert len(top2) == 2
        assert top2[0].score == pytest.approx(0.9)

    def test_best_texture_record(self):
        records = [self._make_record(s) for s in [0.3, 0.95, 0.7]]
        best = best_texture_record(records)
        assert best.score == pytest.approx(0.95)

    def test_best_texture_record_empty(self):
        assert best_texture_record([]) is None

    def test_texture_score_stats_empty(self):
        stats = texture_score_stats([])
        assert stats["count"] == 0

    def test_texture_score_stats_values(self):
        records = [self._make_record(s) for s in [0.2, 0.6, 0.8]]
        stats = texture_score_stats(records)
        assert stats["count"] == 3
        assert stats["min"] == pytest.approx(0.2)
        assert stats["max"] == pytest.approx(0.8)
        assert stats["mean"] == pytest.approx((0.2 + 0.6 + 0.8) / 3)

    def test_batch_pipeline_record_success_rate(self):
        r = self._make_batch_record(n=10, done=8, failed=1, skipped=1)
        assert r.success_rate == pytest.approx(0.8)

    def test_batch_pipeline_record_throughput(self):
        r = make_batch_pipeline_record(
            batch_id=1, n_items=10, n_done=5, n_failed=0,
            n_skipped=0, elapsed=2.0, stage="gabor",
        )
        assert r.throughput == pytest.approx(2.5)

    def test_summarise_batch_pipeline(self):
        records = [
            self._make_batch_record(batch_id=0, n=10, done=10, failed=0, skipped=0),
            self._make_batch_record(batch_id=1, n=10, done=5,  failed=3, skipped=2),
        ]
        s = summarise_batch_pipeline(records)
        assert s.n_batches == 2
        assert s.total_items == 20
        assert s.total_done == 15
        assert s.best_batch_id == 0
        assert s.worst_batch_id == 1
