"""Integration tests for utils batch 5b.

Covers:
  - puzzle_reconstruction.utils.quality_score_utils
  - puzzle_reconstruction.utils.rank_result_utils
  - puzzle_reconstruction.utils.ranking_layout_utils
  - puzzle_reconstruction.utils.ranking_validation_utils
  - puzzle_reconstruction.utils.render_utils
"""
import math
import pytest
import numpy as np

rng = np.random.default_rng(42)

# ── imports ───────────────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.quality_score_utils import (
    QualityScoreConfig, QualityScoreEntry, QualitySummary,
    make_quality_entry, entries_from_reports, summarise_quality,
    filter_acceptable, filter_rejected, filter_by_overall, filter_by_blur,
    top_k_quality_entries, quality_score_stats, compare_quality,
)
from puzzle_reconstruction.utils.rank_result_utils import (
    RankResultConfig, RankResultEntry, RankResultSummary,
    make_rank_result_entry, entries_from_ranked_pairs,
    summarise_rank_results, filter_high_rank_entries, filter_low_rank_entries,
    filter_by_rank_position, filter_rank_by_score_range,
    filter_rank_by_dominant_channel, top_k_rank_entries, best_rank_entry,
    rerank_entries, rank_result_stats, compare_rank_summaries,
)
from puzzle_reconstruction.utils.ranking_layout_utils import (
    GlobalRankingConfig, GlobalRankingEntry, GlobalRankingSummary,
    make_global_ranking_entry, summarise_global_ranking_entries,
    filter_ranking_by_min_score, filter_ranking_by_fragment,
    filter_ranking_by_top_k, top_k_ranking_entries, best_ranking_entry,
    ranking_score_stats, compare_global_ranking_summaries,
    batch_summarise_global_ranking_entries,
    LayoutScoringEntry, LayoutScoringSummary,
    make_layout_scoring_entry, summarise_layout_scoring_entries,
    filter_layout_by_min_score, filter_layout_by_quality,
)
from puzzle_reconstruction.utils.ranking_validation_utils import (
    RankingRunRecord, CandidateSummary, ScoreVectorRecord,
    ValidationRunRecord, BoundaryCheckSummary, PaletteComparisonRecord,
    PaletteRankingRecord, make_ranking_record, make_validation_record,
)
from puzzle_reconstruction.utils.render_utils import (
    CanvasConfig, MosaicConfig,
    rotation_matrix_2d, bounding_box_of_rotated, compute_canvas_size,
    make_blank_canvas, resize_keep_aspect, pad_to_square, make_thumbnail,
    paste_with_mask, compute_grid_layout, make_mosaic, horizontal_concat,
)


# =============================================================================
# 1. quality_score_utils
# =============================================================================

class TestQualityScoreUtils:

    def _make_entries(self, n=6):
        scores = rng.random((n, 5)).tolist()
        return [
            make_quality_entry(i, s[0], s[1], s[2], s[3], s[4])
            for i, s in enumerate(scores)
        ]

    def test_config_valid(self):
        cfg = QualityScoreConfig(min_overall=0.6, min_blur=0.1)
        assert cfg.min_overall == 0.6

    def test_config_invalid_raises(self):
        with pytest.raises(ValueError):
            QualityScoreConfig(min_overall=1.5)

    def test_make_quality_entry_acceptable(self):
        entry = make_quality_entry(0, 0.8, 0.7, 0.9, 1.0, 0.85)
        assert entry.is_acceptable
        assert entry.image_id == 0

    def test_make_quality_entry_rejected(self):
        entry = make_quality_entry(1, 0.3, 0.3, 0.3, 0.3, 0.2)
        assert not entry.is_acceptable

    def test_make_quality_entry_meta(self):
        entry = make_quality_entry(2, 0.5, 0.5, 0.5, 0.5, 0.6, meta={"src": "cam"})
        assert entry.meta["src"] == "cam"

    def test_entries_from_reports(self):
        reports = [
            {"image_id": 10, "blur_score": 0.9, "noise_score": 0.8,
             "contrast_score": 0.7, "completeness": 1.0, "overall": 0.85},
            {"image_id": 11, "blur_score": 0.2, "noise_score": 0.3,
             "contrast_score": 0.1, "completeness": 0.5, "overall": 0.25},
        ]
        entries = entries_from_reports(reports)
        assert len(entries) == 2
        assert entries[0].image_id == 10
        assert entries[1].image_id == 11

    def test_summarise_quality_counts(self):
        entries = self._make_entries(8)
        s = summarise_quality(entries)
        assert s.n_total == 8
        assert s.n_acceptable + s.n_rejected == 8

    def test_summarise_quality_empty(self):
        s = summarise_quality([])
        assert s.n_total == 0 and s.mean_overall == 0.0

    def test_filter_acceptable_rejected(self):
        entries = self._make_entries(10)
        acc = filter_acceptable(entries)
        rej = filter_rejected(entries)
        assert len(acc) + len(rej) == 10
        assert all(e.is_acceptable for e in acc)
        assert all(not e.is_acceptable for e in rej)

    def test_filter_by_overall(self):
        entries = self._make_entries(10)
        filtered = filter_by_overall(entries, min_overall=0.7)
        assert all(e.overall >= 0.7 for e in filtered)

    def test_filter_by_blur(self):
        entries = self._make_entries(10)
        filtered = filter_by_blur(entries, min_blur=0.5)
        assert all(e.blur_score >= 0.5 for e in filtered)

    def test_top_k_quality_entries(self):
        entries = self._make_entries(10)
        top3 = top_k_quality_entries(entries, 3)
        assert len(top3) == 3
        assert top3[0].overall >= top3[1].overall >= top3[2].overall

    def test_quality_score_stats_keys(self):
        entries = self._make_entries(5)
        stats = quality_score_stats(entries)
        for key in ("count", "mean", "std", "min", "max", "n_acceptable", "n_rejected"):
            assert key in stats

    def test_compare_quality(self):
        e1 = self._make_entries(4)
        e2 = self._make_entries(6)
        s1, s2 = summarise_quality(e1), summarise_quality(e2)
        delta = compare_quality(s1, s2)
        assert delta["n_total_delta"] == -2


# =============================================================================
# 2. rank_result_utils
# =============================================================================

class TestRankResultUtils:

    def _make_entries(self, n=6):
        scores = rng.uniform(0, 1, n).tolist()
        return [make_rank_result_entry(i, i + 1, scores[i], i + 1) for i in range(n)]

    def test_config_valid(self):
        cfg = RankResultConfig(good_threshold=0.8, poor_threshold=0.2, top_k=5)
        assert cfg.top_k == 5

    def test_config_invalid_good_threshold(self):
        with pytest.raises(ValueError):
            RankResultConfig(good_threshold=1.5)

    def test_make_entry_basic(self):
        e = make_rank_result_entry(0, 3, 0.75, 1)
        assert e.frag_i == 0 and e.frag_j == 3
        assert e.is_top_match

    def test_pair_key_ordering(self):
        e = make_rank_result_entry(5, 2, 0.5, 2)
        assert e.pair_key == (2, 5)

    def test_dominant_channel(self):
        e = make_rank_result_entry(0, 1, 0.5, 1,
                                   channel_scores={"R": 0.9, "G": 0.4, "B": 0.2})
        assert e.dominant_channel == "R"

    def test_entries_from_ranked_pairs(self):
        pairs = [(0, 1), (2, 3), (4, 5)]
        scores = [0.9, 0.6, 0.3]
        entries = entries_from_ranked_pairs(pairs, scores)
        assert len(entries) == 3
        assert entries[0].score == 0.9

    def test_entries_from_ranked_pairs_length_mismatch(self):
        with pytest.raises(ValueError):
            entries_from_ranked_pairs([(0, 1)], [0.5, 0.6])

    def test_summarise_rank_results(self):
        entries = self._make_entries(5)
        s = summarise_rank_results(entries)
        assert s.n_entries == 5
        assert s.min_score <= s.mean_score <= s.max_score

    def test_summarise_rank_results_empty(self):
        s = summarise_rank_results([])
        assert s.n_entries == 0

    def test_filter_high_rank_entries(self):
        entries = self._make_entries(10)
        high = filter_high_rank_entries(entries, threshold=0.5)
        assert all(e.score >= 0.5 for e in high)

    def test_filter_low_rank_entries(self):
        entries = self._make_entries(10)
        low = filter_low_rank_entries(entries, threshold=0.5)
        assert all(e.score < 0.5 for e in low)

    def test_filter_by_rank_position(self):
        entries = self._make_entries(5)
        filtered = filter_by_rank_position(entries, max_rank=3)
        assert all(e.rank <= 3 for e in filtered)

    def test_filter_rank_by_score_range(self):
        entries = self._make_entries(10)
        ranged = filter_rank_by_score_range(entries, lo=0.2, hi=0.8)
        assert all(0.2 <= e.score <= 0.8 for e in ranged)

    def test_filter_rank_by_dominant_channel(self):
        entries = [
            make_rank_result_entry(0, 1, 0.8, 1, channel_scores={"R": 0.9, "G": 0.1}),
            make_rank_result_entry(1, 2, 0.5, 2, channel_scores={"R": 0.2, "G": 0.7}),
        ]
        r_entries = filter_rank_by_dominant_channel(entries, "R")
        assert len(r_entries) == 1 and r_entries[0].dominant_channel == "R"

    def test_top_k_rank_entries(self):
        entries = self._make_entries(8)
        top3 = top_k_rank_entries(entries, k=3)
        assert len(top3) == 3
        assert top3[0].score >= top3[-1].score

    def test_best_rank_entry(self):
        entries = self._make_entries(6)
        best = best_rank_entry(entries)
        assert best.score == max(e.score for e in entries)

    def test_rerank_entries(self):
        entries = self._make_entries(4)
        reranked = rerank_entries(entries)
        assert [e.rank for e in reranked] == [1, 2, 3, 4]
        assert reranked[0].score >= reranked[-1].score

    def test_rank_result_stats_keys(self):
        entries = self._make_entries(5)
        stats = rank_result_stats(entries)
        assert "count" in stats and "mean_score" in stats

    def test_compare_rank_summaries(self):
        ea = self._make_entries(4)
        eb = self._make_entries(6)
        sa, sb = summarise_rank_results(ea), summarise_rank_results(eb)
        d = compare_rank_summaries(sa, sb)
        assert d["d_n_entries"] == -2


# =============================================================================
# 3. ranking_layout_utils  (GlobalRanking + LayoutScoring)
# =============================================================================

class TestRankingLayoutUtils:

    def _make_global_entries(self, n=5):
        return [
            make_global_ranking_entry(i, i + 1, float(rng.random()), i)
            for i in range(n)
        ]

    def _make_layout_entries(self, n=4):
        levels = ["poor", "fair", "good", "excellent"]
        return [
            make_layout_scoring_entry(
                i, float(rng.random()), float(rng.random()),
                float(rng.random()), float(rng.random()), 10,
                quality_level=levels[i % 4],
            )
            for i in range(n)
        ]

    def test_make_global_ranking_entry(self):
        e = make_global_ranking_entry(0, 1, 0.9, 0, color=0.8, edge=0.7)
        assert e.idx1 == 0 and e.score == pytest.approx(0.9)
        assert "color" in e.component_scores

    def test_summarise_global_ranking_empty(self):
        s = summarise_global_ranking_entries([])
        assert s.n_pairs == 0 and s.top_pair is None

    def test_summarise_global_ranking(self):
        entries = self._make_global_entries(5)
        s = summarise_global_ranking_entries(entries)
        assert s.n_pairs == 5
        assert s.min_score <= s.mean_score <= s.max_score

    def test_filter_ranking_by_min_score(self):
        entries = self._make_global_entries(8)
        filtered = filter_ranking_by_min_score(entries, 0.5)
        assert all(e.score >= 0.5 for e in filtered)

    def test_filter_ranking_by_fragment(self):
        entries = [
            make_global_ranking_entry(0, 1, 0.5, 0),
            make_global_ranking_entry(2, 3, 0.6, 1),
            make_global_ranking_entry(1, 4, 0.7, 2),
        ]
        result = filter_ranking_by_fragment(entries, 1)
        assert all(e.idx1 == 1 or e.idx2 == 1 for e in result)
        assert len(result) == 2

    def test_filter_ranking_by_top_k(self):
        entries = self._make_global_entries(6)
        top3 = filter_ranking_by_top_k(entries, 3)
        assert len(top3) == 3

    def test_top_k_ranking_entries(self):
        entries = self._make_global_entries(6)
        top2 = top_k_ranking_entries(entries, 2)
        assert top2[0].score >= top2[1].score

    def test_best_ranking_entry(self):
        entries = self._make_global_entries(5)
        best = best_ranking_entry(entries)
        assert best.score == max(e.score for e in entries)

    def test_ranking_score_stats(self):
        entries = self._make_global_entries(5)
        stats = ranking_score_stats(entries)
        assert stats["count"] == 5
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_compare_global_ranking_summaries(self):
        ea = self._make_global_entries(3)
        eb = self._make_global_entries(5)
        sa = summarise_global_ranking_entries(ea)
        sb = summarise_global_ranking_entries(eb)
        d = compare_global_ranking_summaries(sa, sb)
        assert "delta_mean_score" in d
        assert d["delta_n_pairs"] == 2

    def test_batch_summarise(self):
        groups = [self._make_global_entries(3), self._make_global_entries(4)]
        summaries = batch_summarise_global_ranking_entries(groups)
        assert len(summaries) == 2

    def test_make_layout_scoring_entry(self):
        e = make_layout_scoring_entry(0, 0.8, 0.9, 0.05, 0.75, 12, quality_level="good")
        assert e.layout_id == 0 and e.quality_level == "good"

    def test_summarise_layout_scoring_empty(self):
        s = summarise_layout_scoring_entries([])
        assert s.n_layouts == 0 and s.best_layout_id is None

    def test_summarise_layout_scoring(self):
        entries = self._make_layout_entries(4)
        s = summarise_layout_scoring_entries(entries)
        assert s.n_layouts == 4
        assert s.best_layout_id is not None

    def test_filter_layout_by_min_score(self):
        entries = self._make_layout_entries(6)
        filtered = filter_layout_by_min_score(entries, 0.5)
        assert all(e.total_score >= 0.5 for e in filtered)

    def test_filter_layout_by_quality(self):
        entries = self._make_layout_entries(8)
        good = filter_layout_by_quality(entries, "good")
        assert all(e.quality_level == "good" for e in good)


# =============================================================================
# 4. ranking_validation_utils
# =============================================================================

class TestRankingValidationUtils:

    def test_ranking_run_record_basic(self):
        r = RankingRunRecord(n_fragments=10, n_pairs_ranked=45, top_score=0.9)
        assert r.has_results
        assert r.n_fragments == 10

    def test_ranking_run_record_invalid_fragments(self):
        with pytest.raises(ValueError):
            RankingRunRecord(n_fragments=-1, n_pairs_ranked=0)

    def test_ranking_run_record_invalid_top_score(self):
        with pytest.raises(ValueError):
            RankingRunRecord(n_fragments=5, n_pairs_ranked=10, top_score=1.5)

    def test_ranking_run_record_no_results(self):
        r = RankingRunRecord(n_fragments=5, n_pairs_ranked=0)
        assert not r.has_results

    def test_candidate_summary_basic(self):
        cs = CandidateSummary(fragment_id=3, n_candidates=5, best_score=0.8, best_partner=7)
        assert cs.has_candidates
        assert cs.best_partner == 7

    def test_candidate_summary_invalid(self):
        with pytest.raises(ValueError):
            CandidateSummary(fragment_id=-1, n_candidates=0)

    def test_score_vector_record(self):
        scores = rng.random(5).tolist()
        sv = ScoreVectorRecord(n_fragments=5, scores=scores)
        assert sv.max_score == pytest.approx(max(scores))
        assert sv.mean_score == pytest.approx(sum(scores) / 5)

    def test_score_vector_record_empty(self):
        sv = ScoreVectorRecord(n_fragments=3, scores=[])
        assert sv.max_score == 0.0
        assert sv.mean_score == 0.0

    def test_score_vector_record_invalid_length(self):
        with pytest.raises(ValueError):
            ScoreVectorRecord(n_fragments=5, scores=[0.1, 0.2])

    def test_validation_run_record(self):
        vr = ValidationRunRecord(step=1, n_pairs=10, n_violations=2, quality_score=0.8)
        assert vr.violation_rate == pytest.approx(0.2)
        assert not vr.is_clean

    def test_validation_run_record_clean(self):
        vr = ValidationRunRecord(step=0, n_pairs=5, n_violations=0, quality_score=1.0)
        assert vr.is_clean

    def test_boundary_check_summary_dominant_violation(self):
        bcs = BoundaryCheckSummary(
            n_assemblies=3, mean_quality=0.8,
            violation_types={"gap": 5, "overlap": 2, "misalign": 1},
        )
        assert bcs.dominant_violation == "gap"

    def test_palette_comparison_record_similarity(self):
        pc = PaletteComparisonRecord(fragment_id_a=0, fragment_id_b=1,
                                      distance=127.5, n_colors=8)
        assert pc.similarity == pytest.approx(0.5)

    def test_palette_comparison_record_invalid_distance(self):
        with pytest.raises(ValueError):
            PaletteComparisonRecord(fragment_id_a=0, fragment_id_b=1,
                                     distance=-1.0, n_colors=4)

    def test_palette_ranking_record_top_k(self):
        pr = PaletteRankingRecord(query_id=0, ranked_ids=[3, 1, 4, 1, 5],
                                   similarities=[0.9, 0.8, 0.7, 0.6, 0.5])
        top2 = pr.top_k(2)
        assert len(top2) == 2
        assert top2[0] == (3, 0.9)

    def test_make_ranking_record(self):
        r = make_ranking_record(8, 28, top_score=0.95, label="run1")
        assert isinstance(r, RankingRunRecord)
        assert r.label == "run1"

    def test_make_validation_record(self):
        v = make_validation_record(step=2, n_pairs=20, n_violations=4,
                                    quality_score=0.8)
        assert isinstance(v, ValidationRunRecord)
        assert v.violation_rate == pytest.approx(0.2)


# =============================================================================
# 5. render_utils
# =============================================================================

class TestRenderUtils:

    def test_rotation_matrix_identity(self):
        R = rotation_matrix_2d(0.0)
        np.testing.assert_allclose(R, np.eye(2), atol=1e-10)

    def test_rotation_matrix_90(self):
        R = rotation_matrix_2d(math.pi / 2)
        expected = np.array([[0, -1], [1, 0]], dtype=np.float64)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_bounding_box_no_rotation(self):
        nw, nh = bounding_box_of_rotated(100, 80, 0.0)
        assert nw == 100 and nh == 80

    def test_bounding_box_90_rotation(self):
        nw, nh = bounding_box_of_rotated(100, 60, math.pi / 2)
        assert nw == 60 and nh == 100

    def test_compute_canvas_size_empty(self):
        w, h = compute_canvas_size({}, {}, margin=20)
        assert w >= 1 and h >= 1

    def test_compute_canvas_size_single(self):
        pos = np.array([0.0, 0.0])
        placements = {0: (pos, 0.0)}
        frag_sizes = {0: (100, 80)}
        w, h = compute_canvas_size(placements, frag_sizes, margin=10)
        assert w >= 100 + 20 and h >= 80 + 20

    def test_make_blank_canvas_shape(self):
        canvas = make_blank_canvas(200, 150, color=(128, 64, 32))
        assert canvas.shape == (150, 200, 3)
        assert canvas[0, 0, 0] == 128

    def test_resize_keep_aspect(self):
        img = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
        resized = resize_keep_aspect(img, 60)
        assert max(resized.shape[:2]) == 60
        assert resized.shape[1] / resized.shape[0] == pytest.approx(120 / 80, rel=0.05)

    def test_pad_to_square(self):
        img = rng.integers(0, 255, (30, 50, 3), dtype=np.uint8)
        padded = pad_to_square(img, 64)
        assert padded.shape == (64, 64, 3)

    def test_make_thumbnail_shape(self):
        img = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
        thumb = make_thumbnail(img, thumb_size=64)
        assert thumb.shape == (64, 64, 3)

    def test_paste_with_mask_full(self):
        canvas = make_blank_canvas(100, 100, color=(0, 0, 0))
        fragment = np.full((20, 20, 3), 255, dtype=np.uint8)
        mask = np.full((20, 20), 255, dtype=np.uint8)
        result = paste_with_mask(canvas, fragment, mask, 10, 10)
        assert result[10, 10, 0] == 255

    def test_paste_with_mask_transparent(self):
        canvas = make_blank_canvas(100, 100, color=(50, 50, 50))
        fragment = np.full((20, 20, 3), 200, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        result = paste_with_mask(canvas, fragment, mask, 5, 5)
        assert result[5, 5, 0] == 50  # unchanged

    def test_compute_grid_layout_zero(self):
        assert compute_grid_layout(0) == (0, 0)

    def test_compute_grid_layout(self):
        rows, cols = compute_grid_layout(10, max_cols=4)
        assert cols == 4 and rows == 3

    def test_make_mosaic_shape(self):
        images = [rng.integers(0, 255, (50, 70, 3), dtype=np.uint8) for _ in range(6)]
        mosaic = make_mosaic(images)
        assert mosaic.ndim == 3

    def test_horizontal_concat_shape(self):
        images = [rng.integers(0, 255, (60, 80, 3), dtype=np.uint8) for _ in range(3)]
        result = horizontal_concat(images, gap=4)
        assert result.ndim == 3
        assert result.shape[0] == 60
