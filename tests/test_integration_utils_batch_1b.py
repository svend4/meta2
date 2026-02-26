"""Integration tests for utils batch 1b.

Covers:
  - puzzle_reconstruction.utils.blend_utils
  - puzzle_reconstruction.utils.candidate_rank_utils
  - puzzle_reconstruction.utils.canvas_build_utils
  - puzzle_reconstruction.utils.color_edge_export_utils
  - puzzle_reconstruction.utils.color_hist_utils
"""
import pytest
import numpy as np

rng = np.random.default_rng(42)

# ── imports ──────────────────────────────────────────────────────────────────

from puzzle_reconstruction.utils.blend_utils import (
    BlendConfig,
    alpha_blend,
    weighted_blend,
    feather_mask,
    paste_with_mask,
)
from puzzle_reconstruction.utils.candidate_rank_utils import (
    CandidateRankConfig,
    CandidateRankEntry,
    make_candidate_entry,
    entries_from_pairs,
    summarise_rankings,
    filter_selected,
    filter_rejected_candidates,
    filter_by_score_range,
    filter_by_rank,
    top_k_candidate_entries,
    candidate_rank_stats,
)
from puzzle_reconstruction.utils.canvas_build_utils import (
    CanvasBuildConfig,
    PlacementEntry,
    make_placement_entry,
    entries_from_placements,
    summarise_canvas_build,
    filter_by_area,
    filter_by_coverage_contribution,
    top_k_by_coverage,
    canvas_build_stats,
    compare_canvas_summaries,
    batch_summarise_canvas_builds,
)
from puzzle_reconstruction.utils.color_edge_export_utils import (
    ColorMatchAnalysisConfig,
    ColorMatchAnalysisEntry,
    make_color_match_analysis_entry,
    summarise_color_match_analysis,
    filter_strong_color_matches,
    filter_weak_color_matches,
    filter_color_by_method,
    top_k_color_match_entries,
    best_color_match_entry,
    color_match_analysis_stats,
    compare_color_match_summaries,
    batch_summarise_color_match_analysis,
    EdgeDetectionAnalysisEntry,
    make_edge_detection_entry,
)
from puzzle_reconstruction.utils.color_hist_utils import (
    ColorHistConfig,
    ColorHistEntry,
    ColorHistSummary,
    make_color_hist_entry,
    entries_from_comparisons,
    summarise_color_hist,
    filter_good_hist_entries,
    filter_poor_hist_entries,
    filter_by_intersection_range,
    filter_by_chi2_range,
    filter_by_space,
    top_k_hist_entries,
    best_hist_entry,
)


# ═══════════════════════════════════════════════════════════════════════════════
# blend_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlendUtils:

    def _img(self, h=20, w=20, c=3):
        return rng.integers(0, 256, (h, w, c), dtype=np.uint8)

    def test_blend_config_defaults(self):
        cfg = BlendConfig()
        assert cfg.feather_px == 8
        assert cfg.gamma == 1.0
        assert cfg.clip_output is True

    def test_blend_config_invalid_feather(self):
        with pytest.raises(ValueError, match="feather_px"):
            BlendConfig(feather_px=-1)

    def test_blend_config_invalid_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            BlendConfig(gamma=0.0)

    def test_alpha_blend_midpoint(self):
        src = np.full((4, 4, 3), 200, dtype=np.uint8)
        dst = np.full((4, 4, 3), 100, dtype=np.uint8)
        result = alpha_blend(src, dst, alpha=0.5)
        assert result.dtype == np.uint8
        assert result.shape == src.shape
        assert np.all(result == 150)

    def test_alpha_blend_alpha_zero(self):
        src = np.full((4, 4), 200, dtype=np.uint8)
        dst = np.full((4, 4), 50, dtype=np.uint8)
        result = alpha_blend(src, dst, alpha=0.0)
        assert np.array_equal(result, dst)

    def test_alpha_blend_alpha_one(self):
        src = np.full((4, 4), 200, dtype=np.uint8)
        dst = np.full((4, 4), 50, dtype=np.uint8)
        result = alpha_blend(src, dst, alpha=1.0)
        assert np.array_equal(result, src)

    def test_alpha_blend_shape_mismatch_raises(self):
        src = np.zeros((4, 4, 3), dtype=np.uint8)
        dst = np.zeros((4, 5, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            alpha_blend(src, dst, alpha=0.5)

    def test_alpha_blend_invalid_alpha_raises(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="alpha"):
            alpha_blend(img, img, alpha=1.5)

    def test_weighted_blend_equal_weights(self):
        a = np.full((4, 4, 3), 100, dtype=np.uint8)
        b = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = weighted_blend([a, b])
        assert result.dtype == np.uint8
        assert np.all(result == 150)

    def test_weighted_blend_custom_weights(self):
        a = np.full((4, 4), 0, dtype=np.uint8)
        b = np.full((4, 4), 200, dtype=np.uint8)
        result = weighted_blend([a, b], weights=[0.0, 1.0])
        assert np.all(result == 200)

    def test_weighted_blend_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_blend([])

    def test_feather_mask_shape_and_range(self):
        mask = feather_mask(40, 40, feather_px=8)
        assert mask.shape == (40, 40)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_feather_mask_center_is_one(self):
        mask = feather_mask(40, 40, feather_px=4)
        assert mask[20, 20] == pytest.approx(1.0)

    def test_paste_with_mask_output_shape(self):
        canvas = np.zeros((20, 20, 3), dtype=np.uint8)
        patch = np.full((10, 10, 3), 128, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, y=5, x=5)
        assert result.shape == canvas.shape
        assert result.dtype == np.uint8
        assert np.all(result[5:15, 5:15] == 128)


# ═══════════════════════════════════════════════════════════════════════════════
# candidate_rank_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestCandidateRankUtils:

    def _pairs(self):
        return [
            {"idx1": 0, "idx2": 1, "score": 0.9},
            {"idx1": 1, "idx2": 2, "score": 0.4},
            {"idx1": 2, "idx2": 3, "score": 0.6},
            {"idx1": 3, "idx2": 4, "score": 0.2},
        ]

    def test_config_defaults(self):
        cfg = CandidateRankConfig()
        assert cfg.min_score == 0.5
        assert cfg.max_pairs == 0
        assert cfg.deduplicate is True

    def test_config_invalid_min_score(self):
        with pytest.raises(ValueError, match="min_score"):
            CandidateRankConfig(min_score=1.5)

    def test_config_invalid_max_pairs(self):
        with pytest.raises(ValueError, match="max_pairs"):
            CandidateRankConfig(max_pairs=-1)

    def test_make_candidate_entry_selected(self):
        cfg = CandidateRankConfig(min_score=0.5)
        entry = make_candidate_entry(0, 1, 0.8, rank=0, cfg=cfg)
        assert entry.is_selected is True
        assert entry.idx1 == 0
        assert entry.score == pytest.approx(0.8)

    def test_make_candidate_entry_rejected(self):
        cfg = CandidateRankConfig(min_score=0.5)
        entry = make_candidate_entry(0, 1, 0.3, rank=1, cfg=cfg)
        assert entry.is_selected is False

    def test_entries_from_pairs_sorted_descending(self):
        entries = entries_from_pairs(self._pairs())
        scores = [e.score for e in entries]
        assert scores == sorted(scores, reverse=True)

    def test_summarise_rankings_counts(self):
        cfg = CandidateRankConfig(min_score=0.5)
        entries = entries_from_pairs(self._pairs(), cfg=cfg)
        summary = summarise_rankings(entries)
        assert summary.n_total == 4
        assert summary.n_selected + summary.n_rejected == summary.n_total

    def test_filter_selected(self):
        cfg = CandidateRankConfig(min_score=0.5)
        entries = entries_from_pairs(self._pairs(), cfg=cfg)
        selected = filter_selected(entries)
        assert all(e.is_selected for e in selected)

    def test_filter_rejected(self):
        cfg = CandidateRankConfig(min_score=0.5)
        entries = entries_from_pairs(self._pairs(), cfg=cfg)
        rejected = filter_rejected_candidates(entries)
        assert all(not e.is_selected for e in rejected)

    def test_filter_by_score_range(self):
        entries = entries_from_pairs(self._pairs())
        filtered = filter_by_score_range(entries, min_score=0.4, max_score=0.7)
        assert all(0.4 <= e.score <= 0.7 for e in filtered)

    def test_filter_by_rank(self):
        entries = entries_from_pairs(self._pairs())
        filtered = filter_by_rank(entries, max_rank=1)
        assert all(e.rank <= 1 for e in filtered)
        assert len(filtered) == 2

    def test_top_k_candidate_entries(self):
        entries = entries_from_pairs(self._pairs())
        top2 = top_k_candidate_entries(entries, k=2)
        assert len(top2) == 2
        assert top2[0].score >= top2[1].score

    def test_candidate_rank_stats_empty(self):
        stats = candidate_rank_stats([])
        assert stats["count"] == 0
        assert stats["mean"] == 0.0

    def test_candidate_rank_stats_populated(self):
        entries = entries_from_pairs(self._pairs())
        stats = candidate_rank_stats(entries)
        assert stats["count"] == 4
        assert 0.0 <= stats["mean"] <= 1.0
        assert stats["max"] >= stats["min"]


# ═══════════════════════════════════════════════════════════════════════════════
# canvas_build_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanvasBuildUtils:

    def _placements(self):
        return [(0, 0, 0, 10, 10), (1, 15, 15, 20, 20), (2, 5, 5, 8, 8)]

    def test_config_defaults(self):
        cfg = CanvasBuildConfig()
        assert cfg.blend_mode == "overwrite"
        assert cfg.max_fragments == 1000

    def test_config_invalid_coverage(self):
        with pytest.raises(ValueError, match="min_coverage"):
            CanvasBuildConfig(min_coverage=1.5)

    def test_config_invalid_blend_mode(self):
        with pytest.raises(ValueError, match="blend_mode"):
            CanvasBuildConfig(blend_mode="unknown")

    def test_placement_entry_area(self):
        entry = make_placement_entry(0, x=0, y=0, w=10, h=20)
        assert entry.area == 200

    def test_placement_entry_x2_y2(self):
        entry = make_placement_entry(1, x=5, y=10, w=15, h=25)
        assert entry.x2 == 20
        assert entry.y2 == 35

    def test_placement_entry_invalid_fragment_id(self):
        with pytest.raises(ValueError, match="fragment_id"):
            PlacementEntry(fragment_id=-1, x=0, y=0, w=10, h=10)

    def test_placement_entry_invalid_dimensions(self):
        with pytest.raises(ValueError):
            PlacementEntry(fragment_id=0, x=0, y=0, w=0, h=10)

    def test_entries_from_placements(self):
        entries = entries_from_placements(self._placements())
        assert len(entries) == 3
        assert entries[0].fragment_id == 0

    def test_summarise_canvas_build(self):
        entries = entries_from_placements(self._placements())
        summary = summarise_canvas_build(entries, canvas_w=100, canvas_h=100, coverage=0.5)
        assert summary.n_placed == 3
        assert summary.canvas_w == 100
        assert summary.total_area == sum(e.area for e in entries)

    def test_filter_by_area(self):
        entries = entries_from_placements(self._placements())
        filtered = filter_by_area(entries, min_area=64, max_area=400)
        assert all(64 <= e.area <= 400 for e in filtered)

    def test_filter_by_coverage_contribution(self):
        entries = [
            make_placement_entry(0, 0, 0, 10, 10, coverage_contribution=0.1),
            make_placement_entry(1, 10, 10, 10, 10, coverage_contribution=0.5),
        ]
        filtered = filter_by_coverage_contribution(entries, min_contrib=0.3)
        assert len(filtered) == 1
        assert filtered[0].fragment_id == 1

    def test_top_k_by_coverage(self):
        entries = [
            make_placement_entry(i, 0, 0, 10, 10, coverage_contribution=float(i) / 5)
            for i in range(5)
        ]
        top2 = top_k_by_coverage(entries, k=2)
        assert len(top2) == 2
        assert top2[0].coverage_contribution >= top2[1].coverage_contribution

    def test_canvas_build_stats(self):
        entries = entries_from_placements(self._placements())
        stats = canvas_build_stats(entries)
        assert stats["n"] == 3
        assert stats["total_area"] > 0

    def test_compare_canvas_summaries(self):
        e1 = entries_from_placements(self._placements())
        e2 = entries_from_placements([(0, 0, 0, 5, 5)])
        s1 = summarise_canvas_build(e1, 100, 100, 0.5)
        s2 = summarise_canvas_build(e2, 50, 50, 0.1)
        diff = compare_canvas_summaries(s1, s2)
        assert diff["n_placed_delta"] == 2
        assert diff["coverage_delta"] == pytest.approx(0.4)

    def test_batch_summarise_canvas_builds(self):
        e = entries_from_placements(self._placements())
        specs = [(e, 100, 100, 0.5), (e, 200, 200, 0.8)]
        summaries = batch_summarise_canvas_builds(specs)
        assert len(summaries) == 2
        assert summaries[0].canvas_w == 100
        assert summaries[1].canvas_w == 200


# ═══════════════════════════════════════════════════════════════════════════════
# color_edge_export_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestColorEdgeExportUtils:

    def _entries(self):
        return [
            make_color_match_analysis_entry(0, 1, 0.9, 0.8, 0.9, 0.95, "hsv"),
            make_color_match_analysis_entry(1, 2, 0.3, 0.3, 0.3, 0.3, "rgb"),
            make_color_match_analysis_entry(2, 3, 0.6, 0.6, 0.6, 0.6, "hsv"),
            make_color_match_analysis_entry(3, 4, 0.1, 0.1, 0.1, 0.1, "rgb"),
        ]

    def test_make_color_match_analysis_entry(self):
        entry = make_color_match_analysis_entry(0, 1, 0.75, 0.7, 0.8, 0.75)
        assert entry.idx1 == 0
        assert entry.idx2 == 1
        assert entry.score == pytest.approx(0.75)
        assert entry.method == "hsv"

    def test_summarise_color_match_empty(self):
        summary = summarise_color_match_analysis([])
        assert summary.n_entries == 0
        assert summary.mean_score == 0.0

    def test_summarise_color_match_populated(self):
        entries = self._entries()
        summary = summarise_color_match_analysis(entries)
        assert summary.n_entries == 4
        assert summary.min_score <= summary.mean_score <= summary.max_score

    def test_filter_strong_color_matches(self):
        entries = self._entries()
        strong = filter_strong_color_matches(entries, threshold=0.5)
        assert all(e.score >= 0.5 for e in strong)
        assert len(strong) == 2

    def test_filter_weak_color_matches(self):
        entries = self._entries()
        weak = filter_weak_color_matches(entries, threshold=0.5)
        assert all(e.score < 0.5 for e in weak)

    def test_filter_color_by_method(self):
        entries = self._entries()
        hsv_only = filter_color_by_method(entries, "hsv")
        assert all(e.method == "hsv" for e in hsv_only)
        assert len(hsv_only) == 2

    def test_top_k_color_match_entries(self):
        entries = self._entries()
        top2 = top_k_color_match_entries(entries, k=2)
        assert len(top2) == 2
        assert top2[0].score >= top2[1].score

    def test_best_color_match_entry(self):
        entries = self._entries()
        best = best_color_match_entry(entries)
        assert best is not None
        assert best.score == max(e.score for e in entries)

    def test_best_color_match_entry_empty(self):
        assert best_color_match_entry([]) is None

    def test_color_match_analysis_stats(self):
        entries = self._entries()
        stats = color_match_analysis_stats(entries)
        assert stats["count"] == pytest.approx(4.0)
        assert stats["max"] >= stats["min"]

    def test_compare_color_match_summaries(self):
        e1 = self._entries()[:2]
        e2 = self._entries()[2:]
        s1 = summarise_color_match_analysis(e1)
        s2 = summarise_color_match_analysis(e2)
        diff = compare_color_match_summaries(s1, s2)
        assert "mean_score_delta" in diff

    def test_batch_summarise_color_match_analysis(self):
        entries = self._entries()
        summaries = batch_summarise_color_match_analysis([entries[:2], entries[2:]])
        assert len(summaries) == 2
        assert summaries[0].n_entries == 2

    def test_make_edge_detection_entry(self):
        entry = make_edge_detection_entry(5, density=0.25, n_contours=10, method="canny")
        assert entry.fragment_id == 5
        assert entry.density == pytest.approx(0.25)
        assert entry.n_contours == 10
        assert entry.method == "canny"


# ═══════════════════════════════════════════════════════════════════════════════
# color_hist_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestColorHistUtils:

    def _entries(self):
        return [
            make_color_hist_entry(0, 1, 0.9, 0.85),
            make_color_hist_entry(1, 2, 0.5, 0.55),
            make_color_hist_entry(2, 3, 0.2, 0.15),
            make_color_hist_entry(3, 4, 0.7, 0.75),
        ]

    def test_config_defaults(self):
        cfg = ColorHistConfig()
        assert cfg.space == "hsv"
        assert cfg.good_threshold == pytest.approx(0.7)

    def test_config_invalid_min_score(self):
        with pytest.raises(ValueError):
            ColorHistConfig(min_score=-0.1)

    def test_config_invalid_max_score(self):
        with pytest.raises(ValueError):
            ColorHistConfig(min_score=0.8, max_score=0.5)

    def test_make_color_hist_entry(self):
        entry = make_color_hist_entry(0, 1, 0.8, 0.6)
        assert entry.frag_i == 0
        assert entry.frag_j == 1
        assert entry.intersection == pytest.approx(0.8)
        assert entry.score == pytest.approx((0.8 + 0.6) / 2.0)

    def test_entries_from_comparisons(self):
        pairs = [(0, 1), (1, 2), (2, 3)]
        inters = [0.9, 0.5, 0.2]
        chi2s = [0.85, 0.55, 0.25]
        entries = entries_from_comparisons(pairs, inters, chi2s)
        assert len(entries) == 3
        assert entries[0].frag_i == 0

    def test_entries_from_comparisons_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            entries_from_comparisons([(0, 1)], [0.5], [0.5, 0.6])

    def test_summarise_color_hist_empty(self):
        summary = summarise_color_hist([])
        assert summary.n_entries == 0
        assert summary.mean_score == 0.0

    def test_summarise_color_hist_populated(self):
        entries = self._entries()
        summary = summarise_color_hist(entries)
        assert summary.n_entries == 4
        assert summary.min_score <= summary.mean_score <= summary.max_score

    def test_filter_good_hist_entries(self):
        entries = self._entries()
        good = filter_good_hist_entries(entries, threshold=0.7)
        assert all(e.score >= 0.7 for e in good)

    def test_filter_poor_hist_entries(self):
        entries = self._entries()
        poor = filter_poor_hist_entries(entries, threshold=0.3)
        assert all(e.score < 0.3 for e in poor)

    def test_filter_by_intersection_range(self):
        entries = self._entries()
        filtered = filter_by_intersection_range(entries, lo=0.4, hi=0.8)
        assert all(0.4 <= e.intersection <= 0.8 for e in filtered)

    def test_filter_by_chi2_range(self):
        entries = self._entries()
        filtered = filter_by_chi2_range(entries, lo=0.5, hi=1.0)
        assert all(0.5 <= e.chi2 <= 1.0 for e in filtered)

    def test_filter_by_space(self):
        entries = [
            make_color_hist_entry(0, 1, 0.5, 0.5, space="hsv"),
            make_color_hist_entry(1, 2, 0.5, 0.5, space="rgb"),
        ]
        hsv_entries = filter_by_space(entries, "hsv")
        assert len(hsv_entries) == 1
        assert hsv_entries[0].space == "hsv"

    def test_top_k_hist_entries(self):
        entries = self._entries()
        top2 = top_k_hist_entries(entries, k=2)
        assert len(top2) == 2
        assert top2[0].score >= top2[1].score

    def test_best_hist_entry(self):
        entries = self._entries()
        best = best_hist_entry(entries)
        assert best is not None
        assert best.score == max(e.score for e in entries)
