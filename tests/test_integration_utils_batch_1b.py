"""Integration tests for utils batch 1b.

Covers: blend_utils, candidate_rank_utils, canvas_build_utils,
        color_edge_export_utils, color_hist_utils.
"""
import pytest
import numpy as np

rng = np.random.default_rng(42)

from puzzle_reconstruction.utils.blend_utils import (
    BlendConfig, alpha_blend, weighted_blend, feather_mask, paste_with_mask,
)
from puzzle_reconstruction.utils.candidate_rank_utils import (
    CandidateRankConfig, CandidateRankEntry,
    make_candidate_entry, entries_from_pairs, summarise_rankings,
    filter_selected, filter_rejected_candidates, filter_by_score_range,
    filter_by_rank, top_k_candidate_entries, candidate_rank_stats,
)
from puzzle_reconstruction.utils.canvas_build_utils import (
    CanvasBuildConfig, PlacementEntry,
    make_placement_entry, entries_from_placements, summarise_canvas_build,
    filter_by_area, filter_by_coverage_contribution, top_k_by_coverage,
    canvas_build_stats, compare_canvas_summaries, batch_summarise_canvas_builds,
)
from puzzle_reconstruction.utils.color_edge_export_utils import (
    make_color_match_analysis_entry, summarise_color_match_analysis,
    filter_strong_color_matches, filter_weak_color_matches,
    filter_color_by_method, top_k_color_match_entries,
    best_color_match_entry, color_match_analysis_stats,
    compare_color_match_summaries, batch_summarise_color_match_analysis,
    make_edge_detection_entry,
)
from puzzle_reconstruction.utils.color_hist_utils import (
    ColorHistConfig, make_color_hist_entry, entries_from_comparisons,
    summarise_color_hist, filter_good_hist_entries, filter_poor_hist_entries,
    filter_by_intersection_range, filter_by_chi2_range, filter_by_space,
    top_k_hist_entries, best_hist_entry,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _img(h=16, w=16, c=3):
    return rng.integers(0, 256, (h, w, c), dtype=np.uint8)

def _rank_pairs():
    return [{"idx1": i, "idx2": i+1, "score": s}
            for i, s in enumerate([0.9, 0.4, 0.6, 0.2])]

def _placements():
    return [(0, 0, 0, 10, 10), (1, 15, 15, 20, 20), (2, 5, 5, 8, 8)]

def _cm_entries():
    return [make_color_match_analysis_entry(i, i+1, s, s, s, s, m)
            for i, s, m in [(0, 0.9, "hsv"), (1, 0.3, "rgb"),
                            (2, 0.6, "hsv"), (3, 0.1, "rgb")]]

def _hist_entries():
    return [make_color_hist_entry(i, i+1, a, b)
            for i, a, b in [(0, 0.9, 0.85), (1, 0.5, 0.55),
                            (2, 0.2, 0.15), (3, 0.7, 0.75)]]


# ═══════════════════════════════════════════════════════════════════════════════
# blend_utils (11 tests)
# ═══════════════════════════════════════════════════════════════════════════════

def test_blend_config_defaults():
    cfg = BlendConfig()
    assert cfg.feather_px == 8 and cfg.gamma == 1.0 and cfg.clip_output is True

def test_blend_config_invalid():
    with pytest.raises(ValueError, match="feather_px"): BlendConfig(feather_px=-1)
    with pytest.raises(ValueError, match="gamma"): BlendConfig(gamma=0.0)

def test_alpha_blend_midpoint():
    src = np.full((4, 4, 3), 200, dtype=np.uint8)
    dst = np.full((4, 4, 3), 100, dtype=np.uint8)
    result = alpha_blend(src, dst, alpha=0.5)
    assert result.dtype == np.uint8 and np.all(result == 150)

def test_alpha_blend_boundary_alphas():
    a = np.full((4, 4), 200, dtype=np.uint8)
    b = np.full((4, 4), 50, dtype=np.uint8)
    assert np.array_equal(alpha_blend(a, b, 0.0), b)
    assert np.array_equal(alpha_blend(a, b, 1.0), a)

def test_alpha_blend_raises_on_bad_inputs():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError): alpha_blend(img, np.zeros((4, 5, 3), dtype=np.uint8), 0.5)
    with pytest.raises(ValueError): alpha_blend(img, img, 1.5)

def test_weighted_blend_equal_weights():
    a = np.full((4, 4, 3), 100, dtype=np.uint8)
    b = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert np.all(weighted_blend([a, b]) == 150)

def test_weighted_blend_custom_weights():
    a, b = np.zeros((4, 4), dtype=np.uint8), np.full((4, 4), 200, dtype=np.uint8)
    assert np.all(weighted_blend([a, b], weights=[0.0, 1.0]) == 200)

def test_weighted_blend_empty_raises():
    with pytest.raises(ValueError): weighted_blend([])

def test_feather_mask_properties():
    mask = feather_mask(40, 40, feather_px=8)
    assert mask.shape == (40, 40) and mask.dtype == np.float32
    assert 0.0 <= mask.min() and mask.max() <= 1.0
    assert mask[20, 20] == pytest.approx(1.0)

def test_feather_mask_invalid():
    with pytest.raises(ValueError): feather_mask(0, 10)
    with pytest.raises(ValueError): feather_mask(10, -1)

def test_paste_with_mask_output():
    canvas = np.zeros((20, 20, 3), dtype=np.uint8)
    patch = np.full((10, 10, 3), 128, dtype=np.uint8)
    mask = np.ones((10, 10), dtype=np.float32)
    result = paste_with_mask(canvas, patch, mask, y=5, x=5)
    assert result.shape == canvas.shape and np.all(result[5:15, 5:15] == 128)


# ═══════════════════════════════════════════════════════════════════════════════
# candidate_rank_utils (11 tests)
# ═══════════════════════════════════════════════════════════════════════════════

def test_rank_config_defaults():
    cfg = CandidateRankConfig()
    assert cfg.min_score == 0.5 and cfg.max_pairs == 0 and cfg.deduplicate is True

def test_rank_config_invalid():
    with pytest.raises(ValueError, match="min_score"): CandidateRankConfig(min_score=1.5)
    with pytest.raises(ValueError, match="max_pairs"): CandidateRankConfig(max_pairs=-1)

def test_make_candidate_entry_selected_and_rejected():
    cfg = CandidateRankConfig(min_score=0.5)
    assert make_candidate_entry(0, 1, 0.8, 0, cfg=cfg).is_selected is True
    assert make_candidate_entry(0, 1, 0.3, 1, cfg=cfg).is_selected is False

def test_entries_from_pairs_sorted():
    entries = entries_from_pairs(_rank_pairs())
    scores = [e.score for e in entries]
    assert scores == sorted(scores, reverse=True)

def test_summarise_rankings_totals():
    cfg = CandidateRankConfig(min_score=0.5)
    entries = entries_from_pairs(_rank_pairs(), cfg=cfg)
    s = summarise_rankings(entries)
    assert s.n_total == 4 and s.n_selected + s.n_rejected == s.n_total

def test_filter_selected_and_rejected():
    cfg = CandidateRankConfig(min_score=0.5)
    entries = entries_from_pairs(_rank_pairs(), cfg=cfg)
    assert all(e.is_selected for e in filter_selected(entries))
    assert all(not e.is_selected for e in filter_rejected_candidates(entries))

def test_filter_by_score_range():
    entries = entries_from_pairs(_rank_pairs())
    filtered = filter_by_score_range(entries, 0.4, 0.7)
    assert all(0.4 <= e.score <= 0.7 for e in filtered)

def test_filter_by_rank():
    entries = entries_from_pairs(_rank_pairs())
    assert len(filter_by_rank(entries, max_rank=1)) == 2

def test_top_k_candidate_entries():
    entries = entries_from_pairs(_rank_pairs())
    top2 = top_k_candidate_entries(entries, k=2)
    assert len(top2) == 2 and top2[0].score >= top2[1].score

def test_candidate_rank_stats_empty():
    s = candidate_rank_stats([])
    assert s["count"] == 0 and s["mean"] == 0.0

def test_candidate_rank_stats_populated():
    s = candidate_rank_stats(entries_from_pairs(_rank_pairs()))
    assert s["count"] == 4 and s["max"] >= s["min"]


# ═══════════════════════════════════════════════════════════════════════════════
# canvas_build_utils (12 tests)
# ═══════════════════════════════════════════════════════════════════════════════

def test_canvas_config_defaults():
    cfg = CanvasBuildConfig()
    assert cfg.blend_mode == "overwrite" and cfg.max_fragments == 1000

def test_canvas_config_invalid():
    with pytest.raises(ValueError): CanvasBuildConfig(min_coverage=1.5)
    with pytest.raises(ValueError): CanvasBuildConfig(blend_mode="unknown")

def test_placement_entry_area_and_coords():
    e = make_placement_entry(1, x=5, y=10, w=15, h=25)
    assert e.area == 375 and e.x2 == 20 and e.y2 == 35

def test_placement_entry_invalid():
    with pytest.raises(ValueError): PlacementEntry(-1, 0, 0, 10, 10)
    with pytest.raises(ValueError): PlacementEntry(0, 0, 0, 0, 10)

def test_entries_from_placements():
    entries = entries_from_placements(_placements())
    assert len(entries) == 3 and entries[0].fragment_id == 0

def test_summarise_canvas_build():
    entries = entries_from_placements(_placements())
    s = summarise_canvas_build(entries, 100, 100, 0.5)
    assert s.n_placed == 3 and s.total_area == sum(e.area for e in entries)

def test_filter_by_area():
    entries = entries_from_placements(_placements())
    filtered = filter_by_area(entries, min_area=64, max_area=400)
    assert all(64 <= e.area <= 400 for e in filtered)

def test_filter_by_coverage_contribution():
    entries = [make_placement_entry(i, 0, 0, 10, 10, coverage_contribution=c)
               for i, c in enumerate([0.1, 0.5])]
    filtered = filter_by_coverage_contribution(entries, min_contrib=0.3)
    assert len(filtered) == 1 and filtered[0].fragment_id == 1

def test_top_k_by_coverage():
    entries = [make_placement_entry(i, 0, 0, 10, 10, coverage_contribution=i/5.0)
               for i in range(5)]
    top2 = top_k_by_coverage(entries, k=2)
    assert len(top2) == 2 and top2[0].coverage_contribution >= top2[1].coverage_contribution

def test_canvas_build_stats():
    s = canvas_build_stats(entries_from_placements(_placements()))
    assert s["n"] == 3 and s["total_area"] > 0

def test_compare_canvas_summaries():
    e1 = entries_from_placements(_placements())
    e2 = entries_from_placements([(0, 0, 0, 5, 5)])
    s1 = summarise_canvas_build(e1, 100, 100, 0.5)
    s2 = summarise_canvas_build(e2, 50, 50, 0.1)
    diff = compare_canvas_summaries(s1, s2)
    assert diff["n_placed_delta"] == 2 and diff["coverage_delta"] == pytest.approx(0.4)

def test_batch_summarise_canvas_builds():
    e = entries_from_placements(_placements())
    summaries = batch_summarise_canvas_builds([(e, 100, 100, 0.5), (e, 200, 200, 0.8)])
    assert len(summaries) == 2 and summaries[1].canvas_w == 200


# ═══════════════════════════════════════════════════════════════════════════════
# color_edge_export_utils (12 tests)
# ═══════════════════════════════════════════════════════════════════════════════

def test_make_color_match_entry():
    e = make_color_match_analysis_entry(0, 1, 0.75, 0.7, 0.8, 0.75)
    assert e.idx1 == 0 and e.score == pytest.approx(0.75) and e.method == "hsv"

def test_summarise_color_match_empty():
    s = summarise_color_match_analysis([])
    assert s.n_entries == 0 and s.mean_score == 0.0

def test_summarise_color_match_populated():
    s = summarise_color_match_analysis(_cm_entries())
    assert s.n_entries == 4 and s.min_score <= s.mean_score <= s.max_score

def test_filter_strong_color_matches():
    strong = filter_strong_color_matches(_cm_entries(), threshold=0.5)
    assert all(e.score >= 0.5 for e in strong) and len(strong) == 2

def test_filter_weak_color_matches():
    weak = filter_weak_color_matches(_cm_entries(), threshold=0.5)
    assert all(e.score < 0.5 for e in weak)

def test_filter_color_by_method():
    hsv = filter_color_by_method(_cm_entries(), "hsv")
    assert len(hsv) == 2 and all(e.method == "hsv" for e in hsv)

def test_top_k_color_match_entries():
    top2 = top_k_color_match_entries(_cm_entries(), k=2)
    assert len(top2) == 2 and top2[0].score >= top2[1].score

def test_best_color_match_entry():
    entries = _cm_entries()
    best = best_color_match_entry(entries)
    assert best is not None and best.score == max(e.score for e in entries)

def test_best_color_match_entry_empty():
    assert best_color_match_entry([]) is None

def test_color_match_analysis_stats():
    stats = color_match_analysis_stats(_cm_entries())
    assert stats["count"] == pytest.approx(4.0) and stats["max"] >= stats["min"]

def test_compare_color_match_summaries():
    entries = _cm_entries()
    diff = compare_color_match_summaries(
        summarise_color_match_analysis(entries[:2]),
        summarise_color_match_analysis(entries[2:]),
    )
    assert "mean_score_delta" in diff

def test_make_edge_detection_entry():
    e = make_edge_detection_entry(5, density=0.25, n_contours=10, method="canny")
    assert e.fragment_id == 5 and e.density == pytest.approx(0.25) and e.n_contours == 10


# ═══════════════════════════════════════════════════════════════════════════════
# color_hist_utils (12 tests)
# ═══════════════════════════════════════════════════════════════════════════════

def test_hist_config_defaults():
    cfg = ColorHistConfig()
    assert cfg.space == "hsv" and cfg.good_threshold == pytest.approx(0.7)

def test_hist_config_invalid():
    with pytest.raises(ValueError): ColorHistConfig(min_score=-0.1)
    with pytest.raises(ValueError): ColorHistConfig(min_score=0.8, max_score=0.5)

def test_make_color_hist_entry():
    e = make_color_hist_entry(0, 1, 0.8, 0.6)
    assert e.frag_i == 0 and e.score == pytest.approx(0.7)

def test_entries_from_comparisons():
    entries = entries_from_comparisons([(0,1),(1,2)], [0.9, 0.5], [0.85, 0.55])
    assert len(entries) == 2 and entries[0].frag_i == 0

def test_entries_from_comparisons_mismatch():
    with pytest.raises(ValueError, match="same length"):
        entries_from_comparisons([(0,1)], [0.5], [0.5, 0.6])

def test_summarise_color_hist_empty():
    s = summarise_color_hist([])
    assert s.n_entries == 0 and s.mean_score == 0.0

def test_summarise_color_hist_populated():
    s = summarise_color_hist(_hist_entries())
    assert s.n_entries == 4 and s.min_score <= s.mean_score <= s.max_score

def test_filter_good_and_poor_hist_entries():
    entries = _hist_entries()
    assert all(e.score >= 0.7 for e in filter_good_hist_entries(entries, 0.7))
    assert all(e.score < 0.3 for e in filter_poor_hist_entries(entries, 0.3))

def test_filter_by_intersection_range():
    filtered = filter_by_intersection_range(_hist_entries(), lo=0.4, hi=0.8)
    assert all(0.4 <= e.intersection <= 0.8 for e in filtered)

def test_filter_by_chi2_range():
    filtered = filter_by_chi2_range(_hist_entries(), lo=0.5, hi=1.0)
    assert all(0.5 <= e.chi2 <= 1.0 for e in filtered)

def test_filter_by_space():
    entries = [make_color_hist_entry(0, 1, 0.5, 0.5, space="hsv"),
               make_color_hist_entry(1, 2, 0.5, 0.5, space="rgb")]
    assert len(filter_by_space(entries, "hsv")) == 1

def test_top_k_and_best_hist_entry():
    entries = _hist_entries()
    top2 = top_k_hist_entries(entries, k=2)
    assert len(top2) == 2 and top2[0].score >= top2[1].score
    best = best_hist_entry(entries)
    assert best is not None and best.score == max(e.score for e in entries)
