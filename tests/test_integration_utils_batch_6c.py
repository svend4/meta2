"""Integration tests for utils batch 6c.

Modules:
  1. puzzle_reconstruction.utils.orient_skew_utils
  2. puzzle_reconstruction.verification.homography_verifier
  3. puzzle_reconstruction.utils.window_tile_records
  4. puzzle_reconstruction.utils.orient_topology_utils
"""
import pytest
import numpy as np

rng = np.random.default_rng(42)

from puzzle_reconstruction.utils.orient_skew_utils import (
    OrientMatchConfig, make_orient_match_entry, summarise_orient_match_entries,
    filter_high_orient_matches, filter_low_orient_matches,
    filter_orient_by_score_range, filter_orient_by_max_angle,
    top_k_orient_match_entries, best_orient_match_entry,
    orient_match_stats, compare_orient_summaries,
)
from puzzle_reconstruction.verification.homography_verifier import (
    HomographyConfig, HomographyResult, HomographyVerifier,
    estimate_homography_dlt, reprojection_error,
    estimate_homography_ransac, check_homography_quality,
)
from puzzle_reconstruction.utils.window_tile_records import (
    WindowOpRecord, WindowFunctionRecord, TileOpRecord,
    TileFilterRecord, OverlapSummaryRecord, ScoreSummaryRecord,
    make_window_op_record, make_tile_op_record,
)
from puzzle_reconstruction.utils.orient_topology_utils import (
    OrientMatchRecord, TopologyRecord, TopologySummary,
    summarize_orient_matches, filter_orient_records,
    topology_records_from_dicts, summarize_topology,
    rank_orient_matches, top_k_orient_matches,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skew_entries(n=6):
    scores = rng.uniform(0.0, 1.0, n).tolist()
    angles = rng.uniform(0.0, 180.0, n).tolist()
    return [make_orient_match_entry(i, i+1, angles[i], scores[i], 36) for i in range(n)]


def _trans_pts(n=20, tx=5.0, ty=3.0):
    src = rng.uniform(10, 200, (n, 2)).astype(float)
    return src, src + np.array([tx, ty])


def _topo_records(n=4):
    return [OrientMatchRecord(i, i+1,
                              float(rng.uniform(0.0, 1.0)),
                              float(rng.uniform(0.0, 360.0)),
                              is_flipped=(i % 2 == 0)) for i in range(n)]


# ===========================================================================
# orient_skew_utils  (13 tests)
# ===========================================================================

def test_skew_make_entry_fields():
    e = make_orient_match_entry(0, 1, 45.0, 0.8, 36)
    assert (e.fragment_a, e.fragment_b, e.best_angle, e.best_score) == (0, 1, 45.0, 0.8)

def test_skew_summarise_empty():
    s = summarise_orient_match_entries([])
    assert s.n_entries == 0 and s.high_score_count == 0

def test_skew_summarise_counts():
    s = summarise_orient_match_entries(_skew_entries(6))
    assert s.n_entries == 6 and s.min_score <= s.mean_score <= s.max_score

def test_skew_filter_high():
    assert all(e.best_score >= 0.7 for e in filter_high_orient_matches(_skew_entries(10), 0.7))

def test_skew_filter_low():
    assert all(e.best_score < 0.7 for e in filter_low_orient_matches(_skew_entries(10), 0.7))

def test_skew_filter_score_range():
    filtered = filter_orient_by_score_range(_skew_entries(10), lo=0.3, hi=0.8)
    assert all(0.3 <= e.best_score <= 0.8 for e in filtered)

def test_skew_filter_by_max_angle():
    assert all(e.best_angle <= 90.0 for e in filter_orient_by_max_angle(_skew_entries(10), 90.0))

def test_skew_top_k_ordered():
    top3 = top_k_orient_match_entries(_skew_entries(10), k=3)
    scores = [e.best_score for e in top3]
    assert len(top3) == 3 and scores == sorted(scores, reverse=True)

def test_skew_best_entry():
    entries = _skew_entries(8)
    best = best_orient_match_entry(entries)
    assert best.best_score == max(e.best_score for e in entries)

def test_skew_best_entry_empty():
    assert best_orient_match_entry([]) is None

def test_skew_stats_keys():
    stats = orient_match_stats(_skew_entries(5))
    assert set(stats.keys()) == {"count", "mean", "std", "min", "max"}

def test_skew_stats_empty():
    assert orient_match_stats([])["count"] == 0

def test_skew_compare_summaries():
    s1 = summarise_orient_match_entries(_skew_entries(5))
    s2 = summarise_orient_match_entries(_skew_entries(5))
    d = compare_orient_summaries(s1, s2)
    assert abs(d["mean_score_delta"] - (s1.mean_score - s2.mean_score)) < 1e-9


# ===========================================================================
# homography_verifier  (13 tests)
# ===========================================================================

def test_hom_dlt_shape():
    src, dst = _trans_pts(n=10)
    H = estimate_homography_dlt(src, dst)
    assert H is not None and H.shape == (3, 3)

def test_hom_dlt_translation_err():
    src, dst = _trans_pts(n=10, tx=10.0, ty=7.0)
    H = estimate_homography_dlt(src, dst)
    assert np.mean(reprojection_error(H, src, dst)) < 1.0

def test_hom_dlt_too_few():
    assert estimate_homography_dlt(rng.uniform(0,100,(3,2)), rng.uniform(0,100,(3,2))) is None

def test_hom_reproj_shape():
    src, dst = _trans_pts(n=15)
    H = estimate_homography_dlt(src, dst)
    errs = reprojection_error(H, src, dst)
    assert errs.shape == (15,) and np.all(errs >= 0)

def test_hom_ransac_translation():
    src, dst = _trans_pts(n=30, tx=8.0, ty=-4.0)
    H, mask = estimate_homography_ransac(src, dst)
    assert H is not None and mask.sum() >= 4

def test_hom_ransac_too_few():
    H, mask = estimate_homography_ransac(rng.uniform(0,100,(3,2)), rng.uniform(0,100,(3,2)))
    assert H is None and mask.sum() == 0

def test_hom_quality_identity():
    assert check_homography_quality(np.eye(3), (256, 256)) is True

def test_hom_quality_flip():
    assert check_homography_quality(np.diag([-1.0, 1.0, 1.0]), (256, 256)) is False

def test_hom_verifier_translation():
    src, dst = _trans_pts(n=25, tx=5.0, ty=2.0)
    result = HomographyVerifier().verify(src, dst, fragment_size=(256, 256))
    assert isinstance(result, HomographyResult) and result.H is not None

def test_hom_verifier_too_few():
    result = HomographyVerifier().verify(rng.uniform(0,100,(2,2)), rng.uniform(0,100,(2,2)))
    assert result.H is None and result.is_valid is False

def test_hom_verifier_score_range():
    src, dst = _trans_pts(n=20)
    assert 0.0 <= HomographyVerifier().verify(src, dst).score <= 1.0

def test_hom_config_defaults():
    cfg = HomographyConfig()
    assert cfg.ransac_threshold == 5.0 and cfg.min_inliers == 4

def test_hom_verifier_inlier_ratio():
    src, dst = _trans_pts(n=30)
    result = HomographyVerifier().verify(src, dst)
    assert result.inlier_ratio >= 0.5


# ===========================================================================
# window_tile_records  (13 tests)
# ===========================================================================

def test_wtr_window_op_valid():
    rec = WindowOpRecord("mean", 100, 10, 5, 18)
    assert rec.operation == "mean" and rec.n_windows == 18

def test_wtr_window_op_invalid_op():
    with pytest.raises(ValueError):
        WindowOpRecord("bogus", 100, 10, 5, 18)

def test_wtr_window_op_coverage():
    assert WindowOpRecord("std", 100, 10, 10, 10).coverage == 1.0

def test_wtr_window_op_overlap():
    assert WindowOpRecord("max", 100, 10, 5, 18).has_overlap is True
    assert WindowOpRecord("max", 100, 10, 10, 9).has_overlap is False

def test_wtr_func_record_attenuation():
    rec = WindowFunctionRecord("hann", 64, sum_before=100.0, sum_after=80.0)
    assert rec.attenuation_ratio == pytest.approx(0.8)

def test_wtr_func_record_invalid():
    with pytest.raises(ValueError):
        WindowFunctionRecord("unknown_func", 64)

def test_wtr_tile_op_areas():
    rec = TileOpRecord("tile", (256, 256), 32, 32, 64)
    assert rec.tile_area == 1024 and rec.image_area == 65536

def test_wtr_tile_op_coverage():
    assert TileOpRecord("tile", (100, 100), 10, 10, 100).coverage_ratio == 1.0

def test_wtr_tile_filter_retention():
    rec = TileFilterRecord(n_input=100, n_kept=75, min_foreground=0.1)
    assert rec.n_removed == 25 and rec.retention_rate == pytest.approx(0.75)

def test_wtr_tile_filter_invalid():
    with pytest.raises(ValueError):
        TileFilterRecord(n_input=50, n_kept=60, min_foreground=0.1)

def test_wtr_overlap_summary_valid():
    rec = OverlapSummaryRecord(10, 45, 0)
    assert rec.is_valid is True and rec.overlap_rate == 0.0

def test_wtr_score_summary_status():
    rec = ScoreSummaryRecord(5, 0.8, True, 0.6)
    assert rec.status == "pass" and rec.margin == pytest.approx(0.2)

def test_wtr_make_helpers():
    wr = make_window_op_record("mean", 200, 20, 10, 18, label="x")
    assert isinstance(wr, WindowOpRecord) and wr.label == "x"
    tr = make_tile_op_record((128, 128), 16, 16, 64)
    assert isinstance(tr, TileOpRecord) and tr.tile_area == 256


# ===========================================================================
# orient_topology_utils  (13 tests)
# ===========================================================================

def test_top_record_pair():
    assert OrientMatchRecord(0, 1, 0.8, 45.0).pair == (0, 1)

def test_top_record_good_match():
    assert OrientMatchRecord(0, 1, 0.7, 0.0).is_good_match is True
    assert OrientMatchRecord(0, 1, 0.3, 0.0).is_good_match is False

def test_top_record_invalid_score():
    with pytest.raises(ValueError):
        OrientMatchRecord(0, 1, best_score=1.5, best_angle=0.0)

def test_top_summarize_empty():
    s = summarize_orient_matches([])
    assert s.total_pairs == 0 and s.good_ratio == 0.0

def test_top_summarize_basic():
    records = _topo_records(6)
    s = summarize_orient_matches(records)
    assert s.total_pairs == 6 and s.good_pairs <= s.total_pairs

def test_top_summarize_all_good():
    records = [OrientMatchRecord(i, i+1, 0.9, 0.0) for i in range(4)]
    assert summarize_orient_matches(records).good_ratio == 1.0

def test_top_filter_min_score():
    records = _topo_records(10)
    assert all(r.best_score >= 0.5 for r in filter_orient_records(records, min_score=0.5))

def test_top_filter_exclude_flipped():
    records = _topo_records(10)
    assert all(not r.is_flipped for r in filter_orient_records(records, exclude_flipped=True))

def test_top_topology_record_convex():
    assert TopologyRecord(solidity=0.9, extent=0.8, convexity=0.95, compactness=0.7, complexity=0.1).is_convex
    assert not TopologyRecord(solidity=0.8, extent=0.7, convexity=0.5, compactness=0.5, complexity=0.2).is_convex

def test_top_topology_record_to_dict():
    d = TopologyRecord(solidity=0.9, extent=0.8, convexity=0.95, compactness=0.85, complexity=0.1).to_dict()
    assert set(d.keys()) == {"solidity", "extent", "convexity", "compactness", "complexity"}

def test_top_records_from_dicts():
    records = topology_records_from_dicts([{"solidity": 0.9, "extent": 0.8,
                                             "convexity": 0.95, "compactness": 0.85, "complexity": 0.1}])
    assert len(records) == 1 and records[0].solidity == pytest.approx(0.9)

def test_top_summarize_topology():
    trs = [TopologyRecord(solidity=0.9, extent=0.8, convexity=0.95, compactness=0.95, complexity=0.1)
           for _ in range(3)]
    s = summarize_topology(trs)
    assert s.n_contours == 3 and s.convex_ratio == 1.0

def test_top_rank_and_top_k():
    records = _topo_records(8)
    ranked = rank_orient_matches(records)
    scores = [r.best_score for r in ranked]
    assert scores == sorted(scores, reverse=True)
    top2 = top_k_orient_matches(records, k=2)
    assert len(top2) == 2 and top2[0].best_score == max(scores)
