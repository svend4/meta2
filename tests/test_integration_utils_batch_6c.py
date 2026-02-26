"""Integration tests for utils batch 6c.

Modules tested:
  1. puzzle_reconstruction.utils.orient_skew_utils
  2. puzzle_reconstruction.verification.homography_verifier
  3. puzzle_reconstruction.utils.window_tile_records
  4. puzzle_reconstruction.utils.orient_topology_utils
"""
import pytest
import numpy as np

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from puzzle_reconstruction.utils.orient_skew_utils import (
    OrientMatchConfig,
    OrientMatchEntry,
    OrientMatchSummary,
    make_orient_match_entry,
    summarise_orient_match_entries,
    filter_high_orient_matches,
    filter_low_orient_matches,
    filter_orient_by_score_range,
    filter_orient_by_max_angle,
    top_k_orient_match_entries,
    best_orient_match_entry,
    orient_match_stats,
    compare_orient_summaries,
)
from puzzle_reconstruction.verification.homography_verifier import (
    HomographyConfig,
    HomographyResult,
    HomographyVerifier,
    estimate_homography_dlt,
    reprojection_error,
    estimate_homography_ransac,
    check_homography_quality,
)
from puzzle_reconstruction.utils.window_tile_records import (
    WindowOpRecord,
    WindowFunctionRecord,
    TileOpRecord,
    TileFilterRecord,
    OverlapSummaryRecord,
    ScoreSummaryRecord,
    make_window_op_record,
    make_tile_op_record,
)
from puzzle_reconstruction.utils.orient_topology_utils import (
    OrientMatchRecord,
    OrientMatchSummary as TopOrientMatchSummary,
    TopologyRecord,
    TopologySummary,
    summarize_orient_matches,
    filter_orient_records,
    topology_records_from_dicts,
    summarize_topology,
    rank_orient_matches,
    top_k_orient_matches,
)


# ===========================================================================
# orient_skew_utils tests
# ===========================================================================

def _make_entries(n=5):
    scores = rng.uniform(0.0, 1.0, n).tolist()
    angles = rng.uniform(0.0, 180.0, n).tolist()
    return [
        make_orient_match_entry(i, i + 1, angles[i], scores[i], 36)
        for i in range(n)
    ]


def test_orient_make_entry_fields():
    e = make_orient_match_entry(0, 1, 45.0, 0.8, 36)
    assert e.fragment_a == 0
    assert e.fragment_b == 1
    assert e.best_angle == 45.0
    assert e.best_score == 0.8
    assert e.n_angles_tested == 36


def test_orient_summarise_empty():
    s = summarise_orient_match_entries([])
    assert s.n_entries == 0
    assert s.mean_score == 0.0
    assert s.high_score_count == 0


def test_orient_summarise_basic():
    entries = _make_entries(6)
    s = summarise_orient_match_entries(entries)
    assert s.n_entries == 6
    assert 0.0 <= s.mean_score <= 1.0
    assert s.min_score <= s.mean_score <= s.max_score


def test_orient_filter_high():
    entries = _make_entries(10)
    high = filter_high_orient_matches(entries, threshold=0.7)
    assert all(e.best_score >= 0.7 for e in high)


def test_orient_filter_low():
    entries = _make_entries(10)
    low = filter_low_orient_matches(entries, threshold=0.7)
    assert all(e.best_score < 0.7 for e in low)


def test_orient_filter_score_range():
    entries = _make_entries(10)
    filtered = filter_orient_by_score_range(entries, lo=0.3, hi=0.8)
    assert all(0.3 <= e.best_score <= 0.8 for e in filtered)


def test_orient_filter_by_max_angle():
    entries = _make_entries(10)
    filtered = filter_orient_by_max_angle(entries, max_angle=90.0)
    assert all(e.best_angle <= 90.0 for e in filtered)


def test_orient_top_k():
    entries = _make_entries(10)
    top3 = top_k_orient_match_entries(entries, k=3)
    assert len(top3) == 3
    scores = [e.best_score for e in top3]
    assert scores == sorted(scores, reverse=True)


def test_orient_best_entry():
    entries = _make_entries(8)
    best = best_orient_match_entry(entries)
    assert best is not None
    assert best.best_score == max(e.best_score for e in entries)


def test_orient_best_entry_empty():
    assert best_orient_match_entry([]) is None


def test_orient_match_stats_keys():
    entries = _make_entries(5)
    stats = orient_match_stats(entries)
    assert set(stats.keys()) == {"count", "mean", "std", "min", "max"}
    assert stats["count"] == 5.0


def test_orient_match_stats_empty():
    stats = orient_match_stats([])
    assert stats["count"] == 0
    assert stats["mean"] == 0.0


def test_orient_compare_summaries():
    e1 = _make_entries(5)
    e2 = _make_entries(5)
    s1 = summarise_orient_match_entries(e1)
    s2 = summarise_orient_match_entries(e2)
    diff = compare_orient_summaries(s1, s2)
    assert "mean_score_delta" in diff
    assert abs(diff["mean_score_delta"] - (s1.mean_score - s2.mean_score)) < 1e-9


def test_orient_config_defaults():
    cfg = OrientMatchConfig()
    assert cfg.min_score == 0.0
    assert cfg.max_angle == 180.0


# ===========================================================================
# homography_verifier tests
# ===========================================================================

def _identity_points(n=20):
    pts = rng.uniform(10, 246, (n, 2)).astype(float)
    return pts, pts.copy()


def _translation_points(n=20, tx=5.0, ty=3.0):
    src = rng.uniform(10, 200, (n, 2)).astype(float)
    dst = src + np.array([tx, ty])
    return src, dst


def test_dlt_identity_transform():
    src, dst = _identity_points(n=10)
    H = estimate_homography_dlt(src, dst)
    assert H is not None
    assert H.shape == (3, 3)
    errs = reprojection_error(H, src, dst)
    assert np.mean(errs) < 1.0


def test_dlt_translation():
    src, dst = _translation_points(n=10, tx=10.0, ty=7.0)
    H = estimate_homography_dlt(src, dst)
    assert H is not None
    errs = reprojection_error(H, src, dst)
    assert np.mean(errs) < 1.0


def test_dlt_too_few_points():
    src = rng.uniform(0, 100, (3, 2))
    dst = rng.uniform(0, 100, (3, 2))
    assert estimate_homography_dlt(src, dst) is None


def test_reprojection_error_shape():
    src, dst = _translation_points(n=15)
    H = estimate_homography_dlt(src, dst)
    errs = reprojection_error(H, src, dst)
    assert errs.shape == (15,)
    assert np.all(errs >= 0)


def test_ransac_translation():
    src, dst = _translation_points(n=30, tx=8.0, ty=-4.0)
    H, mask = estimate_homography_ransac(src, dst)
    assert H is not None
    assert mask.dtype == bool
    assert mask.sum() >= 4


def test_ransac_too_few_points():
    src = rng.uniform(0, 100, (3, 2))
    dst = rng.uniform(0, 100, (3, 2))
    H, mask = estimate_homography_ransac(src, dst)
    assert H is None
    assert mask.sum() == 0


def test_check_homography_quality_identity():
    H = np.eye(3)
    assert check_homography_quality(H, (256, 256)) is True


def test_check_homography_quality_flip():
    H = np.diag([-1.0, 1.0, 1.0])
    assert check_homography_quality(H, (256, 256)) is False


def test_verifier_translation():
    src, dst = _translation_points(n=25, tx=5.0, ty=2.0)
    verifier = HomographyVerifier()
    result = verifier.verify(src, dst, fragment_size=(256, 256))
    assert isinstance(result, HomographyResult)
    assert result.H is not None
    assert result.inlier_ratio >= 0.5


def test_verifier_too_few():
    src = rng.uniform(0, 100, (2, 2))
    dst = rng.uniform(0, 100, (2, 2))
    verifier = HomographyVerifier()
    result = verifier.verify(src, dst)
    assert result.H is None
    assert result.is_valid is False


def test_verifier_score_range():
    src, dst = _translation_points(n=20)
    verifier = HomographyVerifier()
    result = verifier.verify(src, dst)
    assert 0.0 <= result.score <= 1.0


def test_homography_config_defaults():
    cfg = HomographyConfig()
    assert cfg.ransac_threshold == 5.0
    assert cfg.min_inliers == 4
    assert cfg.min_inlier_ratio == 0.5


# ===========================================================================
# window_tile_records tests
# ===========================================================================

def test_window_op_record_valid():
    rec = WindowOpRecord("mean", 100, 10, 5, 18)
    assert rec.operation == "mean"
    assert rec.n_windows == 18


def test_window_op_record_invalid_op():
    with pytest.raises(ValueError):
        WindowOpRecord("invalid_op", 100, 10, 5, 18)


def test_window_op_coverage():
    rec = WindowOpRecord("std", 100, 10, 10, 10)
    assert rec.coverage == 1.0


def test_window_op_has_overlap():
    overlap = WindowOpRecord("max", 100, 10, 5, 18)
    no_overlap = WindowOpRecord("max", 100, 10, 10, 9)
    assert overlap.has_overlap is True
    assert no_overlap.has_overlap is False


def test_window_function_record_valid():
    rec = WindowFunctionRecord("hann", 64, sum_before=100.0, sum_after=80.0)
    assert rec.attenuation_ratio == pytest.approx(0.8)


def test_window_function_record_invalid():
    with pytest.raises(ValueError):
        WindowFunctionRecord("unknown_func", 64)


def test_tile_op_record_area():
    rec = TileOpRecord("tile", (256, 256), 32, 32, 64)
    assert rec.tile_area == 32 * 32
    assert rec.image_area == 256 * 256


def test_tile_op_coverage():
    rec = TileOpRecord("tile", (100, 100), 10, 10, 100)
    assert rec.coverage_ratio == 1.0


def test_tile_filter_record_retention():
    rec = TileFilterRecord(n_input=100, n_kept=75, min_foreground=0.1)
    assert rec.n_removed == 25
    assert rec.retention_rate == pytest.approx(0.75)


def test_tile_filter_record_invalid():
    with pytest.raises(ValueError):
        TileFilterRecord(n_input=50, n_kept=60, min_foreground=0.1)


def test_overlap_summary_valid():
    rec = OverlapSummaryRecord(10, 45, 0)
    assert rec.is_valid is True
    assert rec.overlap_rate == 0.0


def test_score_summary_record_status():
    rec = ScoreSummaryRecord(n_metrics=5, total_score=0.8, passed=True,
                             pass_threshold=0.6)
    assert rec.status == "pass"
    assert rec.margin == pytest.approx(0.2)


def test_make_window_op_record():
    rec = make_window_op_record("mean", 200, 20, 10, 18, label="test")
    assert isinstance(rec, WindowOpRecord)
    assert rec.label == "test"


def test_make_tile_op_record():
    rec = make_tile_op_record((128, 128), 16, 16, 64)
    assert isinstance(rec, TileOpRecord)
    assert rec.tile_area == 256


# ===========================================================================
# orient_topology_utils tests
# ===========================================================================

def _make_orient_records(n=5):
    records = []
    vals = rng.uniform(0.0, 1.0, n).tolist()
    angles = rng.uniform(0.0, 360.0, n).tolist()
    for i in range(n):
        records.append(OrientMatchRecord(
            fragment_a=i, fragment_b=i + 1,
            best_score=vals[i], best_angle=angles[i],
            is_flipped=(i % 2 == 0), n_angles_tested=36,
        ))
    return records


def test_orient_record_pair():
    r = OrientMatchRecord(0, 1, 0.8, 45.0)
    assert r.pair == (0, 1)


def test_orient_record_is_good_match():
    good = OrientMatchRecord(0, 1, 0.7, 0.0)
    bad = OrientMatchRecord(0, 1, 0.3, 0.0)
    assert good.is_good_match is True
    assert bad.is_good_match is False


def test_orient_record_invalid_score():
    with pytest.raises(ValueError):
        OrientMatchRecord(0, 1, best_score=1.5, best_angle=0.0)


def test_summarize_orient_matches_empty():
    s = summarize_orient_matches([])
    assert s.total_pairs == 0
    assert s.good_ratio == 0.0


def test_summarize_orient_matches_basic():
    records = _make_orient_records(6)
    s = summarize_orient_matches(records)
    assert s.total_pairs == 6
    assert 0.0 <= s.mean_score <= 1.0
    assert s.good_pairs <= s.total_pairs


def test_summarize_orient_good_ratio():
    records = [OrientMatchRecord(i, i+1, 0.9, 0.0) for i in range(4)]
    s = summarize_orient_matches(records)
    assert s.good_ratio == 1.0


def test_filter_orient_records_min_score():
    records = _make_orient_records(10)
    filtered = filter_orient_records(records, min_score=0.5)
    assert all(r.best_score >= 0.5 for r in filtered)


def test_filter_orient_records_exclude_flipped():
    records = _make_orient_records(10)
    filtered = filter_orient_records(records, exclude_flipped=True)
    assert all(not r.is_flipped for r in filtered)


def test_topology_record_is_convex():
    convex = TopologyRecord(solidity=0.95, extent=0.8, convexity=0.95,
                            compactness=0.7, complexity=0.1)
    not_convex = TopologyRecord(solidity=0.8, extent=0.7, convexity=0.5,
                                compactness=0.5, complexity=0.2)
    assert convex.is_convex is True
    assert not_convex.is_convex is False


def test_topology_record_to_dict():
    rec = TopologyRecord(solidity=0.9, extent=0.8, convexity=0.95,
                         compactness=0.85, complexity=0.1)
    d = rec.to_dict()
    assert set(d.keys()) == {"solidity", "extent", "convexity", "compactness", "complexity"}


def test_topology_records_from_dicts():
    dicts = [{"solidity": 0.9, "extent": 0.8, "convexity": 0.95,
               "compactness": 0.85, "complexity": 0.1}]
    records = topology_records_from_dicts(dicts)
    assert len(records) == 1
    assert records[0].solidity == pytest.approx(0.9)


def test_summarize_topology_empty():
    s = summarize_topology([])
    assert s.n_contours == 0
    assert s.convex_ratio == 0.0


def test_summarize_topology_basic():
    records = [TopologyRecord(solidity=0.9, extent=0.8, convexity=0.95,
                               compactness=0.95, complexity=0.1) for _ in range(3)]
    s = summarize_topology(records)
    assert s.n_contours == 3
    assert s.n_convex == 3
    assert s.convex_ratio == 1.0


def test_rank_orient_matches():
    records = _make_orient_records(6)
    ranked = rank_orient_matches(records)
    scores = [r.best_score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_top_k_orient_matches():
    records = _make_orient_records(8)
    top2 = top_k_orient_matches(records, k=2)
    assert len(top2) == 2
    all_scores = [r.best_score for r in records]
    assert top2[0].best_score == max(all_scores)
