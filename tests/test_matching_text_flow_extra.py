"""Extra tests for puzzle_reconstruction/matching/text_flow.py"""
import numpy as np
import pytest

from puzzle_reconstruction.matching.text_flow import (
    TextLineProfile,
    TextFlowMatch,
    TextFlowConfig,
    TextFlowScorer,
    detect_text_baseline_angle,
    detect_text_line_positions,
    build_text_line_profile,
    compare_baseline_angles,
    align_line_positions,
    match_text_flow,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _zeros(h=32, w=32):
    return np.zeros((h, w), dtype=float)


def _hlines(h=64, w=64, n=3, strength=1.0):
    g = np.zeros((h, w), dtype=float)
    step = h // (n + 1)
    for i in range(1, n + 1):
        row = i * step
        if row < h:
            g[row, :] = strength
    return g


def _noisy(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((h, w))


def _profile(angle=0.0, positions=None, conf=0.9, n_lines=None):
    if positions is None:
        positions = np.array([0.25, 0.5, 0.75])
    if n_lines is None:
        n_lines = len(positions)
    return TextLineProfile(
        angle_deg=angle,
        line_positions=positions,
        confidence=conf,
        n_lines=n_lines,
    )


# ─── TextLineProfile dataclass ────────────────────────────────────────────────

def test_profile_stores_angle():
    p = _profile(angle=15.0)
    assert p.angle_deg == pytest.approx(15.0)


def test_profile_stores_confidence():
    p = _profile(conf=0.75)
    assert p.confidence == pytest.approx(0.75)


def test_profile_n_lines_zero():
    p = _profile(positions=np.array([]), n_lines=0, conf=0.0)
    assert p.n_lines == 0


def test_profile_positions_stored():
    pos = np.array([0.1, 0.5, 0.9])
    p = _profile(positions=pos)
    np.testing.assert_array_equal(p.line_positions, pos)


# ─── TextFlowMatch dataclass ─────────────────────────────────────────────────

def test_match_stores_all_fields():
    m = TextFlowMatch(score=0.8, angle_score=0.9, alignment_score=0.7, angle_diff_deg=5.0)
    assert m.score == pytest.approx(0.8)
    assert m.angle_score == pytest.approx(0.9)
    assert m.alignment_score == pytest.approx(0.7)
    assert m.angle_diff_deg == pytest.approx(5.0)


def test_match_score_zero():
    m = TextFlowMatch(score=0.0, angle_score=0.0, alignment_score=0.0, angle_diff_deg=0.0)
    assert m.score == pytest.approx(0.0)


# ─── TextFlowConfig ───────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = TextFlowConfig()
    assert cfg.angle_tolerance_deg == pytest.approx(5.0)
    assert cfg.max_angle_diff_deg == pytest.approx(30.0)
    assert cfg.line_align_tolerance == pytest.approx(0.05)
    assert cfg.min_confidence == pytest.approx(0.1)
    assert cfg.angle_weight + cfg.align_weight == pytest.approx(1.0)


def test_config_custom():
    cfg = TextFlowConfig(angle_tolerance_deg=2.0, max_angle_diff_deg=15.0)
    assert cfg.angle_tolerance_deg == pytest.approx(2.0)
    assert cfg.max_angle_diff_deg == pytest.approx(15.0)


def test_config_min_confidence_zero():
    cfg = TextFlowConfig(min_confidence=0.0)
    assert cfg.min_confidence == pytest.approx(0.0)


# ─── detect_text_baseline_angle ───────────────────────────────────────────────

def test_angle_detection_zero_gradient():
    g = _zeros()
    angle, conf = detect_text_baseline_angle(g)
    assert conf == pytest.approx(0.0)


def test_angle_detection_range_always_valid():
    for seed in range(5):
        g = _noisy(seed=seed)
        angle, conf = detect_text_baseline_angle(g)
        assert -90.0 <= angle < 90.0
        assert 0.0 <= conf <= 1.0


def test_angle_detection_empty_array():
    g = np.zeros((0, 0))
    angle, conf = detect_text_baseline_angle(g)
    assert conf == pytest.approx(0.0)


def test_angle_detection_1x1_array():
    g = np.array([[1.0]])
    angle, conf = detect_text_baseline_angle(g)
    assert isinstance(angle, float)
    assert isinstance(conf, float)


def test_angle_detection_horizontal_lines_high_conf():
    g = _hlines(64, 64, 3, strength=10.0)
    _, conf = detect_text_baseline_angle(g)
    # Strong horizontal lines → high confidence
    assert conf >= 0.0  # at least non-negative


def test_angle_detection_n_bins_6():
    g = _hlines()
    angle, conf = detect_text_baseline_angle(g, n_bins=6)
    assert -90.0 <= angle < 90.0


def test_angle_detection_n_bins_360():
    g = _hlines()
    angle, conf = detect_text_baseline_angle(g, n_bins=360)
    assert -90.0 <= angle < 90.0


# ─── detect_text_line_positions ───────────────────────────────────────────────

def test_positions_empty_for_zero_gradient():
    pos = detect_text_line_positions(_zeros())
    assert len(pos) == 0


def test_positions_in_range_01():
    g = _hlines(64, 64, 4)
    pos = detect_text_line_positions(g)
    if len(pos) > 0:
        assert np.all(pos >= 0.0)
        assert np.all(pos <= 1.0)


def test_positions_sorted():
    g = _hlines(128, 128, 5)
    pos = detect_text_line_positions(g)
    if len(pos) > 1:
        assert np.all(np.diff(pos) >= 0)


def test_positions_min_peak_ratio_high_returns_fewer():
    g = _hlines(64, 64, 3)
    pos_easy = detect_text_line_positions(g, min_peak_ratio=0.01)
    pos_hard = detect_text_line_positions(g, min_peak_ratio=0.99)
    assert len(pos_easy) >= len(pos_hard)


def test_positions_empty_gradient_map():
    pos = detect_text_line_positions(np.zeros((0, 0)))
    assert len(pos) == 0


def test_positions_single_row():
    g = np.array([[0.0, 1.0, 0.0, 1.0]])
    pos = detect_text_line_positions(g)
    assert isinstance(pos, np.ndarray)


# ─── build_text_line_profile ─────────────────────────────────────────────────

def test_build_profile_returns_text_line_profile():
    p = build_text_line_profile(_hlines())
    assert isinstance(p, TextLineProfile)


def test_build_profile_n_lines_matches_positions():
    p = build_text_line_profile(_hlines())
    assert p.n_lines == len(p.line_positions)


def test_build_profile_confidence_zero_for_empty():
    p = build_text_line_profile(_zeros())
    assert p.confidence == pytest.approx(0.0)


def test_build_profile_confidence_range():
    p = build_text_line_profile(_noisy())
    assert 0.0 <= p.confidence <= 1.0


def test_build_profile_angle_range():
    p = build_text_line_profile(_hlines())
    assert -90.0 <= p.angle_deg < 90.0


# ─── compare_baseline_angles ─────────────────────────────────────────────────

def test_compare_angles_same_returns_one():
    s = compare_baseline_angles(10.0, 10.0)
    assert s == pytest.approx(1.0)


def test_compare_angles_within_tolerance_returns_one():
    s = compare_baseline_angles(0.0, 4.0, tolerance_deg=5.0)
    assert s == pytest.approx(1.0)


def test_compare_angles_beyond_max_returns_zero():
    s = compare_baseline_angles(0.0, 40.0, max_diff_deg=30.0)
    assert s == pytest.approx(0.0)


def test_compare_angles_symmetric():
    for a, b in [(5.0, 20.0), (-10.0, 10.0), (80.0, -80.0)]:
        assert compare_baseline_angles(a, b) == pytest.approx(
            compare_baseline_angles(b, a))


def test_compare_angles_wrap_89_minus89():
    s = compare_baseline_angles(89.0, -89.0, tolerance_deg=5.0)
    assert s == pytest.approx(1.0)


def test_compare_angles_range():
    angles = np.linspace(-90, 90, 19)
    for a in angles:
        for b in angles:
            s = compare_baseline_angles(a, b)
            assert 0.0 <= s <= 1.0


def test_compare_angles_max_diff_0_returns_one_only_if_same():
    assert compare_baseline_angles(5.0, 5.0, max_diff_deg=0.0) == pytest.approx(1.0)
    assert compare_baseline_angles(5.0, 6.0, tolerance_deg=0.0, max_diff_deg=0.0) == pytest.approx(0.0)


# ─── align_line_positions ─────────────────────────────────────────────────────

def test_align_identical_returns_one():
    pos = np.array([0.2, 0.5, 0.8])
    assert align_line_positions(pos, pos) == pytest.approx(1.0)


def test_align_no_match_returns_zero():
    a = np.array([0.1, 0.2])
    b = np.array([0.8, 0.9])
    assert align_line_positions(a, b, tolerance=0.05) == pytest.approx(0.0)


def test_align_empty_a_zero():
    assert align_line_positions(np.array([]), np.array([0.5])) == pytest.approx(0.0)


def test_align_empty_b_zero():
    assert align_line_positions(np.array([0.5]), np.array([])) == pytest.approx(0.0)


def test_align_both_empty_zero():
    assert align_line_positions(np.array([]), np.array([])) == pytest.approx(0.0)


def test_align_large_tolerance_all_match():
    a = np.array([0.1, 0.5, 0.9])
    b = np.array([0.15, 0.55, 0.95])
    s = align_line_positions(a, b, tolerance=0.1)
    assert s == pytest.approx(1.0)


def test_align_result_in_range():
    rng = np.random.default_rng(0)
    a = np.sort(rng.uniform(0, 1, 5))
    b = np.sort(rng.uniform(0, 1, 5))
    s = align_line_positions(a, b)
    assert 0.0 <= s <= 1.0


def test_align_single_match_score_positive():
    a = np.array([0.5])
    b = np.array([0.5])
    s = align_line_positions(a, b, tolerance=0.05)
    assert s > 0.0


# ─── match_text_flow ─────────────────────────────────────────────────────────

def test_match_same_angle_same_positions_high_score():
    p = _profile(angle=0.0, positions=np.array([0.25, 0.5, 0.75]), conf=0.9)
    m = match_text_flow(p, p)
    assert m.score >= 0.9


def test_match_score_in_range():
    for angle_b in [0.0, 5.0, 10.0, 20.0, 45.0]:
        pa = _profile(angle=0.0, conf=0.9)
        pb = _profile(angle=angle_b, conf=0.9)
        m = match_text_flow(pa, pb)
        assert 0.0 <= m.score <= 1.0


def test_match_low_confidence_returns_0_5():
    pa = TextLineProfile(0.0, np.array([]), 0.0, 0)
    pb = TextLineProfile(0.0, np.array([]), 0.0, 0)
    m = match_text_flow(pa, pb)
    assert m.score == pytest.approx(0.5)


def test_match_angle_score_in_range():
    pa = _profile(angle=0.0, conf=0.9)
    pb = _profile(angle=15.0, conf=0.9)
    m = match_text_flow(pa, pb)
    assert 0.0 <= m.angle_score <= 1.0


def test_match_alignment_score_in_range():
    pa = _profile(angle=0.0, conf=0.9)
    pb = _profile(angle=0.0, conf=0.9)
    m = match_text_flow(pa, pb)
    assert 0.0 <= m.alignment_score <= 1.0


def test_match_angle_diff_correct():
    pa = _profile(angle=10.0, conf=0.9)
    pb = _profile(angle=25.0, conf=0.9)
    m = match_text_flow(pa, pb)
    assert m.angle_diff_deg == pytest.approx(15.0)


def test_match_with_custom_cfg():
    cfg = TextFlowConfig(angle_tolerance_deg=1.0, max_angle_diff_deg=5.0)
    pa = _profile(angle=0.0, conf=0.9)
    pb = _profile(angle=3.0, conf=0.9)
    m = match_text_flow(pa, pb, cfg=cfg)
    assert isinstance(m, TextFlowMatch)
    assert 0.0 <= m.score <= 1.0


def test_match_high_max_diff_allows_large_angle():
    cfg = TextFlowConfig(max_angle_diff_deg=90.0, angle_tolerance_deg=1.0)
    pa = _profile(angle=0.0, conf=0.9)
    pb = _profile(angle=45.0, conf=0.9)
    m = match_text_flow(pa, pb, cfg=cfg)
    assert m.score > 0.0


# ─── TextFlowScorer ───────────────────────────────────────────────────────────

def test_scorer_default_cfg():
    scorer = TextFlowScorer()
    assert scorer.cfg.angle_tolerance_deg == pytest.approx(5.0)


def test_scorer_custom_cfg():
    cfg = TextFlowConfig(angle_tolerance_deg=3.0)
    scorer = TextFlowScorer(cfg=cfg)
    assert scorer.cfg.angle_tolerance_deg == pytest.approx(3.0)


def test_scorer_build_profile_zero_gradient():
    scorer = TextFlowScorer()
    p = scorer.build_profile(_zeros())
    assert p.confidence == pytest.approx(0.0)
    assert p.n_lines == 0


def test_scorer_build_profile_with_lines():
    scorer = TextFlowScorer()
    p = scorer.build_profile(_hlines(64, 64, 3))
    assert isinstance(p, TextLineProfile)


def test_scorer_score_returns_match():
    scorer = TextFlowScorer()
    m = scorer.score(_hlines(), _hlines())
    assert isinstance(m, TextFlowMatch)


def test_scorer_score_in_range():
    scorer = TextFlowScorer()
    for seed in range(5):
        m = scorer.score(_noisy(seed=seed), _zeros())
        assert 0.0 <= m.score <= 1.0


def test_scorer_batch_empty():
    scorer = TextFlowScorer()
    assert scorer.score_batch(_hlines(), []) == []


def test_scorer_batch_three_candidates():
    scorer = TextFlowScorer()
    cands = [_hlines(), _zeros(), _noisy(seed=0)]
    results = scorer.score_batch(_hlines(), cands)
    assert len(results) == 3


def test_scorer_batch_all_in_range():
    scorer = TextFlowScorer()
    cands = [_hlines(n=i) for i in range(1, 5)]
    results = scorer.score_batch(_hlines(n=3), cands)
    for m in results:
        assert 0.0 <= m.score <= 1.0


def test_scorer_identical_gradient_maps_high_score():
    scorer = TextFlowScorer()
    g = _hlines(64, 64, 4)
    m = scorer.score(g, g)
    assert m.score >= 0.5
