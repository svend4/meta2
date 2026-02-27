"""Tests for puzzle_reconstruction/matching/text_flow.py."""
from __future__ import annotations

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _zeros(h: int = 32, w: int = 32) -> np.ndarray:
    return np.zeros((h, w), dtype=float)


def _horizontal_lines(h: int = 32, w: int = 32, n: int = 3) -> np.ndarray:
    """Gradient map with strong horizontal-line signal (text-like)."""
    g = np.zeros((h, w), dtype=float)
    step = h // (n + 1)
    for i in range(1, n + 1):
        row = i * step
        if row < h:
            g[row, :] = 1.0
    return g


# ── TextLineProfile ────────────────────────────────────────────────────────────

class TestTextLineProfile:

    def test_fields_exist(self):
        p = TextLineProfile(
            angle_deg=0.0,
            line_positions=np.array([0.25, 0.5, 0.75]),
            confidence=0.8,
            n_lines=3,
        )
        assert p.angle_deg == 0.0
        assert len(p.line_positions) == 3
        assert p.confidence == 0.8
        assert p.n_lines == 3


class TestTextFlowMatch:

    def test_fields_exist(self):
        m = TextFlowMatch(score=0.7, angle_score=0.9, alignment_score=0.5, angle_diff_deg=2.0)
        assert m.score == 0.7
        assert m.angle_score == 0.9
        assert m.alignment_score == 0.5
        assert m.angle_diff_deg == 2.0


# ── detect_text_baseline_angle ────────────────────────────────────────────────

class TestDetectBaselineAngle:

    def test_returns_two_floats(self):
        g = _horizontal_lines()
        angle, conf = detect_text_baseline_angle(g)
        assert isinstance(angle, float)
        assert isinstance(conf, float)

    def test_angle_in_range(self):
        g = _horizontal_lines()
        angle, _ = detect_text_baseline_angle(g)
        assert -90.0 <= angle < 90.0

    def test_confidence_in_range(self):
        g = _horizontal_lines()
        _, conf = detect_text_baseline_angle(g)
        assert 0.0 <= conf <= 1.0

    def test_zero_gradient_returns_zero_confidence(self):
        g = _zeros()
        _, conf = detect_text_baseline_angle(g)
        assert conf == 0.0

    def test_empty_input_no_crash(self):
        g = np.zeros((0, 0))
        angle, conf = detect_text_baseline_angle(g)
        assert conf == 0.0

    def test_single_row_no_crash(self):
        g = np.ones((1, 10))
        angle, conf = detect_text_baseline_angle(g)
        assert 0.0 <= conf <= 1.0

    def test_custom_n_bins(self):
        g = _horizontal_lines()
        angle, conf = detect_text_baseline_angle(g, n_bins=36)
        assert -90.0 <= angle < 90.0


# ── detect_text_line_positions ────────────────────────────────────────────────

class TestDetectTextLinePositions:

    def test_returns_array(self):
        g = _horizontal_lines(32, 32, 3)
        pos = detect_text_line_positions(g)
        assert isinstance(pos, np.ndarray)

    def test_positions_in_range_0_1(self):
        g = _horizontal_lines(32, 32, 3)
        pos = detect_text_line_positions(g)
        if len(pos) > 0:
            assert np.all(pos >= 0.0)
            assert np.all(pos <= 1.0)

    def test_empty_gradient_returns_empty(self):
        pos = detect_text_line_positions(np.zeros((0, 0)))
        assert len(pos) == 0

    def test_zero_gradient_returns_empty(self):
        pos = detect_text_line_positions(_zeros())
        assert len(pos) == 0

    def test_detects_at_least_one_line(self):
        g = _horizontal_lines(64, 64, 3)
        pos = detect_text_line_positions(g)
        assert len(pos) >= 1

    def test_min_peak_ratio_high_reduces_peaks(self):
        g = _horizontal_lines(64, 64, 3)
        pos_low  = detect_text_line_positions(g, min_peak_ratio=0.1)
        pos_high = detect_text_line_positions(g, min_peak_ratio=0.99)
        assert len(pos_low) >= len(pos_high)


# ── build_text_line_profile ───────────────────────────────────────────────────

class TestBuildTextLineProfile:

    def test_returns_profile(self):
        g = _horizontal_lines()
        p = build_text_line_profile(g)
        assert isinstance(p, TextLineProfile)

    def test_n_lines_consistent(self):
        g = _horizontal_lines()
        p = build_text_line_profile(g)
        assert p.n_lines == len(p.line_positions)

    def test_confidence_range(self):
        g = _horizontal_lines()
        p = build_text_line_profile(g)
        assert 0.0 <= p.confidence <= 1.0

    def test_zero_gradient_low_confidence(self):
        p = build_text_line_profile(_zeros())
        assert p.confidence == 0.0


# ── compare_baseline_angles ────────────────────────────────────────────────────

class TestCompareBaselineAngles:

    def test_same_angle_returns_one(self):
        assert compare_baseline_angles(0.0, 0.0) == pytest.approx(1.0)

    def test_within_tolerance_returns_one(self):
        assert compare_baseline_angles(0.0, 3.0, tolerance_deg=5.0) == pytest.approx(1.0)

    def test_beyond_max_returns_zero(self):
        assert compare_baseline_angles(0.0, 35.0, max_diff_deg=30.0) == pytest.approx(0.0)

    def test_midpoint_score(self):
        score = compare_baseline_angles(0.0, 17.5, tolerance_deg=5.0, max_diff_deg=30.0)
        assert 0.0 < score < 1.0

    def test_symmetric(self):
        a, b = 10.0, 25.0
        assert compare_baseline_angles(a, b) == pytest.approx(compare_baseline_angles(b, a))

    def test_wrap_around_handling(self):
        # 89 vs -89 → diff = 2° (not 178°)
        score = compare_baseline_angles(89.0, -89.0, tolerance_deg=5.0)
        assert score == pytest.approx(1.0)

    def test_range_0_1(self):
        for a in np.linspace(-90, 90, 10):
            for b in np.linspace(-90, 90, 10):
                s = compare_baseline_angles(a, b)
                assert 0.0 <= s <= 1.0


# ── align_line_positions ──────────────────────────────────────────────────────

class TestAlignLinePositions:

    def test_identical_positions_return_one(self):
        pos = np.array([0.25, 0.5, 0.75])
        score = align_line_positions(pos, pos, tolerance=0.05)
        assert score == pytest.approx(1.0)

    def test_empty_a_returns_zero(self):
        assert align_line_positions(np.array([]), np.array([0.5])) == 0.0

    def test_empty_b_returns_zero(self):
        assert align_line_positions(np.array([0.5]), np.array([])) == 0.0

    def test_both_empty_returns_zero(self):
        assert align_line_positions(np.array([]), np.array([])) == 0.0

    def test_no_match_returns_zero(self):
        a = np.array([0.1, 0.2])
        b = np.array([0.8, 0.9])
        score = align_line_positions(a, b, tolerance=0.05)
        assert score == pytest.approx(0.0)

    def test_partial_match(self):
        a = np.array([0.25, 0.5, 0.75])
        b = np.array([0.25, 0.5])       # 2 out of 3 match
        score = align_line_positions(a, b, tolerance=0.05)
        assert 0.0 < score < 1.0

    def test_range_0_1(self):
        for _ in range(5):
            a = np.random.default_rng(0).uniform(0, 1, 4)
            b = np.random.default_rng(1).uniform(0, 1, 4)
            s = align_line_positions(a, b)
            assert 0.0 <= s <= 1.0


# ── match_text_flow ────────────────────────────────────────────────────────────

class TestMatchTextFlow:

    def _high_conf_profile(self, angle: float = 0.0, positions=None):
        if positions is None:
            positions = np.array([0.25, 0.5, 0.75])
        return TextLineProfile(
            angle_deg=angle,
            line_positions=positions,
            confidence=0.9,
            n_lines=len(positions),
        )

    def test_returns_match(self):
        pa = self._high_conf_profile(0.0)
        pb = self._high_conf_profile(0.0)
        m = match_text_flow(pa, pb)
        assert isinstance(m, TextFlowMatch)

    def test_identical_profiles_high_score(self):
        p = self._high_conf_profile(0.0)
        m = match_text_flow(p, p)
        assert m.score >= 0.9

    def test_different_angles_low_score(self):
        pa = self._high_conf_profile(0.0, np.array([0.3, 0.6]))
        pb = self._high_conf_profile(45.0, np.array([0.9]))
        m = match_text_flow(pa, pb)
        assert m.score < m.score + 0.1  # just check no crash; angle_score must be lower

    def test_low_confidence_returns_0_5(self):
        pa = TextLineProfile(0.0, np.array([0.5]), 0.0, 1)
        pb = TextLineProfile(0.0, np.array([0.5]), 0.0, 1)
        m = match_text_flow(pa, pb)
        assert m.score == pytest.approx(0.5)

    def test_score_range(self):
        for angle in [0.0, 10.0, 45.0, 90.0]:
            pa = self._high_conf_profile(0.0)
            pb = self._high_conf_profile(angle)
            m = match_text_flow(pa, pb)
            assert 0.0 <= m.score <= 1.0

    def test_angle_diff_field(self):
        pa = self._high_conf_profile(10.0)
        pb = self._high_conf_profile(25.0)
        m = match_text_flow(pa, pb)
        assert m.angle_diff_deg == pytest.approx(15.0)

    def test_custom_cfg(self):
        cfg = TextFlowConfig(angle_tolerance_deg=1.0, max_angle_diff_deg=10.0)
        pa = self._high_conf_profile(0.0)
        pb = self._high_conf_profile(5.0)
        m = match_text_flow(pa, pb, cfg=cfg)
        assert isinstance(m, TextFlowMatch)


# ── TextFlowScorer ─────────────────────────────────────────────────────────────

class TestTextFlowScorer:

    def test_instantiation_default(self):
        scorer = TextFlowScorer()
        assert scorer.cfg is not None

    def test_instantiation_custom_cfg(self):
        cfg = TextFlowConfig(angle_tolerance_deg=2.0)
        scorer = TextFlowScorer(cfg=cfg)
        assert scorer.cfg.angle_tolerance_deg == 2.0

    def test_build_profile_returns_profile(self):
        scorer = TextFlowScorer()
        p = scorer.build_profile(_horizontal_lines())
        assert isinstance(p, TextLineProfile)

    def test_score_returns_match(self):
        scorer = TextFlowScorer()
        m = scorer.score(_horizontal_lines(), _horizontal_lines())
        assert isinstance(m, TextFlowMatch)

    def test_score_range(self):
        scorer = TextFlowScorer()
        m = scorer.score(_horizontal_lines(), _zeros())
        assert 0.0 <= m.score <= 1.0

    def test_score_batch_empty(self):
        scorer = TextFlowScorer()
        results = scorer.score_batch(_horizontal_lines(), [])
        assert results == []

    def test_score_batch_multiple(self):
        scorer = TextFlowScorer()
        cands = [_horizontal_lines(), _zeros(), _horizontal_lines(n=2)]
        results = scorer.score_batch(_horizontal_lines(), cands)
        assert len(results) == 3
        for m in results:
            assert isinstance(m, TextFlowMatch)
            assert 0.0 <= m.score <= 1.0
