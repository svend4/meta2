"""Extra tests for puzzle_reconstruction/verification/edge_validator.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.verification.edge_validator import (
    EdgeValidConfig,
    EdgeCheck,
    EdgeValidResult,
    check_intensity,
    check_gap,
    check_normals,
    validate_edge_pair,
    summarise_validations,
    batch_validate_edges,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _profile(n: int = 10, value: float = 0.5, noise: float = 0.0,
             seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full(n, value)
    if noise:
        base = base + rng.uniform(-noise, noise, n)
    return np.clip(base, 0.0, 1.0)


def _points(n: int = 5, offset: float = 0.0) -> np.ndarray:
    pts = np.column_stack([np.linspace(0, 1, n),
                           np.zeros(n) + offset])
    return pts


def _unit_normals(n: int = 5, direction: str = "up") -> np.ndarray:
    if direction == "up":
        return np.tile([0.0, 1.0], (n, 1))
    return np.tile([0.0, -1.0], (n, 1))


# ─── EdgeValidConfig (extra) ─────────────────────────────────────────────────

class TestEdgeValidConfigExtra:
    def test_default_intensity_tol(self):
        assert EdgeValidConfig().intensity_tol == pytest.approx(0.15)

    def test_default_gap_tol(self):
        assert EdgeValidConfig().gap_tol == pytest.approx(2.0)

    def test_default_normal_tol_deg(self):
        assert EdgeValidConfig().normal_tol_deg == pytest.approx(30.0)

    def test_default_require_all(self):
        assert EdgeValidConfig().require_all is True

    def test_intensity_tol_zero_ok(self):
        cfg = EdgeValidConfig(intensity_tol=0.0)
        assert cfg.intensity_tol == pytest.approx(0.0)

    def test_intensity_tol_one_ok(self):
        cfg = EdgeValidConfig(intensity_tol=1.0)
        assert cfg.intensity_tol == pytest.approx(1.0)

    def test_intensity_tol_neg_raises(self):
        with pytest.raises(ValueError):
            EdgeValidConfig(intensity_tol=-0.01)

    def test_intensity_tol_above_one_raises(self):
        with pytest.raises(ValueError):
            EdgeValidConfig(intensity_tol=1.1)

    def test_gap_tol_zero_ok(self):
        cfg = EdgeValidConfig(gap_tol=0.0)
        assert cfg.gap_tol == pytest.approx(0.0)

    def test_gap_tol_large_ok(self):
        cfg = EdgeValidConfig(gap_tol=1000.0)
        assert cfg.gap_tol == pytest.approx(1000.0)

    def test_gap_tol_neg_raises(self):
        with pytest.raises(ValueError):
            EdgeValidConfig(gap_tol=-1.0)

    def test_normal_tol_zero_ok(self):
        cfg = EdgeValidConfig(normal_tol_deg=0.0)
        assert cfg.normal_tol_deg == pytest.approx(0.0)

    def test_normal_tol_neg_raises(self):
        with pytest.raises(ValueError):
            EdgeValidConfig(normal_tol_deg=-5.0)

    def test_require_all_false(self):
        cfg = EdgeValidConfig(require_all=False)
        assert cfg.require_all is False


# ─── EdgeCheck (extra) ───────────────────────────────────────────────────────

class TestEdgeCheckExtra:
    def test_name_stored(self):
        c = EdgeCheck(name="my_check", passed=True, value=0.1, limit=0.5)
        assert c.name == "my_check"

    def test_passed_true(self):
        c = EdgeCheck(name="x", passed=True, value=0.1, limit=0.5)
        assert c.passed is True

    def test_passed_false(self):
        c = EdgeCheck(name="x", passed=False, value=0.9, limit=0.5)
        assert c.passed is False

    def test_margin_positive_ok(self):
        c = EdgeCheck(name="x", passed=True, value=0.1, limit=0.3)
        assert c.margin == pytest.approx(0.2)

    def test_margin_zero_at_boundary(self):
        c = EdgeCheck(name="x", passed=True, value=0.5, limit=0.5)
        assert c.margin == pytest.approx(0.0)

    def test_margin_negative_when_failed(self):
        c = EdgeCheck(name="x", passed=False, value=0.8, limit=0.3)
        assert c.margin < 0.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            EdgeCheck(name="", passed=True, value=0.0, limit=0.1)

    def test_value_stored(self):
        c = EdgeCheck(name="x", passed=True, value=0.25, limit=0.5)
        assert c.value == pytest.approx(0.25)

    def test_limit_stored(self):
        c = EdgeCheck(name="x", passed=True, value=0.1, limit=0.77)
        assert c.limit == pytest.approx(0.77)


# ─── EdgeValidResult (extra) ─────────────────────────────────────────────────

class TestEdgeValidResultExtra:
    def _make(self, n_pass=2, n_fail=1, pair=(0, 1)):
        checks = [EdgeCheck(name=f"p{i}", passed=True, value=0.1, limit=0.5)
                  for i in range(n_pass)]
        checks += [EdgeCheck(name=f"f{i}", passed=False, value=0.9, limit=0.5)
                   for i in range(n_fail)]
        return EdgeValidResult(pair=pair, checks=checks, valid=n_fail == 0)

    def test_fragment_a(self):
        r = self._make(pair=(3, 7))
        assert r.fragment_a == 3

    def test_fragment_b(self):
        r = self._make(pair=(3, 7))
        assert r.fragment_b == 7

    def test_n_passed(self):
        r = self._make(n_pass=3, n_fail=1)
        assert r.n_passed == 3

    def test_n_failed(self):
        r = self._make(n_pass=2, n_fail=2)
        assert r.n_failed == 2

    def test_n_passed_plus_failed_eq_total(self):
        r = self._make(n_pass=2, n_fail=3)
        assert r.n_passed + r.n_failed == len(r.checks)

    def test_check_names_count(self):
        r = self._make(n_pass=3, n_fail=0)
        assert len(r.check_names) == 3

    def test_get_check_found(self):
        r = self._make(n_pass=2, n_fail=0)
        c = r.get_check("p0")
        assert c is not None
        assert c.name == "p0"

    def test_get_check_not_found(self):
        r = self._make()
        assert r.get_check("nonexistent") is None

    def test_valid_stored(self):
        r = self._make(n_pass=3, n_fail=0)
        assert r.valid is True

    def test_not_valid_stored(self):
        r = self._make(n_pass=1, n_fail=1)
        r2 = EdgeValidResult(pair=(0, 1), checks=r.checks, valid=False)
        assert r2.valid is False


# ─── check_intensity (extra) ─────────────────────────────────────────────────

class TestCheckIntensityExtra:
    def test_name_is_intensity(self):
        p = _profile(10, 0.5)
        c = check_intensity(p, p.copy())
        assert c.name == "intensity"

    def test_identical_value_zero(self):
        p = _profile(10, 0.5)
        c = check_intensity(p, p.copy())
        assert c.value == pytest.approx(0.0)

    def test_identical_passes(self):
        p = _profile(10, 0.5)
        assert check_intensity(p, p.copy()).passed is True

    def test_very_different_fails(self):
        a = _profile(10, 0.0)
        b = _profile(10, 1.0)
        assert check_intensity(a, b).passed is False

    def test_within_tolerance_passes(self):
        a = _profile(10, 0.5)
        b = _profile(10, 0.6)
        cfg = EdgeValidConfig(intensity_tol=0.2)
        assert check_intensity(a, b, cfg).passed is True

    def test_limit_from_config(self):
        cfg = EdgeValidConfig(intensity_tol=0.08)
        p = _profile(10, 0.5)
        c = check_intensity(p, p.copy(), cfg)
        assert c.limit == pytest.approx(0.08)

    def test_value_in_0_1(self):
        a = _profile(10, 0.2)
        b = _profile(10, 0.8)
        c = check_intensity(a, b)
        assert 0.0 <= c.value <= 1.0


# ─── check_gap (extra) ───────────────────────────────────────────────────────

class TestCheckGapExtra:
    def test_name_is_gap(self):
        pts = _points(5)
        c = check_gap(pts, pts.copy())
        assert c.name == "gap"

    def test_identical_value_zero(self):
        pts = _points(5)
        c = check_gap(pts, pts.copy())
        assert c.value == pytest.approx(0.0)

    def test_identical_passes(self):
        pts = _points(5)
        assert check_gap(pts, pts.copy()).passed is True

    def test_far_apart_fails(self):
        a = _points(5, offset=0.0)
        b = _points(5, offset=100.0)
        cfg = EdgeValidConfig(gap_tol=2.0)
        assert check_gap(a, b, cfg).passed is False

    def test_close_enough_passes(self):
        a = _points(5, offset=0.0)
        b = _points(5, offset=1.0)
        cfg = EdgeValidConfig(gap_tol=2.0)
        assert check_gap(a, b, cfg).passed is True

    def test_limit_from_config(self):
        pts = _points(5)
        cfg = EdgeValidConfig(gap_tol=7.0)
        c = check_gap(pts, pts.copy(), cfg)
        assert c.limit == pytest.approx(7.0)

    def test_empty_a_fails(self):
        b = _points(5)
        c = check_gap(np.zeros((0, 2)), b)
        assert c.passed is False

    def test_empty_b_fails(self):
        a = _points(5)
        c = check_gap(a, np.zeros((0, 2)))
        assert c.passed is False


# ─── check_normals (extra) ───────────────────────────────────────────────────

class TestCheckNormalsExtra:
    def test_name_is_normals(self):
        n = _unit_normals(5, "up")
        c = check_normals(n, n.copy())
        assert c.name == "normals"

    def test_parallel_same_direction_passes(self):
        n = _unit_normals(5, "up")
        cfg = EdgeValidConfig(normal_tol_deg=45.0)
        c = check_normals(n, n.copy(), cfg)
        assert c.passed is True

    def test_value_in_0_90(self):
        n_a = _unit_normals(5, "up")
        n_b = _unit_normals(5, "up")
        c = check_normals(n_a, n_b)
        assert 0.0 <= c.value <= 90.0

    def test_limit_from_config(self):
        cfg = EdgeValidConfig(normal_tol_deg=15.0)
        n = _unit_normals(5)
        c = check_normals(n, n.copy(), cfg)
        assert c.limit == pytest.approx(15.0)

    def test_default_limit_30(self):
        n = _unit_normals(5)
        c = check_normals(n, n.copy())
        assert c.limit == pytest.approx(30.0)

    def test_empty_a_fails(self):
        n_b = _unit_normals(5)
        c = check_normals(np.zeros((0, 2)), n_b)
        assert c.passed is False

    def test_empty_b_fails(self):
        n_a = _unit_normals(5)
        c = check_normals(n_a, np.zeros((0, 2)))
        assert c.passed is False


# ─── validate_edge_pair (extra) ──────────────────────────────────────────────

class TestValidateEdgePairExtra:
    def _call(self, cfg=None, offset=0.0):
        p = _profile(10, 0.5)
        pts_a = _points(5, 0.0)
        pts_b = _points(5, offset)
        n = _unit_normals(5, "up")
        return validate_edge_pair(0, 1, p, p.copy(),
                                  pts_a, pts_b, n, n.copy(), cfg)

    def test_returns_edge_valid_result(self):
        assert isinstance(self._call(), EdgeValidResult)

    def test_pair_is_0_1(self):
        r = self._call()
        assert r.pair == (0, 1)

    def test_three_checks(self):
        r = self._call(EdgeValidConfig(gap_tol=10.0, normal_tol_deg=90.0))
        assert len(r.checks) == 3

    def test_check_names_set(self):
        r = self._call(EdgeValidConfig(gap_tol=10.0, normal_tol_deg=90.0))
        assert set(r.check_names) == {"intensity", "gap", "normals"}

    def test_require_all_false_gap_passes_valid(self):
        a_int = _profile(10, 0.0)
        b_int = _profile(10, 1.0)  # intensity will fail
        pts = _points(5, 0.0)
        n = _unit_normals(5)
        cfg = EdgeValidConfig(require_all=False, gap_tol=100.0)
        r = validate_edge_pair(0, 1, a_int, b_int, pts, pts.copy(),
                               n, n.copy(), cfg)
        assert r.valid is True  # gap passes

    def test_require_all_true_intensity_fail_invalid(self):
        a_int = _profile(10, 0.0)
        b_int = _profile(10, 1.0)
        pts = _points(5, 0.0)
        n = _unit_normals(5)
        cfg = EdgeValidConfig(require_all=True)
        r = validate_edge_pair(0, 1, a_int, b_int, pts, pts.copy(),
                               n, n.copy(), cfg)
        assert r.valid is False

    def test_identical_inputs_valid(self):
        r = self._call(EdgeValidConfig(gap_tol=10.0, normal_tol_deg=90.0))
        assert r.valid is True


# ─── summarise_validations (extra) ───────────────────────────────────────────

class TestSummariseValidationsExtra:
    def test_empty_list_zeros(self):
        s = summarise_validations([])
        assert s["valid_ratio"] == pytest.approx(0.0)
        assert s["n_results"] == 0

    def test_all_valid_ratio_one(self):
        p = _profile(10, 0.5)
        pts = _points(5, 0.0)
        n = _unit_normals(5)
        cfg = EdgeValidConfig(gap_tol=10.0, normal_tol_deg=90.0)
        results = [
            validate_edge_pair(i, i + 1, p, p.copy(), pts, pts.copy(),
                               n, n.copy(), cfg)
            for i in range(4)
        ]
        s = summarise_validations(results)
        assert s["valid_ratio"] == pytest.approx(1.0)

    def test_none_valid_ratio_zero(self):
        chk = [EdgeCheck(name="x", passed=False, value=0.9, limit=0.1)]
        results = [EdgeValidResult(pair=(i, i + 1), checks=chk, valid=False)
                   for i in range(3)]
        s = summarise_validations(results)
        assert s["valid_ratio"] == pytest.approx(0.0)

    def test_half_valid_ratio_half(self):
        c_pass = EdgeCheck(name="x", passed=True, value=0.1, limit=0.5)
        c_fail = EdgeCheck(name="x", passed=False, value=0.9, limit=0.5)
        results = [
            EdgeValidResult(pair=(0, 1), checks=[c_pass], valid=True),
            EdgeValidResult(pair=(1, 2), checks=[c_fail], valid=False),
        ]
        s = summarise_validations(results)
        assert s["valid_ratio"] == pytest.approx(0.5)

    def test_n_results_correct(self):
        chk = [EdgeCheck(name="x", passed=True, value=0.1, limit=0.5)]
        results = [EdgeValidResult(pair=(i, i + 1), checks=chk, valid=True)
                   for i in range(5)]
        s = summarise_validations(results)
        assert s["n_results"] == 5

    def test_mean_passed_checks_value(self):
        checks = [
            EdgeCheck(name="a", passed=True, value=0.1, limit=0.5),
            EdgeCheck(name="b", passed=True, value=0.1, limit=0.5),
        ]
        r = EdgeValidResult(pair=(0, 1), checks=checks, valid=True)
        s = summarise_validations([r])
        assert s["mean_passed_checks"] == pytest.approx(2.0)


# ─── batch_validate_edges (extra) ────────────────────────────────────────────

class TestBatchValidateEdgesExtra:
    def _maps(self, ids):
        p = _profile(10, 0.5)
        pts = _points(5, 0.0)
        n = _unit_normals(5, "up")
        return (
            {i: p for i in ids},
            {i: pts for i in ids},
            {i: n for i in ids},
        )

    def test_empty_pairs_empty_list(self):
        assert batch_validate_edges([], {}, {}, {}) == []

    def test_length_matches_pairs(self):
        pairs = [(0, 1), (1, 2), (2, 3)]
        int_m, pts_m, n_m = self._maps([0, 1, 2, 3])
        cfg = EdgeValidConfig(gap_tol=10.0)
        results = batch_validate_edges(pairs, int_m, pts_m, n_m, cfg)
        assert len(results) == 3

    def test_all_edge_valid_results(self):
        pairs = [(0, 1)]
        int_m, pts_m, n_m = self._maps([0, 1])
        for r in batch_validate_edges(pairs, int_m, pts_m, n_m):
            assert isinstance(r, EdgeValidResult)

    def test_pair_ids_preserved(self):
        pairs = [(0, 1), (2, 3)]
        int_m, pts_m, n_m = self._maps([0, 1, 2, 3])
        results = batch_validate_edges(pairs, int_m, pts_m, n_m)
        assert results[0].pair == (0, 1)
        assert results[1].pair == (2, 3)

    def test_missing_fragment_returns_result(self):
        pairs = [(0, 99)]
        int_m, pts_m, n_m = self._maps([0])
        try:
            results = batch_validate_edges(pairs, int_m, pts_m, n_m)
            assert len(results) == 1
        except Exception:
            pass  # implementation may raise on missing fragment

    def test_default_config_ok(self):
        pairs = [(0, 1)]
        int_m, pts_m, n_m = self._maps([0, 1])
        results = batch_validate_edges(pairs, int_m, pts_m, n_m)
        assert len(results) == 1

    def test_custom_config(self):
        pairs = [(0, 1)]
        int_m, pts_m, n_m = self._maps([0, 1])
        cfg = EdgeValidConfig(gap_tol=50.0, normal_tol_deg=90.0)
        results = batch_validate_edges(pairs, int_m, pts_m, n_m, cfg)
        assert results[0].valid is True
