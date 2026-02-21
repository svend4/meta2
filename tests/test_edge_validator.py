"""Тесты для puzzle_reconstruction.verification.edge_validator."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

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


# ─── TestEdgeValidConfig ──────────────────────────────────────────────────────

class TestEdgeValidConfig:
    def test_defaults(self):
        cfg = EdgeValidConfig()
        assert cfg.intensity_tol == pytest.approx(0.15)
        assert cfg.gap_tol == pytest.approx(2.0)
        assert cfg.normal_tol_deg == pytest.approx(30.0)
        assert cfg.require_all is True

    def test_valid_custom(self):
        cfg = EdgeValidConfig(intensity_tol=0.3, gap_tol=5.0,
                              normal_tol_deg=45.0, require_all=False)
        assert cfg.intensity_tol == pytest.approx(0.3)
        assert cfg.require_all is False

    def test_intensity_tol_zero_ok(self):
        cfg = EdgeValidConfig(intensity_tol=0.0)
        assert cfg.intensity_tol == 0.0

    def test_intensity_tol_one_ok(self):
        cfg = EdgeValidConfig(intensity_tol=1.0)
        assert cfg.intensity_tol == 1.0

    def test_intensity_tol_above_raises(self):
        with pytest.raises(ValueError):
            EdgeValidConfig(intensity_tol=1.1)

    def test_intensity_tol_neg_raises(self):
        with pytest.raises(ValueError):
            EdgeValidConfig(intensity_tol=-0.01)

    def test_gap_tol_zero_ok(self):
        cfg = EdgeValidConfig(gap_tol=0.0)
        assert cfg.gap_tol == 0.0

    def test_gap_tol_neg_raises(self):
        with pytest.raises(ValueError):
            EdgeValidConfig(gap_tol=-1.0)

    def test_normal_tol_zero_ok(self):
        cfg = EdgeValidConfig(normal_tol_deg=0.0)
        assert cfg.normal_tol_deg == 0.0

    def test_normal_tol_neg_raises(self):
        with pytest.raises(ValueError):
            EdgeValidConfig(normal_tol_deg=-5.0)


# ─── TestEdgeCheck ────────────────────────────────────────────────────────────

class TestEdgeCheck:
    def _make(self, passed=True, value=0.1, limit=0.2):
        return EdgeCheck(name="test_check", passed=passed,
                         value=value, limit=limit)

    def test_basic(self):
        c = self._make()
        assert c.name == "test_check"
        assert c.passed is True
        assert c.value == pytest.approx(0.1)
        assert c.limit == pytest.approx(0.2)

    def test_margin_positive_when_ok(self):
        c = self._make(passed=True, value=0.1, limit=0.2)
        assert c.margin == pytest.approx(0.1)

    def test_margin_negative_when_failed(self):
        c = EdgeCheck(name="x", passed=False, value=0.5, limit=0.2)
        assert c.margin < 0

    def test_margin_zero_at_threshold(self):
        c = EdgeCheck(name="x", passed=True, value=0.2, limit=0.2)
        assert c.margin == pytest.approx(0.0)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            EdgeCheck(name="", passed=True, value=0.0, limit=0.1)


# ─── TestEdgeValidResult ──────────────────────────────────────────────────────

class TestEdgeValidResult:
    def _make_result(self, valid=True, n_pass=2, n_fail=1):
        checks = [EdgeCheck(name=f"c{i}", passed=True, value=0.1, limit=0.5)
                  for i in range(n_pass)]
        checks += [EdgeCheck(name=f"f{i}", passed=False, value=0.9, limit=0.5)
                   for i in range(n_fail)]
        return EdgeValidResult(pair=(0, 1), checks=checks, valid=valid)

    def test_fragment_a(self):
        r = self._make_result()
        assert r.fragment_a == 0

    def test_fragment_b(self):
        r = self._make_result()
        assert r.fragment_b == 1

    def test_n_passed(self):
        r = self._make_result(n_pass=2, n_fail=1)
        assert r.n_passed == 2

    def test_n_failed(self):
        r = self._make_result(n_pass=2, n_fail=1)
        assert r.n_failed == 1

    def test_n_passed_plus_failed_equals_total(self):
        r = self._make_result(n_pass=2, n_fail=1)
        assert r.n_passed + r.n_failed == len(r.checks)

    def test_check_names_length(self):
        r = self._make_result(n_pass=2, n_fail=0)
        assert len(r.check_names) == 2

    def test_get_check_found(self):
        r = self._make_result(n_pass=2, n_fail=0)
        c = r.get_check("c0")
        assert c is not None
        assert c.name == "c0"

    def test_get_check_not_found(self):
        r = self._make_result()
        assert r.get_check("nonexistent") is None

    def test_valid_stored(self):
        r = self._make_result(valid=False)
        assert r.valid is False


# ─── TestCheckIntensity ───────────────────────────────────────────────────────

class TestCheckIntensity:
    def test_identical_profiles_pass(self):
        p = _profile(10, 0.5)
        c = check_intensity(p, p.copy())
        assert c.name == "intensity"
        assert c.passed is True
        assert c.value == pytest.approx(0.0)

    def test_very_different_profiles_fail(self):
        a = _profile(10, 0.0)
        b = _profile(10, 1.0)
        c = check_intensity(a, b)
        assert c.passed is False

    def test_within_tolerance_passes(self):
        a = _profile(10, 0.5)
        b = _profile(10, 0.55)
        cfg = EdgeValidConfig(intensity_tol=0.15)
        c = check_intensity(a, b, cfg)
        assert c.passed is True

    def test_limit_matches_config(self):
        cfg = EdgeValidConfig(intensity_tol=0.1)
        a = _profile(10, 0.5)
        c = check_intensity(a, a.copy(), cfg)
        assert c.limit == pytest.approx(0.1)

    def test_value_in_0_1(self):
        a = _profile(10, 0.3, noise=0.1)
        b = _profile(10, 0.7, noise=0.1)
        c = check_intensity(a, b)
        assert 0.0 <= c.value <= 1.0

    def test_default_config_used(self):
        a = _profile(10)
        c = check_intensity(a, a.copy())
        assert c.limit == pytest.approx(0.15)


# ─── TestCheckGap ─────────────────────────────────────────────────────────────

class TestCheckGap:
    def test_overlapping_points_pass(self):
        pts = _points(5, offset=0.0)
        c = check_gap(pts, pts.copy())
        assert c.name == "gap"
        assert c.passed is True
        assert c.value == pytest.approx(0.0)

    def test_far_points_fail(self):
        a = _points(5, offset=0.0)
        b = _points(5, offset=100.0)
        cfg = EdgeValidConfig(gap_tol=2.0)
        c = check_gap(a, b, cfg)
        assert c.passed is False

    def test_close_enough_pass(self):
        a = _points(5, offset=0.0)
        b = _points(5, offset=1.5)
        cfg = EdgeValidConfig(gap_tol=2.0)
        c = check_gap(a, b, cfg)
        assert c.passed is True

    def test_empty_a_fails(self):
        b = _points(5)
        cfg = EdgeValidConfig()
        c = check_gap(np.zeros((0, 2)), b, cfg)
        assert c.passed is False

    def test_empty_b_fails(self):
        a = _points(5)
        cfg = EdgeValidConfig()
        c = check_gap(a, np.zeros((0, 2)), cfg)
        assert c.passed is False

    def test_limit_matches_config(self):
        a = _points(3)
        cfg = EdgeValidConfig(gap_tol=5.0)
        c = check_gap(a, a.copy(), cfg)
        assert c.limit == pytest.approx(5.0)


# ─── TestCheckNormals ─────────────────────────────────────────────────────────

class TestCheckNormals:
    def test_parallel_opposite_pass(self):
        n_a = _unit_normals(5, "up")
        n_b = _unit_normals(5, "up")
        cfg = EdgeValidConfig(normal_tol_deg=45.0)
        c = check_normals(n_a, n_b, cfg)
        assert c.name == "normals"
        assert c.passed is True

    def test_empty_a_fails(self):
        n_b = _unit_normals(5)
        c = check_normals(np.zeros((0, 2)), n_b)
        assert c.passed is False

    def test_empty_b_fails(self):
        n_a = _unit_normals(5)
        c = check_normals(n_a, np.zeros((0, 2)))
        assert c.passed is False

    def test_value_in_0_90(self):
        n_a = _unit_normals(5, "up")
        n_b = _unit_normals(5, "up")
        c = check_normals(n_a, n_b)
        assert 0.0 <= c.value <= 90.0

    def test_limit_matches_config(self):
        cfg = EdgeValidConfig(normal_tol_deg=20.0)
        n = _unit_normals(5)
        c = check_normals(n, n.copy(), cfg)
        assert c.limit == pytest.approx(20.0)

    def test_default_config_used(self):
        n = _unit_normals(5)
        c = check_normals(n, n.copy())
        assert c.limit == pytest.approx(30.0)


# ─── TestValidateEdgePair ─────────────────────────────────────────────────────

class TestValidateEdgePair:
    def _call(self, cfg=None):
        p = _profile(10, 0.5)
        pts = _points(5, 0.0)
        n = _unit_normals(5, "up")
        return validate_edge_pair(0, 1, p, p.copy(), pts, pts.copy(),
                                  n, n.copy(), cfg)

    def test_returns_edge_valid_result(self):
        r = self._call()
        assert isinstance(r, EdgeValidResult)

    def test_pair_ids(self):
        r = self._call()
        assert r.pair == (0, 1)

    def test_three_checks(self):
        r = self._call()
        assert len(r.checks) == 3

    def test_check_names(self):
        r = self._call()
        assert set(r.check_names) == {"intensity", "gap", "normals"}

    def test_require_all_false_any_check(self):
        # Make intensity fail but gap pass
        a_int = _profile(10, 0.0)
        b_int = _profile(10, 1.0)
        pts = _points(5, 0.0)
        n = _unit_normals(5, "up")
        cfg = EdgeValidConfig(require_all=False, gap_tol=100.0)
        r = validate_edge_pair(0, 1, a_int, b_int, pts, pts.copy(),
                               n, n.copy(), cfg)
        # gap should pass (same points), so valid == True even if intensity fails
        assert r.valid is True

    def test_require_all_true_all_must_pass(self):
        a_int = _profile(10, 0.0)
        b_int = _profile(10, 1.0)
        pts = _points(5, 0.0)
        n = _unit_normals(5)
        cfg = EdgeValidConfig(require_all=True)
        r = validate_edge_pair(0, 1, a_int, b_int, pts, pts.copy(),
                               n, n.copy(), cfg)
        # intensity fails → valid is False
        assert r.valid is False

    def test_identical_inputs_high_chance_valid(self):
        r = self._call(EdgeValidConfig(gap_tol=10.0, normal_tol_deg=45.0))
        assert r.valid is True


# ─── TestSummariseValidations ─────────────────────────────────────────────────

class TestSummariseValidations:
    def test_empty_list(self):
        s = summarise_validations([])
        assert s["valid_ratio"] == pytest.approx(0.0)
        assert s["mean_passed_checks"] == pytest.approx(0.0)
        assert s["n_results"] == 0

    def test_all_valid(self):
        p = _profile(10, 0.5)
        pts = _points(5, 0.0)
        n = _unit_normals(5)
        cfg = EdgeValidConfig(gap_tol=10.0, normal_tol_deg=45.0)
        results = [
            validate_edge_pair(i, i + 1, p, p.copy(), pts, pts.copy(),
                               n, n.copy(), cfg)
            for i in range(3)
        ]
        s = summarise_validations(results)
        assert s["n_results"] == 3
        assert s["valid_ratio"] == pytest.approx(1.0)

    def test_none_valid(self):
        checks = [EdgeCheck(name="intensity", passed=False, value=0.9, limit=0.15)]
        results = [
            EdgeValidResult(pair=(i, i + 1), checks=checks, valid=False)
            for i in range(4)
        ]
        s = summarise_validations(results)
        assert s["valid_ratio"] == pytest.approx(0.0)

    def test_partial_valid(self):
        c_pass = EdgeCheck(name="x", passed=True, value=0.1, limit=0.5)
        c_fail = EdgeCheck(name="x", passed=False, value=0.9, limit=0.5)
        results = [
            EdgeValidResult(pair=(0, 1), checks=[c_pass], valid=True),
            EdgeValidResult(pair=(1, 2), checks=[c_fail], valid=False),
        ]
        s = summarise_validations(results)
        assert s["valid_ratio"] == pytest.approx(0.5)
        assert s["n_results"] == 2

    def test_mean_passed_checks_value(self):
        checks = [
            EdgeCheck(name="a", passed=True, value=0.1, limit=0.5),
            EdgeCheck(name="b", passed=True, value=0.1, limit=0.5),
        ]
        r = EdgeValidResult(pair=(0, 1), checks=checks, valid=True)
        s = summarise_validations([r])
        assert s["mean_passed_checks"] == pytest.approx(2.0)


# ─── TestBatchValidateEdges ───────────────────────────────────────────────────

class TestBatchValidateEdges:
    def _maps(self, ids):
        p = _profile(10, 0.5)
        pts = _points(5, 0.0)
        n = _unit_normals(5, "up")
        return (
            {i: p for i in ids},
            {i: pts for i in ids},
            {i: n for i in ids},
        )

    def test_basic(self):
        pairs = [(0, 1), (1, 2), (2, 3)]
        int_m, pts_m, n_m = self._maps([0, 1, 2, 3])
        cfg = EdgeValidConfig(gap_tol=10.0)
        results = batch_validate_edges(pairs, int_m, pts_m, n_m, cfg)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, EdgeValidResult)

    def test_empty_pairs(self):
        assert batch_validate_edges([], {}, {}, {}) == []

    def test_pair_ids_correct(self):
        pairs = [(0, 1), (2, 3)]
        int_m, pts_m, n_m = self._maps([0, 1, 2, 3])
        results = batch_validate_edges(pairs, int_m, pts_m, n_m)
        assert results[0].pair == (0, 1)
        assert results[1].pair == (2, 3)

    def test_missing_fragment_gets_empty(self):
        pairs = [(0, 99)]
        int_m, pts_m, n_m = self._maps([0])
        results = batch_validate_edges(pairs, int_m, pts_m, n_m)
        assert len(results) == 1
        # gap check uses empty points → fail
        gap_check = results[0].get_check("gap")
        assert gap_check is not None
        assert gap_check.passed is False

    def test_default_config(self):
        pairs = [(0, 1)]
        int_m, pts_m, n_m = self._maps([0, 1])
        results = batch_validate_edges(pairs, int_m, pts_m, n_m)
        assert len(results) == 1
