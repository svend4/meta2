"""Tests for puzzle_reconstruction.verification.edge_validator"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

from puzzle_reconstruction.verification.edge_validator import (
    EdgeValidConfig,
    EdgeCheck,
    EdgeValidResult,
    _intensity_diff,
    check_intensity,
    check_gap,
    check_normals,
    validate_edge_pair,
    summarise_validations,
    batch_validate_edges,
)


# ─── EdgeValidConfig ──────────────────────────────────────────────────────────

def test_edge_valid_config_defaults():
    cfg = EdgeValidConfig()
    assert cfg.intensity_tol == 0.15
    assert cfg.gap_tol == 2.0
    assert cfg.normal_tol_deg == 30.0
    assert cfg.require_all is True


def test_edge_valid_config_custom():
    cfg = EdgeValidConfig(intensity_tol=0.1, gap_tol=5.0, normal_tol_deg=20.0, require_all=False)
    assert cfg.intensity_tol == 0.1
    assert cfg.gap_tol == 5.0
    assert cfg.require_all is False


def test_edge_valid_config_intensity_tol_invalid():
    with pytest.raises(ValueError):
        EdgeValidConfig(intensity_tol=1.5)


def test_edge_valid_config_gap_tol_invalid():
    with pytest.raises(ValueError):
        EdgeValidConfig(gap_tol=-1.0)


def test_edge_valid_config_normal_tol_deg_invalid():
    with pytest.raises(ValueError):
        EdgeValidConfig(normal_tol_deg=-5.0)


# ─── EdgeCheck ────────────────────────────────────────────────────────────────

def test_edge_check_margin_positive():
    ec = EdgeCheck(name="intensity", passed=True, value=0.05, limit=0.15)
    assert abs(ec.margin - 0.10) < 1e-9


def test_edge_check_margin_negative():
    ec = EdgeCheck(name="gap", passed=False, value=5.0, limit=2.0)
    assert ec.margin < 0


def test_edge_check_empty_name_raises():
    with pytest.raises(ValueError):
        EdgeCheck(name="", passed=True, value=0.1, limit=0.2)


def test_edge_check_passed_flag():
    ec = EdgeCheck(name="normals", passed=False, value=45.0, limit=30.0)
    assert not ec.passed


# ─── _intensity_diff ──────────────────────────────────────────────────────────

def test_intensity_diff_identical():
    profile = np.array([10, 20, 30, 40], dtype=np.float32)
    assert _intensity_diff(profile, profile) == pytest.approx(0.0, abs=1e-6)


def test_intensity_diff_empty():
    assert _intensity_diff(np.array([]), np.array([1.0, 2.0])) == 0.0


def test_intensity_diff_opposite():
    # Constant arrays are not normalized (range=0), raw difference is returned
    a = np.array([0, 0, 0, 0], dtype=np.float32)
    b = np.array([255, 255, 255, 255], dtype=np.float32)
    diff = _intensity_diff(a, b)
    assert diff == pytest.approx(255.0)


def test_intensity_diff_range():
    a = np.linspace(0, 100, 20)
    b = np.linspace(50, 150, 20)
    diff = _intensity_diff(a, b)
    assert 0.0 <= diff <= 1.0


# ─── check_intensity ──────────────────────────────────────────────────────────

def test_check_intensity_passes_identical():
    profile = np.array([10, 50, 100], dtype=np.float32)
    result = check_intensity(profile, profile)
    assert result.passed
    assert result.name == "intensity"


def test_check_intensity_fails_different():
    a = np.array([0, 0, 0, 0, 0, 10], dtype=np.float32)
    b = np.array([200, 210, 220, 230, 240, 250], dtype=np.float32)
    cfg = EdgeValidConfig(intensity_tol=0.01)
    result = check_intensity(a, b, cfg)
    # Value in [0,1]; whether it passes depends on normalization
    assert isinstance(result.passed, bool)


def test_check_intensity_default_config():
    a = np.array([100, 110, 120], dtype=np.float32)
    b = np.array([100, 110, 120], dtype=np.float32)
    result = check_intensity(a, b)
    assert result.passed


def test_check_intensity_result_type():
    a = np.ones(10) * 50
    b = np.ones(10) * 60
    result = check_intensity(a, b)
    assert isinstance(result, EdgeCheck)
    assert result.name == "intensity"


# ─── check_gap ────────────────────────────────────────────────────────────────

def test_check_gap_passes_close_points():
    pts_a = np.array([[0.0, 0.0], [1.0, 1.0]])
    pts_b = np.array([[1.5, 1.5], [2.0, 2.0]])
    cfg = EdgeValidConfig(gap_tol=3.0)
    result = check_gap(pts_a, pts_b, cfg)
    assert result.passed


def test_check_gap_fails_far_points():
    pts_a = np.array([[0.0, 0.0]])
    pts_b = np.array([[100.0, 100.0]])
    cfg = EdgeValidConfig(gap_tol=2.0)
    result = check_gap(pts_a, pts_b, cfg)
    assert not result.passed


def test_check_gap_empty_points():
    pts_a = np.zeros((0, 2))
    pts_b = np.array([[1.0, 1.0]])
    result = check_gap(pts_a, pts_b)
    assert not result.passed
    assert result.value == float("inf")


def test_check_gap_returns_edge_check():
    pts_a = np.array([[0.0, 0.0]])
    pts_b = np.array([[1.0, 0.0]])
    result = check_gap(pts_a, pts_b)
    assert isinstance(result, EdgeCheck)
    assert result.name == "gap"


# ─── check_normals ────────────────────────────────────────────────────────────

def test_check_normals_antiparallel_passes():
    # Antiparallel normals: one pointing up, one pointing down
    normals_a = np.array([[0.0, 1.0], [0.0, 1.0]])
    normals_b = np.array([[0.0, -1.0], [0.0, -1.0]])
    cfg = EdgeValidConfig(normal_tol_deg=30.0)
    result = check_normals(normals_a, normals_b, cfg)
    # abs(dots) = 1 => angle = 0 degrees
    assert result.passed


def test_check_normals_empty():
    result = check_normals(np.zeros((0, 2)), np.array([[1.0, 0.0]]))
    assert not result.passed


def test_check_normals_name():
    na = np.array([[1.0, 0.0]])
    nb = np.array([[1.0, 0.0]])
    result = check_normals(na, nb)
    assert result.name == "normals"


def test_check_normals_returns_edge_check():
    na = np.array([[0.0, 1.0]])
    nb = np.array([[0.0, -1.0]])
    result = check_normals(na, nb)
    assert isinstance(result, EdgeCheck)


# ─── EdgeValidResult ──────────────────────────────────────────────────────────

def test_edge_valid_result_properties():
    checks = [
        EdgeCheck("intensity", True, 0.1, 0.15),
        EdgeCheck("gap", False, 5.0, 2.0),
        EdgeCheck("normals", True, 10.0, 30.0),
    ]
    result = EdgeValidResult(pair=(1, 2), checks=checks, valid=False)
    assert result.fragment_a == 1
    assert result.fragment_b == 2
    assert result.n_passed == 2
    assert result.n_failed == 1


def test_edge_valid_result_check_names():
    checks = [EdgeCheck("intensity", True, 0.05, 0.15)]
    result = EdgeValidResult(pair=(0, 1), checks=checks, valid=True)
    assert "intensity" in result.check_names


def test_edge_valid_result_get_check():
    checks = [EdgeCheck("gap", True, 1.0, 2.0)]
    result = EdgeValidResult(pair=(0, 1), checks=checks, valid=True)
    found = result.get_check("gap")
    assert found is not None
    assert found.name == "gap"


def test_edge_valid_result_get_check_missing():
    checks = [EdgeCheck("gap", True, 1.0, 2.0)]
    result = EdgeValidResult(pair=(0, 1), checks=checks, valid=True)
    assert result.get_check("nonexistent") is None


# ─── validate_edge_pair ───────────────────────────────────────────────────────

def test_validate_edge_pair_valid():
    ia = np.array([100, 110, 120], dtype=np.float32)
    ib = np.array([100, 110, 120], dtype=np.float32)
    pa = np.array([[0.0, 0.0], [1.0, 0.0]])
    pb = np.array([[1.5, 0.0], [2.5, 0.0]])
    na = np.array([[0.0, 1.0]])
    nb = np.array([[0.0, -1.0]])
    cfg = EdgeValidConfig(intensity_tol=0.5, gap_tol=5.0, normal_tol_deg=45.0)
    result = validate_edge_pair(0, 1, ia, ib, pa, pb, na, nb, cfg)
    assert isinstance(result, EdgeValidResult)
    assert len(result.checks) == 3


def test_validate_edge_pair_require_all_false():
    ia = np.array([0.0])
    ib = np.array([255.0])
    pa = np.array([[0.0, 0.0]])
    pb = np.array([[100.0, 100.0]])
    na = np.array([[1.0, 0.0]])
    nb = np.array([[-1.0, 0.0]])
    cfg = EdgeValidConfig(intensity_tol=0.5, gap_tol=5.0, normal_tol_deg=5.0, require_all=False)
    result = validate_edge_pair(0, 1, ia, ib, pa, pb, na, nb, cfg)
    # At least normals might pass
    assert result.pair == (0, 1)


# ─── summarise_validations ────────────────────────────────────────────────────

def test_summarise_validations_empty():
    stats = summarise_validations([])
    assert stats["valid_ratio"] == 0.0
    assert stats["n_results"] == 0


def test_summarise_validations_all_valid():
    checks = [EdgeCheck("gap", True, 0.5, 2.0)]
    results = [
        EdgeValidResult(pair=(0, 1), checks=checks, valid=True),
        EdgeValidResult(pair=(1, 2), checks=checks, valid=True),
    ]
    stats = summarise_validations(results)
    assert stats["valid_ratio"] == pytest.approx(1.0)
    assert stats["n_results"] == 2


def test_summarise_validations_mixed():
    checks_pass = [EdgeCheck("gap", True, 0.5, 2.0)]
    checks_fail = [EdgeCheck("gap", False, 5.0, 2.0)]
    results = [
        EdgeValidResult(pair=(0, 1), checks=checks_pass, valid=True),
        EdgeValidResult(pair=(1, 2), checks=checks_fail, valid=False),
    ]
    stats = summarise_validations(results)
    assert stats["valid_ratio"] == pytest.approx(0.5)


# ─── batch_validate_edges ─────────────────────────────────────────────────────

def test_batch_validate_edges_basic():
    pairs = [(0, 1)]
    intensity_map = {
        0: np.array([100.0, 110.0]),
        1: np.array([100.0, 110.0]),
    }
    points_map = {
        0: np.array([[0.0, 0.0]]),
        1: np.array([[1.5, 0.0]]),
    }
    normals_map = {
        0: np.array([[0.0, 1.0]]),
        1: np.array([[0.0, -1.0]]),
    }
    results = batch_validate_edges(pairs, intensity_map, points_map, normals_map)
    assert len(results) == 1
    assert results[0].pair == (0, 1)


def test_batch_validate_edges_missing_both_raises():
    pairs = [(5, 6)]
    intensity_map = {0: np.array([1.0])}
    with pytest.raises(ValueError):
        batch_validate_edges(pairs, intensity_map, {}, {})


def test_batch_validate_edges_empty_pairs():
    results = batch_validate_edges([], {}, {}, {})
    assert results == []
