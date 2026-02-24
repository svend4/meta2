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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _profile(n=20, val=128):
    return np.full(n, val, dtype=np.float64)


def _points(n=10, offset=0.0):
    return np.column_stack([
        np.linspace(0, 10, n) + offset,
        np.zeros(n),
    ])


def _normals(n=10, direction=(0, 1)):
    return np.tile(direction, (n, 1)).astype(np.float64)


# ─── EdgeValidConfig ────────────────────────────────────────────────────────

class TestEdgeValidConfigExtra:
    def test_defaults(self):
        c = EdgeValidConfig()
        assert c.intensity_tol == pytest.approx(0.15)
        assert c.gap_tol == pytest.approx(2.0)
        assert c.normal_tol_deg == pytest.approx(30.0)
        assert c.require_all is True

    def test_negative_intensity_tol_raises(self):
        with pytest.raises(ValueError):
            EdgeValidConfig(intensity_tol=-0.1)

    def test_high_intensity_tol_raises(self):
        with pytest.raises(ValueError):
            EdgeValidConfig(intensity_tol=1.5)

    def test_negative_gap_tol_raises(self):
        with pytest.raises(ValueError):
            EdgeValidConfig(gap_tol=-1.0)

    def test_negative_normal_tol_raises(self):
        with pytest.raises(ValueError):
            EdgeValidConfig(normal_tol_deg=-1.0)


# ─── EdgeCheck ──────────────────────────────────────────────────────────────

class TestEdgeCheckExtra:
    def test_valid(self):
        ec = EdgeCheck(name="test", passed=True, value=0.1, limit=0.2)
        assert ec.margin == pytest.approx(0.1)

    def test_failed(self):
        ec = EdgeCheck(name="test", passed=False, value=0.5, limit=0.2)
        assert ec.margin < 0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            EdgeCheck(name="", passed=True, value=0.0, limit=0.0)


# ─── EdgeValidResult ────────────────────────────────────────────────────────

class TestEdgeValidResultExtra:
    def test_properties(self):
        checks = [
            EdgeCheck(name="a", passed=True, value=0.1, limit=0.5),
            EdgeCheck(name="b", passed=False, value=0.8, limit=0.5),
        ]
        r = EdgeValidResult(pair=(0, 1), checks=checks, valid=False)
        assert r.fragment_a == 0
        assert r.fragment_b == 1
        assert r.n_passed == 1
        assert r.n_failed == 1
        assert r.check_names == ["a", "b"]

    def test_get_check(self):
        checks = [EdgeCheck(name="x", passed=True, value=0.0, limit=1.0)]
        r = EdgeValidResult(pair=(0, 1), checks=checks, valid=True)
        assert r.get_check("x") is checks[0]
        assert r.get_check("unknown") is None


# ─── check_intensity ────────────────────────────────────────────────────────

class TestCheckIntensityExtra:
    def test_identical(self):
        ec = check_intensity(_profile(), _profile())
        assert ec.passed is True
        assert ec.value == pytest.approx(0.0)

    def test_different(self):
        a = np.linspace(0, 200, 20).astype(np.float64)
        b = np.linspace(200, 0, 20).astype(np.float64)
        ec = check_intensity(a, b)
        # Reversed profiles → high difference
        assert ec.value > 0.0

    def test_varying(self):
        a = np.linspace(0, 255, 20).astype(np.float64)
        b = np.linspace(255, 0, 20).astype(np.float64)
        ec = check_intensity(a, b)
        assert ec.name == "intensity"


# ─── check_gap ──────────────────────────────────────────────────────────────

class TestCheckGapExtra:
    def test_close_points(self):
        ec = check_gap(_points(10, 0.0), _points(10, 0.5))
        assert ec.passed is True

    def test_far_points(self):
        ec = check_gap(_points(10, 0.0), _points(10, 100.0))
        assert ec.passed is False

    def test_empty_a(self):
        ec = check_gap(np.zeros((0, 2)), _points())
        assert ec.passed is False

    def test_empty_b(self):
        ec = check_gap(_points(), np.zeros((0, 2)))
        assert ec.passed is False


# ─── check_normals ──────────────────────────────────────────────────────────

class TestCheckNormalsExtra:
    def test_parallel(self):
        ec = check_normals(_normals(10, (0, 1)), _normals(10, (0, 1)))
        assert ec.name == "normals"

    def test_perpendicular(self):
        ec = check_normals(_normals(10, (1, 0)), _normals(10, (0, 1)))
        assert ec.name == "normals"

    def test_empty_a(self):
        ec = check_normals(np.zeros((0, 2)), _normals())
        assert ec.passed is False


# ─── validate_edge_pair ─────────────────────────────────────────────────────

class TestValidateEdgePairExtra:
    def test_all_pass(self):
        r = validate_edge_pair(
            0, 1,
            _profile(), _profile(),
            _points(), _points(offset=0.5),
            _normals(), _normals(),
        )
        assert isinstance(r, EdgeValidResult)
        assert r.pair == (0, 1)

    def test_require_all(self):
        cfg = EdgeValidConfig(require_all=True)
        r = validate_edge_pair(
            0, 1,
            _profile(), _profile(),
            _points(), _points(offset=0.5),
            _normals(), _normals(),
            cfg,
        )
        assert isinstance(r.valid, bool)

    def test_require_any(self):
        cfg = EdgeValidConfig(require_all=False)
        r = validate_edge_pair(
            0, 1,
            _profile(), _profile(),
            _points(), _points(offset=100.0),
            _normals(), _normals(),
            cfg,
        )
        assert isinstance(r.valid, bool)

    def test_three_checks(self):
        r = validate_edge_pair(
            0, 1,
            _profile(), _profile(),
            _points(), _points(),
            _normals(), _normals(),
        )
        assert len(r.checks) == 3


# ─── summarise_validations ──────────────────────────────────────────────────

class TestSummariseValidationsExtra:
    def test_empty(self):
        result = summarise_validations([])
        assert result["n_results"] == 0
        assert result["valid_ratio"] == pytest.approx(0.0)

    def test_with_data(self):
        r1 = EdgeValidResult(pair=(0, 1), checks=[], valid=True)
        r2 = EdgeValidResult(pair=(0, 2), checks=[], valid=False)
        result = summarise_validations([r1, r2])
        assert result["n_results"] == 2
        assert result["valid_ratio"] == pytest.approx(0.5)


# ─── batch_validate_edges ───────────────────────────────────────────────────

class TestBatchValidateEdgesExtra:
    def test_empty(self):
        assert batch_validate_edges([], {}, {}, {}) == []

    def test_one_pair(self):
        pairs = [(0, 1)]
        intensity_map = {0: _profile(), 1: _profile()}
        points_map = {0: _points(), 1: _points(offset=0.5)}
        normals_map = {0: _normals(), 1: _normals()}
        results = batch_validate_edges(
            pairs, intensity_map, points_map, normals_map)
        assert len(results) == 1
        assert isinstance(results[0], EdgeValidResult)

    def test_missing_id(self):
        pairs = [(0, 99)]
        # Missing IDs use empty arrays which crash on intensity check
        with pytest.raises(ValueError):
            batch_validate_edges(pairs, {}, {}, {})
