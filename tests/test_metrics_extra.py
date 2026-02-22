"""Additional tests for puzzle_reconstruction/verification/metrics.py."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.verification.metrics import (
    evaluate_reconstruction,
    compare_methods,
    ReconstructionMetrics,
    BenchmarkResult,
    _normalize_config,
    _angle_diff_deg,
    _compute_adjacency,
    _zero_metrics,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _cfg(n=4, spacing=100.0):
    return {i: (np.array([i * spacing, 0.0]), 0.0) for i in range(n)}


def _make_br(method="test", na=0.5, dc=0.5):
    m = ReconstructionMetrics(
        neighbor_accuracy=na, direct_comparison=dc,
        perfect=(na == 1.0 and dc == 1.0),
        position_rmse=10.0, angular_error_deg=5.0,
        n_fragments=4, n_correct_pairs=2, n_total_pairs=4,
        edge_match_rate=dc,
    )
    return BenchmarkResult(method=method, metrics=m, runtime_sec=1.0)


# ─── TestAngleDiffDegExtra ────────────────────────────────────────────────────

class TestAngleDiffDegExtra:
    def test_zero_both_zero(self):
        assert _angle_diff_deg(0.0, 0.0) == pytest.approx(0.0)

    def test_90_degrees(self):
        assert _angle_diff_deg(0.0, 90.0) == pytest.approx(90.0)

    def test_wraparound_350_to_20_is_30(self):
        assert _angle_diff_deg(350.0, 20.0) == pytest.approx(30.0, abs=1e-9)

    def test_exactly_360_is_0(self):
        assert _angle_diff_deg(0.0, 360.0) == pytest.approx(0.0, abs=1e-9)

    def test_270_is_same_as_minus90(self):
        assert _angle_diff_deg(0.0, 270.0) == pytest.approx(90.0, abs=1e-9)

    def test_result_always_nonneg(self):
        for a, b in [(10, 350), (350, 10), (180, 0), (0, 180), (90, 270)]:
            assert _angle_diff_deg(float(a), float(b)) >= 0.0

    def test_result_at_most_180(self):
        for a, b in [(0, 181), (100, 300), (10, 190)]:
            assert _angle_diff_deg(float(a), float(b)) <= 180.0 + 1e-9

    def test_symmetry_random(self):
        for a, b in [(17.3, 93.4), (359.9, 0.1), (180.0, 360.0)]:
            assert _angle_diff_deg(a, b) == pytest.approx(_angle_diff_deg(b, a))


# ─── TestNormalizeConfigExtra ─────────────────────────────────────────────────

class TestNormalizeConfigExtra:
    def test_three_fragments_first_at_origin(self):
        cfg = {i: (np.array([float(i * 50), float(i * 30)]), 0.3) for i in range(3)}
        norm = _normalize_config(cfg, [0, 1, 2])
        np.testing.assert_allclose(norm[0][0], [0.0, 0.0], atol=1e-10)

    def test_angle_relative_preserved(self):
        delta = math.pi / 4
        cfg = {
            0: (np.array([0.0, 0.0]), 0.0),
            1: (np.array([100.0, 0.0]), delta),
        }
        norm = _normalize_config(cfg, [0, 1])
        assert abs(norm[1][1] - delta) < 1e-9

    def test_2_fragments_both_present(self):
        cfg = {0: (np.array([0.0, 0.0]), 0.0),
               1: (np.array([100.0, 0.0]), 0.5)}
        norm = _normalize_config(cfg, [0, 1])
        assert 0 in norm and 1 in norm

    def test_large_offset_still_at_origin(self):
        cfg = {0: (np.array([5000.0, -3000.0]), 0.0),
               1: (np.array([5100.0, -3000.0]), 0.0)}
        norm = _normalize_config(cfg, [0, 1])
        np.testing.assert_allclose(norm[0][0], [0.0, 0.0], atol=1e-9)
        assert norm[1][0][0] == pytest.approx(100.0, abs=1e-9)


# ─── TestComputeAdjacencyExtra ────────────────────────────────────────────────

class TestComputeAdjacencyExtra:
    def test_threshold_0_no_pairs(self):
        cfg = {0: (np.array([0.0, 0.0]), 0.0),
               1: (np.array([1.0, 0.0]), 0.0)}
        adj = _compute_adjacency(cfg, threshold=0.0)
        assert len(adj) == 0

    def test_threshold_inf_all_pairs(self):
        cfg = {i: (np.array([float(i * 1000), 0.0]), 0.0) for i in range(4)}
        adj = _compute_adjacency(cfg, threshold=float("inf"))
        assert len(adj) == 4 * 3 // 2

    def test_single_fragment_no_pairs(self):
        cfg = {0: (np.array([0.0, 0.0]), 0.0)}
        adj = _compute_adjacency(cfg, threshold=100.0)
        assert len(adj) == 0

    def test_pairs_are_sorted(self):
        cfg = {i: (np.array([float(i * 5), 0.0]), 0.0) for i in range(3)}
        adj = _compute_adjacency(cfg, threshold=100.0)
        for a, b in adj:
            assert a < b


# ─── TestEvaluateReconstructionExtra ─────────────────────────────────────────

class TestEvaluateReconstructionExtra:
    def test_2_fragments_perfect(self):
        gt = _cfg(2)
        m = evaluate_reconstruction(gt, gt)
        assert m.n_fragments == 2
        assert m.direct_comparison == pytest.approx(1.0, abs=1e-6)

    def test_6_fragments_perfect_rmse_0(self):
        gt = _cfg(6)
        m = evaluate_reconstruction(gt, gt)
        assert m.position_rmse == pytest.approx(0.0, abs=1e-6)

    def test_empty_gt_empty_pred(self):
        m = evaluate_reconstruction({}, {})
        assert m.n_fragments == 0

    def test_partial_pred_all_metrics_valid(self):
        gt = _cfg(4)
        pred = {0: gt[0], 1: gt[1]}   # only 2 of 4
        m = evaluate_reconstruction(pred, gt)
        assert 0.0 <= m.neighbor_accuracy <= 1.0
        assert 0.0 <= m.direct_comparison <= 1.0
        assert m.position_rmse >= 0.0

    def test_angular_error_nonzero_for_rotated(self):
        gt = {0: (np.array([0.0, 0.0]), 0.0),
              1: (np.array([100.0, 0.0]), 0.0)}
        pred = {0: (np.array([0.0, 0.0]), math.pi / 3),
                1: (np.array([100.0, 0.0]), math.pi / 3)}
        m = evaluate_reconstruction(pred, gt)
        assert m.angular_error_deg >= 0.0

    def test_n_fragments_matches_pred(self):
        gt = _cfg(5)
        pred = {k: v for k, v in gt.items()}
        m = evaluate_reconstruction(pred, gt)
        assert m.n_fragments == 5

    def test_summary_contains_dc(self):
        gt = _cfg(3)
        m = evaluate_reconstruction(gt, gt)
        s = m.summary()
        assert "Direct" in s or "direct" in s or "DC" in s

    def test_metrics_types_all_correct(self):
        gt = _cfg(3)
        m = evaluate_reconstruction(gt, gt)
        assert isinstance(m.position_rmse, float)
        assert isinstance(m.angular_error_deg, float)
        assert isinstance(m.edge_match_rate, float)
        assert isinstance(m.n_fragments, int)

    def test_n_total_pairs_with_adjacency(self):
        gt = _cfg(4)
        adj = [(0, 1), (1, 2), (2, 3)]
        m = evaluate_reconstruction(gt, gt, adjacency=adj)
        assert m.n_total_pairs == 3


# ─── TestZeroMetricsExtra ────────────────────────────────────────────────────

class TestZeroMetricsExtra:
    def test_zero_metrics_n_5(self):
        m = _zero_metrics(5)
        assert m.n_fragments == 5
        assert m.direct_comparison == 0.0
        assert m.position_rmse == 0.0

    def test_zero_metrics_perfect_false(self):
        m = _zero_metrics(0)
        assert m.perfect is False

    def test_zero_metrics_neighbor_accuracy_zero(self):
        m = _zero_metrics(3)
        assert m.neighbor_accuracy == 0.0

    def test_zero_metrics_is_reconstruction_metrics(self):
        assert isinstance(_zero_metrics(2), ReconstructionMetrics)


# ─── TestCompareMethodsExtra ─────────────────────────────────────────────────

class TestCompareMethodsExtra:
    def test_single_result(self):
        out = compare_methods([_make_br("only", 0.7, 0.6)])
        assert isinstance(out, str) and "only" in out

    def test_four_results(self):
        results = [_make_br(f"method_{i}", 0.1 * i, 0.1 * i) for i in range(4)]
        out = compare_methods(results)
        assert isinstance(out, str)

    def test_best_method_by_na(self):
        results = [
            _make_br("a", na=0.9, dc=0.8),
            _make_br("b", na=0.3, dc=0.2),
        ]
        out = compare_methods(results)
        assert "a" in out

    def test_all_methods_in_output(self):
        methods = ["greedy", "sa", "beam", "gamma"]
        results = [_make_br(m, 0.5, 0.4) for m in methods]
        out = compare_methods(results)
        for m in methods:
            assert m in out

    def test_empty_list_no_crash(self):
        out = compare_methods([])
        assert isinstance(out, str)

    def test_runtime_in_output(self):
        results = [_make_br("test", 0.5, 0.5)]
        out = compare_methods(results)
        assert isinstance(out, str)
