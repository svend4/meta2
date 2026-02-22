"""Additional tests for puzzle_reconstruction/assembly/gamma_optimizer.py."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.assembly.gamma_optimizer import (
    GammaEdgeModel,
    gamma_optimizer,
    _fit_gamma_model,
    _rotate_curve,
)
from puzzle_reconstruction.models import (
    Assembly, CompatEntry, EdgeSide, EdgeSignature, Fragment,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_edge(edge_id: int, n_points: int = 32) -> EdgeSignature:
    t = np.linspace(0, 2 * math.pi, n_points)
    curve = np.column_stack([t / (2 * math.pi), 0.05 * np.sin(t)])
    return EdgeSignature(
        edge_id=edge_id, side=EdgeSide.TOP,
        virtual_curve=curve,
        fd=1.3 + 0.05 * (edge_id % 3),
        css_vec=np.zeros(32),
        ifs_coeffs=np.zeros(8),
        length=float(n_points),
    )


def _make_fragment(frag_id: int, n_edges: int = 4) -> Fragment:
    img = np.full((100, 80, 3), 255, dtype=np.uint8)
    mask = np.ones((100, 80), dtype=np.uint8)
    contour = np.array([[0, 0], [80, 0], [80, 100], [0, 100]], dtype=float)
    frag = Fragment(fragment_id=frag_id, image=img, mask=mask, contour=contour)
    start = frag_id * n_edges
    frag.edges = [_make_edge(start + i) for i in range(n_edges)]
    return frag


def _make_entry(ei: EdgeSignature, ej: EdgeSignature,
                score: float = 0.8, dtw_dist: float = 0.2) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej,
        score=score, dtw_dist=dtw_dist,
        css_sim=0.7, fd_diff=0.05, text_score=0.0,
    )


# ─── TestGammaEdgeModelExtra ──────────────────────────────────────────────────

class TestGammaEdgeModelExtra:
    def test_default_k_positive(self):
        assert GammaEdgeModel().k > 0.0

    def test_default_theta_positive(self):
        assert GammaEdgeModel().theta > 0.0

    def test_custom_k_theta(self):
        m = GammaEdgeModel(k=3.0, theta=1.5)
        assert m.k == 3.0 and m.theta == 1.5

    def test_fit_returns_self_type(self):
        m = GammaEdgeModel()
        data = np.random.default_rng(1).gamma(2.0, 0.5, 200)
        assert isinstance(m.fit(data), GammaEdgeModel)

    def test_fit_varied_data_positive_params(self):
        rng = np.random.default_rng(99)
        data = rng.gamma(3.0, 0.3, 500)
        m = GammaEdgeModel().fit(data)
        assert m.k > 0 and m.theta > 0

    def test_fit_large_dataset(self):
        rng = np.random.default_rng(7)
        data = rng.gamma(5.0, 0.2, 2000)
        m = GammaEdgeModel().fit(data)
        assert m.k > 0 and m.theta > 0

    def test_log_likelihood_is_float(self):
        m = GammaEdgeModel(k=2.0, theta=0.5)
        ll = m.log_likelihood(np.array([0.2, 0.4, 0.6]))
        assert isinstance(ll, float)

    def test_log_likelihood_single_point(self):
        m = GammaEdgeModel(k=2.0, theta=0.5)
        ll = m.log_likelihood(np.array([0.5]))
        assert isinstance(ll, float) and ll <= 0.0

    def test_log_likelihood_large_deviations_very_negative(self):
        m = GammaEdgeModel(k=2.0, theta=0.3)
        ll_small = m.log_likelihood(np.array([0.3, 0.5]))
        ll_large = m.log_likelihood(np.array([100.0, 200.0]))
        assert ll_small > ll_large

    def test_pair_score_identical_curves_finite(self):
        m = GammaEdgeModel(k=2.0, theta=0.5)
        curve = np.column_stack([np.linspace(0, 1, 32), np.zeros(32)])
        score = m.pair_score(curve, curve.copy())
        assert math.isfinite(score)

    def test_pair_score_very_far_negative(self):
        m = GammaEdgeModel(k=2.0, theta=0.3)
        a = np.column_stack([np.linspace(0, 1, 32), np.zeros(32)])
        b = a + 1000.0
        score = m.pair_score(a, b)
        assert score < 0.0

    def test_pair_score_1d_curve_no_crash(self):
        m = GammaEdgeModel(k=2.0, theta=0.5)
        a = np.arange(16, dtype=float).reshape(-1, 1)
        b = a + 0.1
        score = m.pair_score(a, b)
        assert math.isfinite(score)

    def test_pair_score_type_float(self):
        m = GammaEdgeModel()
        a = np.random.default_rng(3).random((20, 2))
        b = np.random.default_rng(4).random((20, 2))
        assert isinstance(m.pair_score(a, b), float)


# ─── TestRotateCurveExtra ─────────────────────────────────────────────────────

class TestRotateCurveExtra:
    def test_full_360_returns_original(self):
        curve = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        rotated = _rotate_curve(curve, 2 * math.pi)
        np.testing.assert_allclose(rotated, curve, atol=1e-9)

    def test_arbitrary_angle_preserves_norm(self):
        rng = np.random.default_rng(17)
        curve = rng.standard_normal((25, 2))
        norms_before = np.linalg.norm(curve, axis=1)
        norms_after = np.linalg.norm(_rotate_curve(curve, 0.753), axis=1)
        np.testing.assert_allclose(norms_before, norms_after, atol=1e-10)

    def test_negative_angle_norm_preserved(self):
        curve = np.array([[3.0, 4.0], [0.0, 5.0]])
        r = _rotate_curve(curve, -0.5)
        norms_before = np.linalg.norm(curve, axis=1)
        norms_after = np.linalg.norm(r, axis=1)
        np.testing.assert_allclose(norms_before, norms_after, atol=1e-10)

    def test_270_rotation_correct(self):
        # 270 deg = -90 deg: (1,0) → (0,-1)
        curve = np.array([[1.0, 0.0]])
        r = _rotate_curve(curve, 3 * math.pi / 2)
        np.testing.assert_allclose(r, [[0.0, -1.0]], atol=1e-10)

    def test_output_shape_preserved(self):
        curve = np.random.default_rng(2).random((50, 2))
        assert _rotate_curve(curve, 1.1).shape == (50, 2)

    def test_zero_angle_identity(self):
        curve = np.array([[2.0, 3.0], [-1.0, 4.0]])
        np.testing.assert_allclose(_rotate_curve(curve, 0.0), curve, atol=1e-12)

    def test_single_point_rotation(self):
        curve = np.array([[5.0, 0.0]])
        r = _rotate_curve(curve, math.pi / 2)
        np.testing.assert_allclose(r, [[0.0, 5.0]], atol=1e-10)


# ─── TestFitGammaModelExtra ───────────────────────────────────────────────────

class TestFitGammaModelExtra:
    def test_single_entry_varied(self):
        e1 = _make_edge(0)
        e2 = _make_edge(1)
        entry = _make_entry(e1, e2, dtw_dist=0.5)
        # Single entry: not enough data to fit → defaults kept
        model = _fit_gamma_model([entry])
        assert model.k > 0 and model.theta > 0

    def test_varied_dtw_entries(self):
        e1 = _make_edge(0)
        e2 = _make_edge(1)
        entries = [
            _make_entry(e1, e2, dtw_dist=float(i) * 0.1 + 0.01)
            for i in range(30)
        ]
        model = _fit_gamma_model(entries)
        assert model.k > 0 and model.theta > 0

    def test_returns_gamma_model(self):
        model = _fit_gamma_model([])
        assert isinstance(model, GammaEdgeModel)

    def test_default_params_on_empty(self):
        model = _fit_gamma_model([])
        assert model.k == 2.0 and model.theta == 0.5


# ─── TestGammaOptimizerExtra ──────────────────────────────────────────────────

class TestGammaOptimizerExtra:
    def test_two_fragments_both_placed(self):
        frags = [_make_fragment(i) for i in range(2)]
        entries = [_make_entry(frags[0].edges[0], frags[1].edges[2])]
        asm = gamma_optimizer(frags, entries, n_iter=50, seed=0)
        assert len(asm.placements) == 2

    def test_placements_contains_all_fragment_ids(self):
        n = 5
        frags = [_make_fragment(i) for i in range(n)]
        entries = [
            _make_entry(frags[i].edges[0], frags[(i + 1) % n].edges[2])
            for i in range(n)
        ]
        asm = gamma_optimizer(frags, entries, n_iter=50, seed=1)
        assert set(asm.placements.keys()) == {f.fragment_id for f in frags}

    def test_placements_positions_2d(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = [_make_entry(frags[0].edges[0], frags[1].edges[2])]
        asm = gamma_optimizer(frags, entries, n_iter=30, seed=0)
        for pos, _ in asm.placements.values():
            assert pos.shape == (2,)

    def test_placements_angles_finite(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = [_make_entry(frags[0].edges[0], frags[1].edges[2])]
        asm = gamma_optimizer(frags, entries, n_iter=30, seed=0)
        for _, angle in asm.placements.values():
            assert math.isfinite(angle)

    def test_no_entries_still_places_all(self):
        frags = [_make_fragment(i) for i in range(4)]
        asm = gamma_optimizer(frags, [], n_iter=20, seed=0)
        assert len(asm.placements) == 4

    def test_returns_assembly_type(self):
        frags = [_make_fragment(0)]
        asm = gamma_optimizer(frags, [], n_iter=10, seed=0)
        assert isinstance(asm, Assembly)

    def test_different_seeds_may_differ(self):
        """Two different seeds can produce different placements."""
        frags = [_make_fragment(i) for i in range(4)]
        entries = [_make_entry(frags[i].edges[0], frags[(i + 1) % 4].edges[2])
                   for i in range(4)]
        asm1 = gamma_optimizer(frags, entries, n_iter=200, seed=1)
        asm2 = gamma_optimizer(frags, entries, n_iter=200, seed=2)
        # Both should place all fragments
        assert len(asm1.placements) == 4
        assert len(asm2.placements) == 4

    def test_n_iter_1_completes(self):
        frags = [_make_fragment(i) for i in range(2)]
        entries = [_make_entry(frags[0].edges[0], frags[1].edges[2])]
        asm = gamma_optimizer(frags, entries, n_iter=1, seed=0)
        assert len(asm.placements) == 2
