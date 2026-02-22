"""
Юнит-тесты для модуля assembly/gamma_optimizer.py.

Тесты покрывают:
    - GammaEdgeModel.fit()       — MLE оценка параметров
    - GammaEdgeModel.log_likelihood() — корректность расчёта
    - GammaEdgeModel.pair_score()    — оценка пары краёв
    - gamma_optimizer()             — сборка завершается без ошибок
    - _fit_gamma_model()            — деградация при пустых данных
"""
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


# ─── Фикстуры ────────────────────────────────────────────────────────────

def _make_edge(edge_id: int, n_points: int = 32) -> EdgeSignature:
    """Создаёт синтетическую EdgeSignature с волнообразной кривой."""
    t = np.linspace(0, 2 * math.pi, n_points)
    curve = np.column_stack([t / (2 * math.pi), 0.05 * np.sin(t)])
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=curve,
        fd=1.3 + 0.05 * (edge_id % 3),
        css_vec=np.zeros(32),
        ifs_coeffs=np.zeros(8),
        length=float(n_points),
    )


def _make_fragment(frag_id: int, n_edges: int = 4) -> Fragment:
    """Создаёт фрагмент с синтетическими краями."""
    img = np.full((100, 80, 3), 255, dtype=np.uint8)
    mask = np.ones((100, 80), dtype=np.uint8)
    contour = np.array([[0,0],[80,0],[80,100],[0,100]], dtype=float)
    frag = Fragment(fragment_id=frag_id, image=img, mask=mask, contour=contour)
    start_id = frag_id * n_edges
    frag.edges = [_make_edge(start_id + i) for i in range(n_edges)]
    return frag


def _make_compat_entry(edge_i: EdgeSignature,
                        edge_j: EdgeSignature,
                        score: float = 0.8,
                        dtw_dist: float = 0.2) -> CompatEntry:
    return CompatEntry(
        edge_i=edge_i, edge_j=edge_j,
        score=score, dtw_dist=dtw_dist,
        css_sim=0.7, fd_diff=0.05, text_score=0.0,
    )


# ─── GammaEdgeModel ───────────────────────────────────────────────────────

class TestGammaEdgeModel:

    def test_default_parameters(self):
        model = GammaEdgeModel()
        assert model.k > 0
        assert model.theta > 0

    def test_fit_returns_self(self):
        model = GammaEdgeModel()
        data  = np.random.RandomState(0).gamma(2.0, 0.5, 200)
        result = model.fit(data)
        assert result is model

    def test_fit_estimates_reasonable_parameters(self):
        """После fit на данных из Gamma(2, 0.5) параметры должны быть близки."""
        rng  = np.random.RandomState(42)
        data = rng.gamma(shape=2.0, scale=0.5, size=1000)
        model = GammaEdgeModel().fit(data)
        assert 1.0 <= model.k <= 5.0,     f"k = {model.k} вне ожидаемого диапазона"
        assert 0.1 <= model.theta <= 2.0, f"theta = {model.theta}"

    def test_fit_with_zeros_does_not_crash(self):
        """Нулевые отклонения должны фильтроваться, а не вызывать ошибку."""
        data  = np.zeros(10)
        model = GammaEdgeModel(k=2.0, theta=0.5)
        model.fit(data)  # Не должно бросать исключение

    def test_fit_with_small_data_keeps_defaults(self):
        """Менее 5 точек — параметры остаются дефолтными."""
        model = GammaEdgeModel(k=3.0, theta=1.0)
        model.fit(np.array([0.1, 0.2, 0.3]))
        assert model.k == 3.0 and model.theta == 1.0

    def test_log_likelihood_negative(self):
        """Лог-правдоподобие должно быть ≤ 0."""
        model = GammaEdgeModel(k=2.0, theta=0.5)
        deviations = np.array([0.1, 0.2, 0.3, 0.5])
        ll = model.log_likelihood(deviations)
        assert ll <= 0.0

    def test_log_likelihood_higher_for_typical_values(self):
        """Типичные значения дают выше LL, чем нетипичные."""
        model = GammaEdgeModel(k=2.0, theta=0.5)
        model.fit(np.random.RandomState(0).gamma(2.0, 0.5, 500))
        typical  = model.log_likelihood(np.array([0.5, 0.8, 1.0, 0.7]))
        atypical = model.log_likelihood(np.array([50.0, 80.0, 100.0]))
        assert typical > atypical

    def test_pair_score_finite(self):
        """pair_score должен возвращать конечное число."""
        model = GammaEdgeModel(k=2.0, theta=0.3)
        edge_a = np.linspace(0, 1, 32).reshape(-1, 1) * np.array([1.0, 0.1])
        edge_b = edge_a + 0.05
        score = model.pair_score(edge_a, edge_b)
        assert math.isfinite(score)

    def test_pair_score_higher_for_closer_edges(self):
        """Близкие края должны иметь выше score (меньше по абс. значению LL)."""
        model = GammaEdgeModel(k=2.0, theta=0.5)
        n = 32
        t = np.linspace(0, 1, n)
        base = np.column_stack([t, np.zeros(n)])
        close  = base + np.random.RandomState(0).randn(n, 2) * 0.01
        far    = base + np.random.RandomState(0).randn(n, 2) * 2.0
        s_close = model.pair_score(base, close)
        s_far   = model.pair_score(base, far)
        assert s_close > s_far  # Близкие края → выше LL (ближе к 0)

    def test_pair_score_empty_edge(self):
        """Пустой массив краёв → -inf."""
        model = GammaEdgeModel()
        score = model.pair_score(np.empty((0, 2)), np.empty((0, 2)))
        assert score == -math.inf


# ─── _rotate_curve ────────────────────────────────────────────────────────

class TestRotateCurve:

    def test_identity_rotation(self):
        curve = np.array([[1.0, 0.0], [0.0, 1.0]])
        rotated = _rotate_curve(curve, 0.0)
        np.testing.assert_allclose(rotated, curve, atol=1e-10)

    def test_90_degree_rotation(self):
        curve = np.array([[1.0, 0.0]])
        rotated = _rotate_curve(curve, math.pi / 2)
        np.testing.assert_allclose(rotated, [[0.0, 1.0]], atol=1e-10)

    def test_180_degree_rotation(self):
        curve = np.array([[1.0, 2.0]])
        rotated = _rotate_curve(curve, math.pi)
        np.testing.assert_allclose(rotated, [[-1.0, -2.0]], atol=1e-10)

    def test_preserves_norm(self):
        """Поворот — изометрия: длины сохраняются."""
        curve = np.random.RandomState(0).randn(20, 2)
        norms_before = np.linalg.norm(curve, axis=1)
        norms_after  = np.linalg.norm(_rotate_curve(curve, 1.23), axis=1)
        np.testing.assert_allclose(norms_before, norms_after, atol=1e-10)


# ─── _fit_gamma_model ─────────────────────────────────────────────────────

class TestFitGammaModel:

    def test_with_entries(self):
        e1 = _make_edge(0)
        e2 = _make_edge(1)
        entry = _make_compat_entry(e1, e2, dtw_dist=0.3)
        model = _fit_gamma_model([entry] * 20)
        assert model.k > 0
        assert model.theta > 0

    def test_with_no_entries(self):
        """Пустой список → модель с дефолтными параметрами."""
        model = _fit_gamma_model([])
        assert model.k == 2.0
        assert model.theta == 0.5


# ─── gamma_optimizer ──────────────────────────────────────────────────────

class TestGammaOptimizer:

    def test_places_all_fragments(self):
        frags   = [_make_fragment(i) for i in range(4)]
        entries = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    e = _make_compat_entry(frags[i].edges[0], frags[j].edges[2])
                    entries.append(e)
        asm = gamma_optimizer(frags, entries, n_iter=100, seed=0)
        assert len(asm.placements) == 4

    def test_all_placements_finite(self):
        frags   = [_make_fragment(i) for i in range(3)]
        entries = [
            _make_compat_entry(frags[0].edges[0], frags[1].edges[2]),
            _make_compat_entry(frags[1].edges[0], frags[2].edges[2]),
        ]
        asm = gamma_optimizer(frags, entries, n_iter=50, seed=42)
        for pos, angle in asm.placements.values():
            assert np.all(np.isfinite(pos))
            assert math.isfinite(angle)

    def test_empty_fragments(self):
        asm = gamma_optimizer([], [], n_iter=10, seed=0)
        assert len(asm.placements) == 0

    def test_single_fragment(self):
        frag = _make_fragment(0)
        asm  = gamma_optimizer([frag], [], n_iter=10, seed=0)
        assert 0 in asm.placements

    def test_deterministic_with_seed(self):
        frags   = [_make_fragment(i) for i in range(4)]
        entries = [_make_compat_entry(frags[0].edges[0], frags[1].edges[2], score=0.9)]

        asm1 = gamma_optimizer(frags, entries, n_iter=100, seed=7)
        asm2 = gamma_optimizer(frags, entries, n_iter=100, seed=7)

        for fid in asm1.placements:
            pos1, ang1 = asm1.placements[fid]
            pos2, ang2 = asm2.placements[fid]
            np.testing.assert_array_equal(pos1, pos2)
            assert ang1 == ang2

    def test_with_init_assembly(self):
        """Инициализация существующей сборкой не ломает алгоритм."""
        frags = [_make_fragment(i) for i in range(3)]
        init_placements = {f.fragment_id: (np.array([i * 100.0, 0.0]), 0.0)
                           for i, f in enumerate(frags)}
        init_asm = Assembly(
            fragments=frags,
            placements=init_placements,
            compat_matrix=np.zeros((12, 12)),
        )
        entries = [_make_compat_entry(frags[0].edges[0], frags[1].edges[2])]
        asm = gamma_optimizer(frags, entries, n_iter=50, init_assembly=init_asm, seed=0)
        assert len(asm.placements) == 3
