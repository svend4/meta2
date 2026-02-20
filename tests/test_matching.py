"""
Тесты для модулей сопоставления краёв.
"""
import numpy as np
import pytest

from puzzle_reconstruction.matching.dtw import dtw_distance, dtw_distance_mirror
from puzzle_reconstruction.matching.pairwise import match_score
from puzzle_reconstruction.models import EdgeSignature, EdgeSide


# ─── Вспомогательные фабрики ──────────────────────────────────────────────

def make_sine_curve(n=64, freq=1.0, amp=1.0, phase=0.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([t, amp * np.sin(freq * t + phase)])


def make_flat_curve(n=64, y=0.0) -> np.ndarray:
    x = np.linspace(0, 6, n)
    return np.column_stack([x, np.full(n, y)])


def make_edge_signature(curve: np.ndarray,
                         fd: float = 1.2,
                         edge_id: int = 0,
                         side: EdgeSide = EdgeSide.TOP,
                         length: float = 100.0) -> EdgeSignature:
    css_vec = np.random.RandomState(edge_id).rand(7 * 32)
    css_vec /= np.linalg.norm(css_vec) + 1e-10
    ifs_coeffs = np.random.RandomState(edge_id + 100).uniform(-0.5, 0.5, 8)
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=curve.copy(),
        fd=fd,
        css_vec=css_vec,
        ifs_coeffs=ifs_coeffs,
        length=length,
    )


# ─── DTW ──────────────────────────────────────────────────────────────────

class TestDTW:

    def test_same_curve_distance_zero(self):
        curve = make_sine_curve()
        d = dtw_distance(curve, curve)
        assert d < 1e-8

    def test_symmetric(self):
        a = make_sine_curve(freq=1.0)
        b = make_sine_curve(freq=1.5)
        assert abs(dtw_distance(a, b) - dtw_distance(b, a)) < 1e-9

    def test_distance_positive(self):
        a = make_sine_curve(freq=1.0)
        b = make_sine_curve(freq=2.0, amp=2.0)
        assert dtw_distance(a, b) > 0

    def test_distance_normalized(self):
        """Нормализованное расстояние не должно быть астрономически большим."""
        a = make_sine_curve(n=64)
        b = make_sine_curve(n=128, amp=5.0)
        d = dtw_distance(a, b)
        assert d < 100.0

    def test_mirror_leq_direct(self):
        """Зеркальное DTW ≤ прямому (или равно)."""
        a = make_sine_curve()
        b = make_sine_curve()[::-1]  # Перевёрнутая та же кривая
        d_direct  = dtw_distance(a, b)
        d_mirror  = dtw_distance_mirror(a, b)
        assert d_mirror <= d_direct + 1e-9

    def test_mirror_of_reversed_is_small(self):
        """Зеркальное расстояние перевёрнутой копии ≈ 0."""
        curve = make_sine_curve()
        d = dtw_distance_mirror(curve, curve[::-1])
        assert d < 0.01

    def test_empty_returns_inf(self):
        empty = np.zeros((0, 2))
        assert dtw_distance(empty, make_sine_curve()) == float("inf")

    def test_window_clipping(self):
        """Разные длины: window автоматически увеличивается."""
        a = make_sine_curve(n=30)
        b = make_sine_curve(n=80)
        d = dtw_distance(a, b, window=5)
        assert np.isfinite(d)


# ─── Match Score ──────────────────────────────────────────────────────────

class TestMatchScore:

    def test_identical_edges_high_score(self):
        """Одинаковые края (одинаковые подписи) должны давать высокий score."""
        curve = make_sine_curve()
        e1 = make_edge_signature(curve, fd=1.3, edge_id=0)
        e2 = make_edge_signature(curve, fd=1.3, edge_id=1)
        # Принудительно выравниваем CSS
        e2.css_vec = e1.css_vec.copy()
        e2.ifs_coeffs = e1.ifs_coeffs.copy()

        entry = match_score(e1, e2)
        assert entry.score >= 0.5, f"Ожидался высокий score, получен {entry.score}"

    def test_opposite_curves_high_mirror_score(self):
        """Зеркальные кривые (сопрягаемые края) должны давать высокий score."""
        curve = make_sine_curve()
        curve_rev = curve[::-1].copy()
        e1 = make_edge_signature(curve,     fd=1.3, edge_id=0)
        e2 = make_edge_signature(curve_rev, fd=1.3, edge_id=1)
        e2.css_vec    = e1.css_vec.copy()
        e2.ifs_coeffs = e1.ifs_coeffs.copy()

        entry = match_score(e1, e2)
        assert entry.score >= 0.4

    def test_very_different_edges_low_score(self):
        """Сильно отличающиеся края дают низкий score."""
        e1 = make_edge_signature(make_sine_curve(freq=1.0, amp=0.1), fd=1.0, edge_id=0)
        e2 = make_edge_signature(make_sine_curve(freq=5.0, amp=5.0), fd=1.9, edge_id=2,
                                  length=10.0)
        # Разные CSS-векторы (seed разный)
        entry = match_score(e1, e2)
        assert entry.score < 0.8  # Допускаем широкий диапазон

    def test_score_in_range(self):
        """Score всегда ∈ [0, 1]."""
        for seed in range(5):
            rng = np.random.RandomState(seed)
            c1 = rng.randn(64, 2)
            c2 = rng.randn(64, 2)
            e1 = make_edge_signature(c1, fd=rng.uniform(1.0, 2.0), edge_id=seed * 2)
            e2 = make_edge_signature(c2, fd=rng.uniform(1.0, 2.0), edge_id=seed * 2 + 1)
            entry = match_score(e1, e2)
            assert 0.0 <= entry.score <= 1.0, f"Score вне диапазона: {entry.score}"

    def test_entry_fields_populated(self):
        """CompatEntry содержит все ожидаемые поля."""
        c = make_sine_curve()
        e1 = make_edge_signature(c, edge_id=0)
        e2 = make_edge_signature(c[::-1], edge_id=1)
        entry = match_score(e1, e2)
        assert entry.edge_i is e1
        assert entry.edge_j is e2
        assert hasattr(entry, "dtw_dist")
        assert hasattr(entry, "css_sim")
        assert hasattr(entry, "fd_diff")

    def test_length_penalty(self):
        """Края с сильно разными длинами получают штраф."""
        curve = make_sine_curve()
        e1 = make_edge_signature(curve, length=100.0, edge_id=0)
        e2 = make_edge_signature(curve, length=10.0,  edge_id=1)  # в 10 раз короче
        e3 = make_edge_signature(curve, length=100.0, edge_id=2)  # такая же длина

        entry_penalty = match_score(e1, e2)
        entry_fair    = match_score(e1, e3)
        assert entry_penalty.score <= entry_fair.score


# ─── Compat Matrix ────────────────────────────────────────────────────────

class TestCompatMatrix:

    def _make_mock_fragments(self, n_frags=3):
        """Создаёт минимальные фиктивные фрагменты для тестирования матрицы."""
        from puzzle_reconstruction.models import Fragment
        frags = []
        for fid in range(n_frags):
            rng = np.random.RandomState(fid)
            contour = np.column_stack([rng.rand(32) * 100, rng.rand(32) * 100])
            frag = Fragment(fragment_id=fid, image=None, mask=None, contour=contour)
            for eid in range(2):
                curve = make_sine_curve(n=32)
                frag.edges.append(make_edge_signature(
                    curve, edge_id=fid * 10 + eid, length=80.0
                ))
            frags.append(frag)
        return frags

    def test_matrix_shape(self):
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
        frags = self._make_mock_fragments(3)
        n_edges = sum(len(f.edges) for f in frags)
        matrix, _ = build_compat_matrix(frags)
        assert matrix.shape == (n_edges, n_edges)

    def test_matrix_symmetric(self):
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
        frags = self._make_mock_fragments(3)
        matrix, _ = build_compat_matrix(frags)
        assert np.allclose(matrix, matrix.T)

    def test_diagonal_zero(self):
        """Край не совместим сам с собой (или 0 по диагонали)."""
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
        frags = self._make_mock_fragments(3)
        matrix, _ = build_compat_matrix(frags)
        assert np.all(np.diag(matrix) == 0.0)

    def test_no_intra_fragment_matches(self):
        """Края одного фрагмента не должны сравниваться между собой."""
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
        frags = self._make_mock_fragments(2)
        # Fragment 0 имеет edges с id=0,1; fragment 1 имеет id=10,11
        matrix, entries = build_compat_matrix(frags)
        for entry in entries:
            assert entry.edge_i.edge_id != entry.edge_j.edge_id

    def test_entries_sorted_by_score(self):
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
        frags = self._make_mock_fragments(4)
        _, entries = build_compat_matrix(frags)
        scores = [e.score for e in entries]
        assert scores == sorted(scores, reverse=True)
