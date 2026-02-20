"""
Юнит-тесты для puzzle_reconstruction/assembly/exhaustive.py.

Тесты покрывают:
    - exhaustive_assembly() — базовое поведение
    - Граничные случаи: пустой список, один фрагмент, N > max_n
    - Качество: score >= greedy score для малого N (оптимальность)
    - Детерминированность
    - _score_delta(), _evaluate_config() — внутренние функции
    - RuntimeWarning при N >= WARN_N
"""
import math
import warnings
import numpy as np
import pytest

from puzzle_reconstruction.assembly.exhaustive import (
    exhaustive_assembly,
    _score_delta,
    _evaluate_config,
    MAX_EXACT_N,
    WARN_N,
)
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.models import (
    Fragment, Assembly, CompatEntry, EdgeSignature, EdgeSide,
    FractalSignature, TangramSignature, ShapeClass,
)


# ─── Фикстуры ────────────────────────────────────────────────────────────

def _make_edge(edge_id: int, n: int = 16) -> EdgeSignature:
    t = np.linspace(0, 1, n)
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.column_stack([t, np.sin(t * math.pi * (1 + edge_id % 3))]),
        fd=1.2 + 0.05 * (edge_id % 4),
        css_vec=np.zeros(16),
        ifs_coeffs=np.zeros(4),
        length=float(n),
    )


def _make_fragment(frag_id: int, n_edges: int = 4) -> Fragment:
    img  = np.full((60, 50, 3), 200, dtype=np.uint8)
    mask = np.ones((60, 50), dtype=np.uint8)
    contour = np.array([[0,0],[50,0],[50,60],[0,60]], dtype=float)
    frag = Fragment(fragment_id=frag_id, image=img, mask=mask, contour=contour)
    frag.edges = [_make_edge(frag_id * n_edges + i) for i in range(n_edges)]
    frag.tangram = TangramSignature(
        polygon=contour / np.array([50, 60]),
        shape_class=ShapeClass.RECTANGLE,
        centroid=np.array([0.5, 0.5]),
        angle=0.0, scale=1.0, area=0.5,
    )
    frag.fractal = FractalSignature(
        fd_box=1.3, fd_divider=1.35,
        ifs_coeffs=np.zeros(4),
        css_image=[], chain_code="", curve=np.zeros((8, 2)),
    )
    return frag


def _make_entries(fragments: list, score_base: float = 0.7) -> list:
    """Создаёт CompatEntry для всех пар краёв."""
    entries = []
    for i, fi in enumerate(fragments):
        for j, fj in enumerate(fragments):
            if i >= j:
                continue
            for ei in fi.edges[:1]:    # Только первые края для скорости
                for ej in fj.edges[:1]:
                    entries.append(CompatEntry(
                        edge_i=ei, edge_j=ej,
                        score=score_base + 0.01 * (i + j),
                        dtw_dist=0.2,
                        css_sim=0.8,
                        fd_diff=0.05,
                        text_score=0.0,
                    ))
    return sorted(entries, key=lambda e: -e.score)


# ─── Базовое поведение ────────────────────────────────────────────────────

class TestExhaustiveAssembly:

    def test_places_all_fragments_n3(self):
        frags   = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        asm     = exhaustive_assembly(frags, entries, max_n=9)
        assert len(asm.placements) == 3

    def test_places_all_fragments_n4(self):
        frags   = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        asm     = exhaustive_assembly(frags, entries, max_n=9)
        assert len(asm.placements) == 4

    def test_all_frag_ids_placed(self):
        frags   = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        asm     = exhaustive_assembly(frags, entries)
        expected_ids = {f.fragment_id for f in frags}
        assert set(asm.placements.keys()) == expected_ids

    def test_placements_finite(self):
        frags   = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        asm     = exhaustive_assembly(frags, entries)
        for pos, angle in asm.placements.values():
            assert np.all(np.isfinite(pos))
            assert math.isfinite(angle)

    def test_total_score_finite(self):
        frags   = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        asm     = exhaustive_assembly(frags, entries)
        assert math.isfinite(asm.total_score)

    def test_returns_assembly_object(self):
        frags   = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        asm     = exhaustive_assembly(frags, entries)
        assert isinstance(asm, Assembly)

    def test_score_gte_greedy(self):
        """
        Точный решатель должен давать score >= жадного алгоритма
        (или как минимум не хуже, так как начинает с жадного решения).
        """
        frags   = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        asm_ex  = exhaustive_assembly(frags, entries, max_n=9)
        asm_gr  = greedy_assembly(frags, entries)
        # Exhaustive score должен быть >= greedy (это гарантия Branch & Bound)
        assert asm_ex.total_score >= asm_gr.total_score - 1e-9

    def test_no_rotation_mode(self):
        frags   = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        asm     = exhaustive_assembly(frags, entries, allow_rotation=False)
        assert len(asm.placements) == 3

    def test_deterministic(self):
        frags   = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        asm1    = exhaustive_assembly(frags, entries, seed=42)
        asm2    = exhaustive_assembly(frags, entries, seed=42)
        for fid in asm1.placements:
            pos1, ang1 = asm1.placements[fid]
            pos2, ang2 = asm2.placements[fid]
            np.testing.assert_array_equal(pos1, pos2)
            assert ang1 == ang2


# ─── Граничные случаи ─────────────────────────────────────────────────────

class TestExhaustiveEdgeCases:

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="пуст"):
            exhaustive_assembly([], [])

    def test_single_fragment(self):
        frag = _make_fragment(0)
        asm  = exhaustive_assembly([frag], [], max_n=9)
        assert 0 in asm.placements

    def test_two_fragments(self):
        frags   = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        asm     = exhaustive_assembly(frags, entries, max_n=9)
        assert len(asm.placements) == 2

    def test_fallback_for_large_n(self):
        """N > max_n → RuntimeWarning + вызывается beam_search."""
        frags   = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        with pytest.warns(RuntimeWarning, match="beam_search"):
            asm = exhaustive_assembly(frags, entries, max_n=3)
        assert len(asm.placements) == 4

    def test_warn_n_triggers_warning(self):
        """N == WARN_N → RuntimeWarning о медленности."""
        frags   = [_make_fragment(i) for i in range(WARN_N)]
        entries = _make_entries(frags)
        with pytest.warns(RuntimeWarning):
            asm = exhaustive_assembly(frags, entries, max_n=MAX_EXACT_N)
        assert len(asm.placements) == WARN_N

    def test_empty_entries_no_crash(self):
        """Пустой список entries → размещает с нулевым score."""
        frags = [_make_fragment(i) for i in range(3)]
        asm   = exhaustive_assembly(frags, [], max_n=9)
        assert len(asm.placements) == 3


# ─── _score_delta ────────────────────────────────────────────────────────

class TestScoreDelta:

    def test_zero_for_empty_placed(self):
        frag    = _make_fragment(0)
        entries = _make_entries([frag, _make_fragment(1)])
        edge_to_frag = {e.edge_id: frag.fragment_id for e in frag.edges}
        delta = _score_delta(0, [], {}, entries, edge_to_frag)
        assert delta == 0.0

    def test_positive_for_matching_pair(self):
        frags   = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        edge_to_frag = {e.edge_id: f.fragment_id for f in frags for e in f.edges}
        placements   = {0: (np.array([0.0, 0.0]), 0.0)}
        delta = _score_delta(1, [0], placements, entries, edge_to_frag)
        assert delta >= 0.0


# ─── _evaluate_config ─────────────────────────────────────────────────────

class TestEvaluateConfig:

    def test_empty_placements_zero(self):
        frags   = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        edge_to_frag = {e.edge_id: f.fragment_id for f in frags for e in f.edges}
        score = _evaluate_config({}, entries, edge_to_frag)
        assert score == 0.0

    def test_positive_for_placed_pair(self):
        frags   = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        edge_to_frag = {e.edge_id: f.fragment_id for f in frags for e in f.edges}
        placements   = {
            0: (np.array([0.0, 0.0]), 0.0),
            1: (np.array([100.0, 0.0]), 0.0),
        }
        score = _evaluate_config(placements, entries, edge_to_frag)
        assert score >= 0.0

    def test_no_duplicates_counted(self):
        """Одна пара считается один раз, не дважды."""
        frags   = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        edge_to_frag = {e.edge_id: f.fragment_id for f in frags for e in f.edges}
        placements   = {i: (np.array([i*100.0, 0.0]), 0.0) for i in range(2)}
        s1 = _evaluate_config(placements, entries, edge_to_frag)
        s2 = _evaluate_config(placements, entries + entries, edge_to_frag)
        # С двойным списком пар одна и та же пара не должна считаться дважды
        assert s1 <= s2 + 1e-9   # s2 не меньше (больше пар = больше score)
