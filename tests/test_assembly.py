"""
Тесты для алгоритмов сборки: greedy, SA, beam search.
"""
import numpy as np
import pytest

from puzzle_reconstruction.models import Fragment, Assembly, CompatEntry, EdgeSignature, EdgeSide
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.assembly.annealing import simulated_annealing
from puzzle_reconstruction.assembly.beam_search import beam_search


# ─── Фабрики тестовых данных ──────────────────────────────────────────────

def _make_fragment(fid: int, n_edges: int = 4) -> Fragment:
    rng = np.random.RandomState(fid)
    contour = np.column_stack([rng.rand(32) * 100, rng.rand(32) * 100])
    frag = Fragment(fragment_id=fid, image=None, mask=None, contour=contour)
    for eid in range(n_edges):
        t = np.linspace(0, 2 * np.pi, 64)
        curve = np.column_stack([t, rng.randn(64) * 0.5])
        css_vec = rng.rand(7 * 32)
        css_vec /= np.linalg.norm(css_vec) + 1e-10
        frag.edges.append(EdgeSignature(
            edge_id=fid * 100 + eid,
            side=list(EdgeSide)[eid % 4],
            virtual_curve=curve,
            fd=rng.uniform(1.1, 1.5),
            css_vec=css_vec,
            ifs_coeffs=rng.uniform(-0.5, 0.5, 8),
            length=80.0 + rng.rand() * 20,
        ))
    return frag


def _make_entries(fragments, n_entries=20) -> list[CompatEntry]:
    """Генерирует случайные записи о совместимости."""
    rng = np.random.RandomState(999)
    all_edges = [(e, f) for f in fragments for e in f.edges]
    entries = []
    for _ in range(n_entries):
        i, j = rng.choice(len(all_edges), 2, replace=False)
        e_i, f_i = all_edges[i]
        e_j, f_j = all_edges[j]
        if f_i.fragment_id == f_j.fragment_id:
            continue
        entries.append(CompatEntry(
            edge_i=e_i, edge_j=e_j,
            score=float(rng.uniform(0.3, 0.95)),
            dtw_dist=float(rng.rand()),
            css_sim=float(rng.rand()),
            fd_diff=float(rng.rand() * 0.3),
            text_score=0.5,
        ))
    entries.sort(key=lambda e: e.score, reverse=True)
    return entries


# ─── Greedy Assembly ──────────────────────────────────────────────────────

class TestGreedyAssembly:

    def test_all_fragments_placed(self):
        frags   = [_make_fragment(i) for i in range(5)]
        entries = _make_entries(frags, n_entries=30)
        asm     = greedy_assembly(frags, entries)
        assert len(asm.placements) == len(frags)

    def test_placement_ids_match_fragments(self):
        frags   = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        asm     = greedy_assembly(frags, entries)
        frag_ids = {f.fragment_id for f in frags}
        assert set(asm.placements.keys()) == frag_ids

    def test_placement_has_position_and_angle(self):
        frags   = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        asm     = greedy_assembly(frags, entries)
        for pos, angle in asm.placements.values():
            assert isinstance(pos, np.ndarray)
            assert pos.shape == (2,)
            assert isinstance(angle, float)

    def test_empty_input(self):
        asm = greedy_assembly([], [])
        assert asm.placements == {}

    def test_single_fragment(self):
        frags = [_make_fragment(0)]
        asm   = greedy_assembly(frags, [])
        assert 0 in asm.placements

    def test_no_entries_still_places_all(self):
        """Даже без совпадений все фрагменты должны быть размещены."""
        frags = [_make_fragment(i) for i in range(4)]
        asm   = greedy_assembly(frags, [])
        assert len(asm.placements) == 4


# ─── Simulated Annealing ──────────────────────────────────────────────────

class TestSimulatedAnnealing:

    def test_returns_assembly(self):
        frags   = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags, 20)
        asm0    = greedy_assembly(frags, entries)
        asm1    = simulated_annealing(asm0, entries, T_max=50, max_iter=200)
        assert isinstance(asm1, Assembly)

    def test_all_fragments_in_result(self):
        frags   = [_make_fragment(i) for i in range(5)]
        entries = _make_entries(frags, 25)
        asm0    = greedy_assembly(frags, entries)
        asm1    = simulated_annealing(asm0, entries, max_iter=100)
        assert set(asm1.placements.keys()) == {f.fragment_id for f in frags}

    def test_score_nonnegative(self):
        frags   = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        asm0    = greedy_assembly(frags, entries)
        asm1    = simulated_annealing(asm0, entries, max_iter=50)
        assert asm1.total_score >= 0.0

    def test_single_fragment_unchanged(self):
        frag = _make_fragment(0)
        asm0 = greedy_assembly([frag], [])
        asm1 = simulated_annealing(asm0, [], max_iter=50)
        assert len(asm1.placements) == 1


# ─── Beam Search ──────────────────────────────────────────────────────────

class TestBeamSearch:

    def test_returns_assembly(self):
        frags   = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags, 20)
        asm     = beam_search(frags, entries, beam_width=3)
        assert isinstance(asm, Assembly)

    def test_all_fragments_placed(self):
        frags   = [_make_fragment(i) for i in range(5)]
        entries = _make_entries(frags, 30)
        asm     = beam_search(frags, entries, beam_width=5)
        assert len(asm.placements) == len(frags)

    def test_placements_have_correct_format(self):
        frags   = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags, 15)
        asm     = beam_search(frags, entries, beam_width=3)
        for pos, angle in asm.placements.values():
            assert pos.shape == (2,)
            assert np.isfinite(angle)

    def test_beam_width_one_equals_greedy_approx(self):
        """beam_width=1 должен давать результат, сравнимый с жадным."""
        frags   = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags, 20)
        asm_b1  = beam_search(frags, entries, beam_width=1)
        asm_g   = greedy_assembly(frags, entries)
        # Оба должны разместить все фрагменты
        assert len(asm_b1.placements) == len(asm_g.placements)

    def test_wider_beam_score_geq_narrow(self):
        """Широкий луч обычно даёт score ≥ узкому (не гарантия, но тенденция)."""
        frags   = [_make_fragment(i) for i in range(5)]
        entries = _make_entries(frags, 30)
        asm_narrow = beam_search(frags, entries, beam_width=1)
        asm_wide   = beam_search(frags, entries, beam_width=8)
        # Мягкая проверка: широкий луч не хуже чем на 50%
        assert asm_wide.total_score >= asm_narrow.total_score * 0.5

    def test_empty_fragments(self):
        asm = beam_search([], [], beam_width=3)
        assert asm.placements == {}
