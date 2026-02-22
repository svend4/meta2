"""Расширенные тесты для puzzle_reconstruction/assembly/beam_search.py."""
import numpy as np
import pytest

from puzzle_reconstruction.assembly.beam_search import (
    Hypothesis,
    _compute_placement,
    _expand,
    _fill_orphans,
    beam_search,
)
from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _edge(edge_id: int, side: EdgeSide = EdgeSide.RIGHT) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=80.0,
    )


def _frag(fid: int, n_edges: int = 2) -> Fragment:
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        mask=np.zeros((32, 32), dtype=np.uint8),
        contour=np.zeros((8, 2)),
    )
    frag.edges = [_edge(fid * 10 + i) for i in range(n_edges)]
    return frag


def _entry(ei: EdgeSignature, ej: EdgeSignature, score: float = 0.7) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


def _entries(frags) -> list:
    result = []
    for i, fi in enumerate(frags):
        for j, fj in enumerate(frags):
            if i >= j:
                continue
            result.append(_entry(fi.edges[0], fj.edges[0], 0.5 + 0.1 * (i + j)))
    return sorted(result, key=lambda e: e.score, reverse=True)


def _edge_to_frag(frags):
    d = {}
    for f in frags:
        for e in f.edges:
            d[e.edge_id] = f
    return d


# ─── TestHypothesis ───────────────────────────────────────────────────────────

class TestHypothesis:
    def test_default_score_zero(self):
        h = Hypothesis(placements={}, placed_ids=set())
        assert h.score == pytest.approx(0.0)

    def test_default_last_entries_empty(self):
        h = Hypothesis(placements={}, placed_ids=set())
        assert h.last_entries == []

    def test_placements_stored(self):
        p = {0: (np.array([1.0, 2.0]), 0.5)}
        h = Hypothesis(placements=p, placed_ids={0})
        assert 0 in h.placements

    def test_placed_ids_stored(self):
        h = Hypothesis(placements={}, placed_ids={1, 2, 3})
        assert h.placed_ids == {1, 2, 3}

    def test_score_stored(self):
        h = Hypothesis(placements={}, placed_ids=set(), score=3.14)
        assert h.score == pytest.approx(3.14)

    def test_last_entries_stored(self):
        frags = [_frag(0), _frag(1)]
        ent = _entry(frags[0].edges[0], frags[1].edges[0])
        h = Hypothesis(placements={}, placed_ids=set(), last_entries=[ent])
        assert len(h.last_entries) == 1

    def test_default_factory_independent(self):
        """last_entries default list is not shared between instances."""
        h1 = Hypothesis(placements={}, placed_ids=set())
        h2 = Hypothesis(placements={}, placed_ids=set())
        h1.last_entries.append("x")
        assert h2.last_entries == []


# ─── TestBeamSearch ───────────────────────────────────────────────────────────

class TestBeamSearch:
    # --- Return type ---

    def test_returns_assembly(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _entries(frags))
        assert isinstance(result, Assembly)

    def test_empty_fragments_returns_assembly(self):
        result = beam_search([], [])
        assert isinstance(result, Assembly)

    def test_empty_fragments_empty_placements(self):
        result = beam_search([], [])
        assert result.placements == {}

    # --- Fragment placement ---

    def test_single_fragment_placed(self):
        frags = [_frag(0)]
        result = beam_search(frags, [])
        assert 0 in result.placements

    def test_two_fragments_both_placed(self):
        f0, f1 = _frag(0), _frag(1)
        ents = [_entry(f0.edges[0], f1.edges[0])]
        result = beam_search([f0, f1], ents)
        assert 0 in result.placements and 1 in result.placements

    def test_all_fragments_placed(self):
        frags = [_frag(i) for i in range(5)]
        result = beam_search(frags, _entries(frags))
        assert all(f.fragment_id in result.placements for f in frags)

    def test_placements_count_equals_fragment_count(self):
        frags = [_frag(i) for i in range(4)]
        result = beam_search(frags, _entries(frags))
        assert len(result.placements) == 4

    def test_no_entries_all_placed_as_orphans(self):
        frags = [_frag(i) for i in range(5)]
        result = beam_search(frags, [])
        assert len(result.placements) == 5

    # --- Score ---

    def test_total_score_nonneg(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _entries(frags))
        assert result.total_score >= 0.0

    def test_total_score_is_float(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _entries(frags))
        assert isinstance(result.total_score, float)

    def test_total_score_zero_with_no_entries(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, [])
        assert result.total_score == pytest.approx(0.0)

    def test_score_positive_with_good_entries(self):
        f0, f1 = _frag(0), _frag(1)
        ents = [_entry(f0.edges[0], f1.edges[0], score=1.0)]
        result = beam_search([f0, f1], ents)
        assert result.total_score > 0.0

    # --- Beam width ---

    def test_beam_width_1_runs(self):
        frags = [_frag(i) for i in range(4)]
        result = beam_search(frags, _entries(frags), beam_width=1)
        assert isinstance(result, Assembly)

    def test_beam_width_100_runs(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _entries(frags), beam_width=100)
        assert isinstance(result, Assembly)

    def test_beam_width_higher_score_ge_width_1(self):
        """Wider beam may find better solution."""
        frags = [_frag(i) for i in range(4)]
        ents = _entries(frags)
        r1 = beam_search(frags, ents, beam_width=1)
        r10 = beam_search(frags, ents, beam_width=10)
        assert r10.total_score >= r1.total_score - 1e-9

    # --- max_depth ---

    def test_max_depth_1_runs(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _entries(frags), max_depth=1)
        assert len(result.placements) == 3  # orphans fill the rest

    def test_max_depth_none_uses_fragment_count(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _entries(frags), max_depth=None)
        assert len(result.placements) == 3

    def test_max_depth_exceeds_fragment_count_ok(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _entries(frags), max_depth=100)
        assert len(result.placements) == 3

    # --- Placement structure ---

    def test_first_fragment_at_origin(self):
        frags = [_frag(0)]
        result = beam_search(frags, [])
        pos, angle = result.placements[0]
        np.testing.assert_allclose(np.asarray(pos), [0.0, 0.0], atol=1e-10)
        assert angle == pytest.approx(0.0)

    def test_placement_pos_len_2(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _entries(frags))
        for fid, (pos, angle) in result.placements.items():
            assert len(pos) == 2

    def test_placement_angle_is_float(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _entries(frags))
        for fid, (pos, angle) in result.placements.items():
            assert isinstance(float(angle), float)

    # --- Assembly fields ---

    def test_fragments_stored_in_assembly(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _entries(frags))
        assert result.fragments is frags

    def test_compat_matrix_is_array(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _entries(frags))
        assert isinstance(result.compat_matrix, np.ndarray)

    # --- Chain structure ---

    def test_chain_of_3_all_placed(self):
        f0 = _frag(0, n_edges=1)
        f1 = _frag(1, n_edges=1)
        f2 = _frag(2, n_edges=1)
        ents = [
            _entry(f0.edges[0], f1.edges[0], 0.9),
            _entry(f1.edges[0], f2.edges[0], 0.8),
        ]
        result = beam_search([f0, f1, f2], ents)
        assert all(i in result.placements for i in [0, 1, 2])


# ─── TestExpand ───────────────────────────────────────────────────────────────

class TestExpand:
    def test_returns_list(self):
        frags = [_frag(0), _frag(1)]
        e2f = _edge_to_frag(frags)
        hyp = Hypothesis(
            placements={0: (np.array([0.0, 0.0]), 0.0)},
            placed_ids={0},
        )
        ents = [_entry(frags[0].edges[0], frags[1].edges[0])]
        result = _expand(hyp, frags, ents, e2f, n_expand=5)
        assert isinstance(result, list)

    def test_expansion_adds_new_fragment(self):
        frags = [_frag(0), _frag(1)]
        e2f = _edge_to_frag(frags)
        hyp = Hypothesis(
            placements={0: (np.array([0.0, 0.0]), 0.0)},
            placed_ids={0},
        )
        ents = [_entry(frags[0].edges[0], frags[1].edges[0])]
        result = _expand(hyp, frags, ents, e2f, n_expand=5)
        if result:
            assert 1 in result[0].placed_ids

    def test_both_placed_skipped(self):
        frags = [_frag(0), _frag(1)]
        e2f = _edge_to_frag(frags)
        hyp = Hypothesis(
            placements={0: (np.array([0.0, 0.0]), 0.0),
                        1: (np.array([100.0, 0.0]), 0.0)},
            placed_ids={0, 1},
        )
        ents = [_entry(frags[0].edges[0], frags[1].edges[0])]
        result = _expand(hyp, frags, ents, e2f, n_expand=5)
        assert result == []

    def test_neither_placed_skipped(self):
        frags = [_frag(0), _frag(1), _frag(2)]
        e2f = _edge_to_frag(frags)
        hyp = Hypothesis(
            placements={0: (np.array([0.0, 0.0]), 0.0)},
            placed_ids={0},
        )
        # Entry between unplaced frags 1 and 2
        ents = [_entry(frags[1].edges[0], frags[2].edges[0])]
        result = _expand(hyp, frags, ents, e2f, n_expand=5)
        assert result == []

    def test_n_expand_limits_results(self):
        frags = [_frag(i) for i in range(5)]
        e2f = _edge_to_frag(frags)
        hyp = Hypothesis(
            placements={0: (np.array([0.0, 0.0]), 0.0)},
            placed_ids={0},
        )
        ents = _entries(frags)
        result = _expand(hyp, frags, ents, e2f, n_expand=2)
        assert len(result) <= 2

    def test_expansion_score_increases(self):
        frags = [_frag(0), _frag(1)]
        e2f = _edge_to_frag(frags)
        hyp = Hypothesis(
            placements={0: (np.array([0.0, 0.0]), 0.0)},
            placed_ids={0},
            score=0.3,
        )
        ents = [_entry(frags[0].edges[0], frags[1].edges[0], score=0.8)]
        result = _expand(hyp, frags, ents, e2f, n_expand=5)
        if result:
            assert result[0].score > hyp.score

    def test_expansion_carries_last_entries(self):
        frags = [_frag(0), _frag(1)]
        e2f = _edge_to_frag(frags)
        ent = _entry(frags[0].edges[0], frags[1].edges[0])
        hyp = Hypothesis(
            placements={0: (np.array([0.0, 0.0]), 0.0)},
            placed_ids={0},
            last_entries=[],
        )
        result = _expand(hyp, frags, [ent], e2f, n_expand=5)
        if result:
            assert ent in result[0].last_entries


# ─── TestComputePlacement ─────────────────────────────────────────────────────

class TestComputePlacement:
    def test_returns_tuple(self):
        f0 = _frag(0, 1)
        f1 = _frag(1, 1)
        placement = (np.array([0.0, 0.0]), 0.0)
        result = _compute_placement(f0, f0.edges[0], f1, f1.edges[0], placement)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pos_has_len_2(self):
        f0 = _frag(0, 1)
        f1 = _frag(1, 1)
        placement = (np.array([0.0, 0.0]), 0.0)
        pos, _ = _compute_placement(f0, f0.edges[0], f1, f1.edges[0], placement)
        assert len(pos) == 2

    def test_angle_is_float(self):
        f0 = _frag(0, 1)
        f1 = _frag(1, 1)
        placement = (np.array([0.0, 0.0]), 0.0)
        _, angle = _compute_placement(f0, f0.edges[0], f1, f1.edges[0], placement)
        assert isinstance(angle, float)

    def test_angle_is_anchor_plus_pi(self):
        f0 = _frag(0, 1)
        f1 = _frag(1, 1)
        anchor_angle = 0.5
        placement = (np.array([0.0, 0.0]), anchor_angle)
        _, angle = _compute_placement(f0, f0.edges[0], f1, f1.edges[0], placement)
        assert angle == pytest.approx(anchor_angle + np.pi)

    def test_anchor_offset_shifts_pos(self):
        f0 = _frag(0, 1)
        f1 = _frag(1, 1)
        p0, _ = _compute_placement(f0, f0.edges[0], f1, f1.edges[0],
                                    (np.array([0.0, 0.0]), 0.0))
        p1, _ = _compute_placement(f0, f0.edges[0], f1, f1.edges[0],
                                    (np.array([50.0, 0.0]), 0.0))
        # Shifting anchor by 50 in x should shift result by 50 in x
        assert abs(p1[0] - p0[0]) == pytest.approx(50.0, abs=1e-6)


# ─── TestFillOrphans ──────────────────────────────────────────────────────────

class TestFillOrphans:
    def test_all_placed_no_change(self):
        frags = [_frag(0), _frag(1)]
        hyp = Hypothesis(
            placements={0: (np.array([0.0, 0.0]), 0.0),
                        1: (np.array([100.0, 0.0]), 0.0)},
            placed_ids={0, 1},
        )
        _fill_orphans(frags, hyp)
        assert len(hyp.placements) == 2

    def test_orphan_added(self):
        frags = [_frag(0), _frag(1)]
        hyp = Hypothesis(
            placements={0: (np.array([0.0, 0.0]), 0.0)},
            placed_ids={0},
        )
        _fill_orphans(frags, hyp)
        assert 1 in hyp.placements

    def test_orphan_y_below_placed(self):
        frags = [_frag(0), _frag(1)]
        hyp = Hypothesis(
            placements={0: (np.array([0.0, 300.0]), 0.0)},
            placed_ids={0},
        )
        _fill_orphans(frags, hyp)
        pos_orphan, _ = hyp.placements[1]
        assert pos_orphan[1] >= 300.0

    def test_empty_placements_y_base_zero(self):
        frags = [_frag(0)]
        hyp = Hypothesis(placements={}, placed_ids=set())
        _fill_orphans(frags, hyp)
        pos, _ = hyp.placements[0]
        assert pos[1] == pytest.approx(0.0)

    def test_multiple_orphans_placed_at_different_x(self):
        frags = [_frag(i) for i in range(4)]
        hyp = Hypothesis(
            placements={0: (np.array([0.0, 0.0]), 0.0)},
            placed_ids={0},
        )
        _fill_orphans(frags, hyp)
        positions = [hyp.placements[fid][0][0] for fid in [1, 2, 3]]
        # x coordinates should differ (k * 200.0)
        assert len(set(positions)) == 3
