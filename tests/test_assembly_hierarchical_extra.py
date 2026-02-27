"""Extra tests for puzzle_reconstruction/assembly/hierarchical.py"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.hierarchical import (
    Cluster,
    HierarchicalConfig,
    _inter_cluster_score,
    _merge_clusters,
    average_linkage_score,
    complete_linkage_score,
    hierarchical_assembly,
    single_linkage_score,
)
from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_edge(eid: int) -> EdgeSignature:
    curve = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])
    return EdgeSignature(
        edge_id=eid,
        side=EdgeSide.RIGHT,
        virtual_curve=curve,
        fd=1.0,
        css_vec=np.ones(16) / 4.0,
        ifs_coeffs=np.zeros(8),
        length=1.0,
    )


def _make_frag(fid: int, n_edges: int = 2) -> Fragment:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    frag = Fragment(fragment_id=fid, image=img)
    frag.edges = [_make_edge(fid * 10 + k) for k in range(n_edges)]
    return frag


def _make_entry(fa: Fragment, fb: Fragment, score: float) -> CompatEntry:
    return CompatEntry(edge_i=fa.edges[0], edge_j=fb.edges[0], score=score)


def _make_cluster(cid: int, frag: Fragment) -> Cluster:
    return Cluster(
        cluster_id=cid,
        fragment_ids={frag.fragment_id},
        placements={frag.fragment_id: (np.array([float(cid) * 120, 0.0]), 0.0)},
    )


def _e2f(*frags: Fragment):
    m = {}
    for f in frags:
        for e in f.edges:
            m[e.edge_id] = f
    return m


# ── HierarchicalConfig extra ──────────────────────────────────────────────────

class TestHierarchicalConfigExtra:

    def test_linkage_single_accepted(self):
        cfg = HierarchicalConfig(linkage="single")
        assert cfg.linkage == "single"

    def test_linkage_complete_accepted(self):
        cfg = HierarchicalConfig(linkage="complete")
        assert cfg.linkage == "complete"

    def test_min_merge_score_float(self):
        cfg = HierarchicalConfig(min_merge_score=0.5)
        assert isinstance(cfg.min_merge_score, float)

    def test_max_clusters_integer(self):
        cfg = HierarchicalConfig(max_clusters=3)
        assert isinstance(cfg.max_clusters, int)
        assert cfg.max_clusters == 3

    def test_default_max_clusters_is_1(self):
        assert HierarchicalConfig().max_clusters == 1


# ── Cluster dataclass extra ───────────────────────────────────────────────────

class TestClusterExtra:

    def test_default_empty_fragment_ids(self):
        c = Cluster(cluster_id=0)
        assert c.fragment_ids == set()

    def test_default_empty_placements(self):
        c = Cluster(cluster_id=0)
        assert c.placements == {}

    def test_default_total_score_zero(self):
        c = Cluster(cluster_id=0)
        assert c.total_score == 0.0

    def test_cluster_id_stored(self):
        c = Cluster(cluster_id=99)
        assert c.cluster_id == 99

    def test_fragment_ids_mutable_set(self):
        c = Cluster(cluster_id=0, fragment_ids={1, 2})
        c.fragment_ids.add(3)
        assert 3 in c.fragment_ids


# ── Linkage functions extra ───────────────────────────────────────────────────

class TestLinkageFunctionsExtra:

    def test_single_with_identical_values(self):
        assert single_linkage_score([0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_average_with_identical_values(self):
        assert average_linkage_score([0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_complete_with_identical_values(self):
        assert complete_linkage_score([0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_single_many_values(self):
        vals = [0.1, 0.3, 0.7, 0.95, 0.2]
        assert single_linkage_score(vals) == pytest.approx(0.95)

    def test_average_many_values(self):
        vals = [0.2, 0.4, 0.6, 0.8]
        assert average_linkage_score(vals) == pytest.approx(0.5)

    def test_complete_many_values(self):
        vals = [0.1, 0.5, 0.9]
        assert complete_linkage_score(vals) == pytest.approx(0.1)

    def test_single_zero_values(self):
        assert single_linkage_score([0.0, 0.0]) == pytest.approx(0.0)

    def test_all_functions_return_float(self):
        for fn in (single_linkage_score, average_linkage_score, complete_linkage_score):
            result = fn([0.3, 0.7])
            assert isinstance(result, float)


# ── _inter_cluster_score extra ────────────────────────────────────────────────

class TestInterClusterScoreExtra:

    def test_same_cluster_no_cross_entries(self):
        f0 = _make_frag(0)
        f1 = _make_frag(1)
        e2f = _e2f(f0, f1)
        # Both frags in same cluster_a, cluster_b is separate with no frags
        c0 = Cluster(cluster_id=0, fragment_ids={0, 1},
                     placements={0: (np.zeros(2), 0.0), 1: (np.zeros(2), 0.0)})
        c1 = Cluster(cluster_id=1, fragment_ids=set(), placements={})
        entries = [_make_entry(f0, f1, 0.9)]
        score = _inter_cluster_score(c0, c1, entries, e2f)
        assert score < 0  # no cross entries

    def test_multiple_cross_entries_average(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        e2f = _e2f(f0, f1)
        c0 = _make_cluster(0, f0)
        c1 = _make_cluster(1, f1)
        entries = [
            CompatEntry(edge_i=f0.edges[0], edge_j=f1.edges[0], score=0.2),
            CompatEntry(edge_i=f0.edges[1], edge_j=f1.edges[1], score=0.8),
        ]
        avg = _inter_cluster_score(c0, c1, entries, e2f, linkage="average")
        assert avg == pytest.approx(0.5)

    def test_multiple_cross_entries_single(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        e2f = _e2f(f0, f1)
        c0 = _make_cluster(0, f0)
        c1 = _make_cluster(1, f1)
        entries = [
            CompatEntry(edge_i=f0.edges[0], edge_j=f1.edges[0], score=0.2),
            CompatEntry(edge_i=f0.edges[1], edge_j=f1.edges[1], score=0.8),
        ]
        single = _inter_cluster_score(c0, c1, entries, e2f, linkage="single")
        assert single == pytest.approx(0.8)

    def test_multiple_cross_entries_complete(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        e2f = _e2f(f0, f1)
        c0 = _make_cluster(0, f0)
        c1 = _make_cluster(1, f1)
        entries = [
            CompatEntry(edge_i=f0.edges[0], edge_j=f1.edges[0], score=0.2),
            CompatEntry(edge_i=f0.edges[1], edge_j=f1.edges[1], score=0.8),
        ]
        complete = _inter_cluster_score(c0, c1, entries, e2f, linkage="complete")
        assert complete == pytest.approx(0.2)

    def test_entry_direction_agnostic(self):
        """Score should be same regardless of edge_i/edge_j direction."""
        f0, f1 = _make_frag(0), _make_frag(1)
        e2f = _e2f(f0, f1)
        c0 = _make_cluster(0, f0)
        c1 = _make_cluster(1, f1)
        entry_forward  = CompatEntry(edge_i=f0.edges[0], edge_j=f1.edges[0], score=0.75)
        entry_backward = CompatEntry(edge_i=f1.edges[0], edge_j=f0.edges[0], score=0.75)
        s_forward  = _inter_cluster_score(c0, c1, [entry_forward],  e2f)
        s_backward = _inter_cluster_score(c0, c1, [entry_backward], e2f)
        assert s_forward == pytest.approx(s_backward)


# ── _merge_clusters extra ──────────────────────────────────────────────────────

class TestMergeClustersExtra:

    def test_merged_cluster_id_is_new_id(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        c0, c1 = _make_cluster(0, f0), _make_cluster(1, f1)
        merged = _merge_clusters(c0, c1, [], _e2f(f0, f1), new_id=77, merge_score=0.5)
        assert merged.cluster_id == 77

    def test_total_score_sum_of_both_plus_merge(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        c0 = Cluster(cluster_id=0, fragment_ids={0},
                     placements={0: (np.array([0.0, 0.0]), 0.0)}, total_score=1.0)
        c1 = Cluster(cluster_id=1, fragment_ids={1},
                     placements={1: (np.array([120.0, 0.0]), 0.0)}, total_score=0.5)
        merged = _merge_clusters(c0, c1, [], _e2f(f0, f1), new_id=2, merge_score=0.25)
        assert merged.total_score == pytest.approx(1.75)

    def test_merged_fragment_ids_union(self):
        f0, f1, f2 = _make_frag(0), _make_frag(1), _make_frag(2)
        c01 = Cluster(cluster_id=0, fragment_ids={0, 1},
                      placements={
                          0: (np.zeros(2), 0.0),
                          1: (np.array([120.0, 0.0]), 0.0),
                      })
        c2 = _make_cluster(2, f2)
        e2f = _e2f(f0, f1, f2)
        merged = _merge_clusters(c01, c2, [], e2f, new_id=99, merge_score=0.3)
        assert merged.fragment_ids == {0, 1, 2}

    def test_merged_positions_all_finite(self):
        f0, f1, f2 = _make_frag(0), _make_frag(1), _make_frag(2)
        c01 = Cluster(cluster_id=0, fragment_ids={0, 1},
                      placements={
                          0: (np.zeros(2), 0.0),
                          1: (np.array([120.0, 0.0]), 0.0),
                      })
        c2 = _make_cluster(2, f2)
        e2f = _e2f(f0, f1, f2)
        merged = _merge_clusters(c01, c2, [], e2f, new_id=99, merge_score=0.3)
        for fid, (pos, rot) in merged.placements.items():
            assert np.all(np.isfinite(pos))

    def test_empty_cluster_a_placements(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        c_empty = Cluster(cluster_id=0, fragment_ids={0}, placements={})
        c1 = _make_cluster(1, f1)
        merged = _merge_clusters(c_empty, c1, [], _e2f(f0, f1), new_id=99, merge_score=0.0)
        # Should not crash; positions should be finite
        for fid, (pos, rot) in merged.placements.items():
            assert np.all(np.isfinite(pos))


# ── hierarchical_assembly extra ────────────────────────────────────────────────

class TestHierarchicalAssemblyExtra:

    def test_two_fragments_single_linkage(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        cfg = HierarchicalConfig(linkage="single")
        result = hierarchical_assembly([f0, f1], [_make_entry(f0, f1, 0.9)], cfg)
        assert isinstance(result, Assembly)
        assert len(result.placements) == 2

    def test_two_fragments_complete_linkage(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        cfg = HierarchicalConfig(linkage="complete")
        result = hierarchical_assembly([f0, f1], [_make_entry(f0, f1, 0.9)], cfg)
        assert isinstance(result, Assembly)

    def test_five_fragments_all_placed(self):
        frags = [_make_frag(i) for i in range(5)]
        entries = [_make_entry(frags[i], frags[i + 1], 0.8) for i in range(4)]
        result = hierarchical_assembly(frags, entries)
        assert len(result.placements) == 5

    def test_ten_fragments_all_placed(self):
        frags = [_make_frag(i) for i in range(10)]
        entries = [_make_entry(frags[i], frags[i + 1], 0.7) for i in range(9)]
        result = hierarchical_assembly(frags, entries)
        assert len(result.placements) == 10

    def test_total_score_ge_zero(self):
        frags = [_make_frag(i) for i in range(4)]
        entries = [_make_entry(frags[i], frags[i + 1], 0.8) for i in range(3)]
        result = hierarchical_assembly(frags, entries)
        assert result.total_score >= 0.0

    def test_no_merge_when_min_score_exceeds_all(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        cfg = HierarchicalConfig(min_merge_score=1.0)  # impossible to meet
        entries = [_make_entry(f0, f1, 0.5)]
        result = hierarchical_assembly([f0, f1], entries, cfg)
        # Both should appear in placements even without merging
        assert len(result.placements) == 2

    def test_max_clusters_stops_early(self):
        frags = [_make_frag(i) for i in range(6)]
        entries = [_make_entry(frags[i], frags[i + 1], 0.8) for i in range(5)]
        cfg = HierarchicalConfig(max_clusters=3)
        result = hierarchical_assembly(frags, entries, cfg)
        # All fragments still placed even when merging stops early
        assert len(result.placements) == 6

    def test_empty_entries_each_frag_placed_separately(self):
        frags = [_make_frag(i) for i in range(4)]
        result = hierarchical_assembly(frags, [])
        # All fragments placed even with no entries (no merges)
        assert len(result.placements) == 4

    def test_method_attribute(self):
        frags = [_make_frag(i) for i in range(2)]
        result = hierarchical_assembly(frags, [])
        assert result.method == "hierarchical"

    def test_compat_matrix_is_ndarray(self):
        frags = [_make_frag(i) for i in range(2)]
        result = hierarchical_assembly(frags, [])
        assert isinstance(result.compat_matrix, np.ndarray)

    def test_higher_score_higher_total(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        r_low  = hierarchical_assembly([f0, f1], [_make_entry(f0, f1, 0.1)])
        r_high = hierarchical_assembly([f0, f1], [_make_entry(f0, f1, 0.9)])
        assert r_high.total_score >= r_low.total_score

    def test_placements_have_2d_positions(self):
        frags = [_make_frag(i) for i in range(3)]
        entries = [_make_entry(frags[i], frags[i + 1], 0.8) for i in range(2)]
        result = hierarchical_assembly(frags, entries)
        for fid, (pos, rot) in result.placements.items():
            assert pos.shape == (2,)

    def test_repeated_calls_same_result(self):
        frags = [_make_frag(i) for i in range(3)]
        entries = [_make_entry(frags[i], frags[i + 1], 0.8) for i in range(2)]
        r1 = hierarchical_assembly(frags, entries)
        r2 = hierarchical_assembly(frags, entries)
        assert r1.total_score == pytest.approx(r2.total_score)
