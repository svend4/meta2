"""Tests for puzzle_reconstruction/assembly/hierarchical.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.hierarchical import (
    HierarchicalConfig,
    Cluster,
    hierarchical_assembly,
    _inter_cluster_score,
    _merge_clusters,
    single_linkage_score,
    average_linkage_score,
    complete_linkage_score,
)
from puzzle_reconstruction.models import Fragment, CompatEntry, Assembly, EdgeSignature, EdgeSide


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


def _make_frag(fid: int) -> Fragment:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    frag = Fragment(fragment_id=fid, image=img)
    frag.edges = [_make_edge(fid * 10 + k) for k in range(2)]
    return frag


def _make_entry(fa: Fragment, fb: Fragment, score: float) -> CompatEntry:
    return CompatEntry(edge_i=fa.edges[0], edge_j=fb.edges[0], score=score)


def _make_cluster(cid: int, frag: Fragment) -> Cluster:
    return Cluster(
        cluster_id=cid,
        fragment_ids={frag.fragment_id},
        placements={frag.fragment_id: (np.array([float(cid) * 120, 0.0]), 0.0)},
    )


# ── HierarchicalConfig ────────────────────────────────────────────────────────

class TestHierarchicalConfig:

    def test_defaults(self):
        cfg = HierarchicalConfig()
        assert cfg.linkage == "average"
        assert cfg.min_merge_score == 0.0
        assert cfg.max_clusters == 1

    def test_custom(self):
        cfg = HierarchicalConfig(linkage="single", min_merge_score=0.5, max_clusters=2)
        assert cfg.linkage == "single"
        assert cfg.min_merge_score == 0.5
        assert cfg.max_clusters == 2


# ── Cluster ────────────────────────────────────────────────────────────────────

class TestCluster:

    def test_creation(self):
        f = _make_frag(0)
        c = Cluster(cluster_id=0, fragment_ids={0}, placements={0: (np.zeros(2), 0.0)})
        assert c.cluster_id == 0
        assert 0 in c.fragment_ids


# ── Linkage functions ──────────────────────────────────────────────────────────

class TestLinkageFunctions:

    def test_single_max(self):
        assert single_linkage_score([0.1, 0.9, 0.5]) == pytest.approx(0.9)

    def test_average_mean(self):
        assert average_linkage_score([0.2, 0.8]) == pytest.approx(0.5)

    def test_complete_min(self):
        assert complete_linkage_score([0.3, 0.7, 0.5]) == pytest.approx(0.3)

    def test_empty_returns_zero(self):
        assert single_linkage_score([]) == pytest.approx(0.0)
        assert average_linkage_score([]) == pytest.approx(0.0)
        assert complete_linkage_score([]) == pytest.approx(0.0)

    def test_single_item_all_equal(self):
        assert single_linkage_score([0.6]) == pytest.approx(0.6)
        assert average_linkage_score([0.6]) == pytest.approx(0.6)
        assert complete_linkage_score([0.6]) == pytest.approx(0.6)


# ── _inter_cluster_score ──────────────────────────────────────────────────────

class TestInterClusterScore:

    def test_no_shared_entries_returns_negative(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        c0 = _make_cluster(0, f0)
        c1 = _make_cluster(1, f1)
        e2f = {}
        score = _inter_cluster_score(c0, c1, [], e2f)
        assert score < 0

    def test_with_shared_entry(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        e2f = {}
        for frag in [f0, f1]:
            for edge in frag.edges:
                e2f[edge.edge_id] = frag
        c0 = _make_cluster(0, f0)
        c1 = _make_cluster(1, f1)
        entries = [_make_entry(f0, f1, 0.7)]
        score = _inter_cluster_score(c0, c1, entries, e2f, linkage="average")
        assert score == pytest.approx(0.7)

    def test_single_linkage(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        e2f = {}
        for frag in [f0, f1]:
            for edge in frag.edges:
                e2f[edge.edge_id] = frag
        c0 = _make_cluster(0, f0)
        c1 = _make_cluster(1, f1)
        entries = [
            CompatEntry(edge_i=f0.edges[0], edge_j=f1.edges[0], score=0.4),
            CompatEntry(edge_i=f0.edges[1], edge_j=f1.edges[1], score=0.9),
        ]
        score = _inter_cluster_score(c0, c1, entries, e2f, linkage="single")
        assert score == pytest.approx(0.9)

    def test_complete_linkage(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        e2f = {}
        for frag in [f0, f1]:
            for edge in frag.edges:
                e2f[edge.edge_id] = frag
        c0 = _make_cluster(0, f0)
        c1 = _make_cluster(1, f1)
        entries = [
            CompatEntry(edge_i=f0.edges[0], edge_j=f1.edges[0], score=0.4),
            CompatEntry(edge_i=f0.edges[1], edge_j=f1.edges[1], score=0.9),
        ]
        score = _inter_cluster_score(c0, c1, entries, e2f, linkage="complete")
        assert score == pytest.approx(0.4)


# ── _merge_clusters ────────────────────────────────────────────────────────────

class TestMergeClusters:

    def _e2f(self, frags):
        m = {}
        for f in frags:
            for e in f.edges:
                m[e.edge_id] = f
        return m

    def test_merged_has_all_fragment_ids(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        c0 = _make_cluster(0, f0)
        c1 = _make_cluster(1, f1)
        e2f = self._e2f([f0, f1])
        merged = _merge_clusters(c0, c1, [], e2f, new_id=99, merge_score=0.5)
        assert 0 in merged.fragment_ids
        assert 1 in merged.fragment_ids

    def test_merged_has_all_placements(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        c0 = _make_cluster(0, f0)
        c1 = _make_cluster(1, f1)
        e2f = self._e2f([f0, f1])
        merged = _merge_clusters(c0, c1, [], e2f, new_id=99, merge_score=0.5)
        assert 0 in merged.placements
        assert 1 in merged.placements

    def test_total_score_includes_merge_score(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        c0 = _make_cluster(0, f0)
        c1 = _make_cluster(1, f1)
        e2f = self._e2f([f0, f1])
        merged = _merge_clusters(c0, c1, [], e2f, new_id=99, merge_score=0.75)
        assert merged.total_score == pytest.approx(0.75)

    def test_placements_positions_finite(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        c0 = _make_cluster(0, f0)
        c1 = _make_cluster(1, f1)
        e2f = self._e2f([f0, f1])
        merged = _merge_clusters(c0, c1, [], e2f, new_id=99, merge_score=0.5)
        for fid, (pos, rot) in merged.placements.items():
            assert np.all(np.isfinite(pos))


# ── hierarchical_assembly ──────────────────────────────────────────────────────

class TestHierarchicalAssembly:

    def test_returns_assembly(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        entries = [_make_entry(f0, f1, 0.8)]
        result = hierarchical_assembly([f0, f1], entries)
        assert isinstance(result, Assembly)

    def test_empty_fragments(self):
        result = hierarchical_assembly([], [])
        assert isinstance(result, Assembly)
        assert result.placements == {}

    def test_single_fragment(self):
        f0 = _make_frag(0)
        result = hierarchical_assembly([f0], [])
        assert isinstance(result, Assembly)
        assert len(result.placements) == 1

    def test_method_is_hierarchical(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        result = hierarchical_assembly([f0, f1], [])
        assert result.method == "hierarchical"

    def test_all_fragments_placed(self):
        frags = [_make_frag(i) for i in range(4)]
        entries = [_make_entry(frags[i], frags[i+1], 0.9) for i in range(3)]
        result = hierarchical_assembly(frags, entries)
        assert len(result.placements) == 4

    def test_total_score_non_negative(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        result = hierarchical_assembly([f0, f1], [_make_entry(f0, f1, 0.7)])
        assert result.total_score >= 0.0

    def test_linkage_single_works(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        cfg = HierarchicalConfig(linkage="single")
        result = hierarchical_assembly([f0, f1], [_make_entry(f0, f1, 0.8)], cfg=cfg)
        assert isinstance(result, Assembly)

    def test_linkage_complete_works(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        cfg = HierarchicalConfig(linkage="complete")
        result = hierarchical_assembly([f0, f1], [_make_entry(f0, f1, 0.8)], cfg=cfg)
        assert isinstance(result, Assembly)

    def test_min_merge_score_prevents_merge(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        cfg = HierarchicalConfig(min_merge_score=0.99)
        entries = [_make_entry(f0, f1, 0.1)]  # Low score → won't merge
        result = hierarchical_assembly([f0, f1], entries, cfg=cfg)
        # With no valid merge, both fragments should still appear in placements
        assert len(result.placements) == 2

    def test_max_clusters_2_stops_early(self):
        frags = [_make_frag(i) for i in range(4)]
        entries = [_make_entry(frags[i], frags[i+1], 0.8) for i in range(3)]
        cfg = HierarchicalConfig(max_clusters=2)
        result = hierarchical_assembly(frags, entries, cfg=cfg)
        assert isinstance(result, Assembly)
        assert len(result.placements) == 4  # All fragments still placed

    def test_placements_positions_finite(self):
        frags = [_make_frag(i) for i in range(3)]
        entries = [_make_entry(frags[i], frags[i+1], 0.8) for i in range(2)]
        result = hierarchical_assembly(frags, entries)
        for fid, (pos, rot) in result.placements.items():
            assert np.all(np.isfinite(pos))
            assert np.isfinite(rot)

    def test_no_entries_no_crash(self):
        frags = [_make_frag(i) for i in range(5)]
        result = hierarchical_assembly(frags, [])
        assert isinstance(result, Assembly)
