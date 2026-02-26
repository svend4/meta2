"""
Integration tests for puzzle_reconstruction assembly extra modules.

Covers:
- assembly/astar.py
- assembly/candidate_filter.py
- assembly/canvas_builder.py
- assembly/collision_detector.py
- assembly/fragment_arranger.py
- assembly/fragment_mapper.py
- assembly/fragment_sequencer.py
- assembly/fragment_sorter.py
- assembly/gap_analyzer.py
- assembly/hierarchical.py
- assembly/layout_builder.py
- assembly/layout_refiner.py
- assembly/overlap_resolver.py
- assembly/placement_optimizer.py
"""
from __future__ import annotations

import numpy as np
import pytest

# ── helpers to build domain objects ──────────────────────────────────────────

def _make_fragment(fid: int, n_edges: int = 2):
    """Create a minimal Fragment with simple Edge objects."""
    from puzzle_reconstruction.models import Fragment, Edge
    edges = []
    for k in range(n_edges):
        eid = fid * 10 + k
        edges.append(Edge(edge_id=eid, contour=np.zeros((4, 2))))
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    return Fragment(fragment_id=fid, image=img, edges=edges)


def _make_compat_entry(edge_i, edge_j, score: float):
    from puzzle_reconstruction.models import CompatEntry
    return CompatEntry(edge_i=edge_i, edge_j=edge_j, score=score)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. astar.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestAstar:
    def _make_assembly_inputs(self, n: int = 3):
        frags = [_make_fragment(i) for i in range(n)]
        entries = []
        for i in range(n):
            for j in range(i + 1, n):
                ei = frags[i].edges[0]
                ej = frags[j].edges[0]
                entries.append(_make_compat_entry(ei, ej, score=0.8))
        return frags, entries

    def test_astar_returns_assembly(self):
        from puzzle_reconstruction.assembly.astar import astar_assembly
        frags, entries = self._make_assembly_inputs(3)
        result = astar_assembly(frags, entries, max_states=200, beam_width=10)
        assert result is not None
        assert result.method == "astar"

    def test_astar_empty_fragments(self):
        from puzzle_reconstruction.assembly.astar import astar_assembly
        result = astar_assembly([], [])
        assert result.method == "astar"
        assert result.placements == {}

    def test_astar_single_fragment(self):
        from puzzle_reconstruction.assembly.astar import astar_assembly
        frags = [_make_fragment(0)]
        result = astar_assembly(frags, [])
        assert len(result.placements) >= 1

    def test_astar_places_all_fragments(self):
        from puzzle_reconstruction.assembly.astar import astar_assembly
        frags, entries = self._make_assembly_inputs(4)
        result = astar_assembly(frags, entries, max_states=500, beam_width=20)
        assert len(result.placements) == 4

    def test_astar_total_score_nonnegative(self):
        from puzzle_reconstruction.assembly.astar import astar_assembly
        frags, entries = self._make_assembly_inputs(3)
        result = astar_assembly(frags, entries)
        assert result.total_score >= 0.0

    def test_astar_no_entries(self):
        from puzzle_reconstruction.assembly.astar import astar_assembly
        frags = [_make_fragment(i) for i in range(3)]
        result = astar_assembly(frags, [])
        # should still place all fragments even with no compat entries
        assert len(result.placements) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 2. candidate_filter.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestCandidateFilter:
    def _candidates(self):
        from puzzle_reconstruction.assembly.candidate_filter import Candidate
        return [
            Candidate(idx1=0, idx2=1, score=0.9),
            Candidate(idx1=1, idx2=2, score=0.5),
            Candidate(idx1=2, idx2=3, score=0.3),
            Candidate(idx1=0, idx2=3, score=0.7),
        ]

    def test_filter_by_threshold_basic(self):
        from puzzle_reconstruction.assembly.candidate_filter import filter_by_threshold
        cands = self._candidates()
        result = filter_by_threshold(cands, threshold=0.6)
        assert result.n_kept == 2
        assert result.n_removed == 2

    def test_filter_by_threshold_sorted_descending(self):
        from puzzle_reconstruction.assembly.candidate_filter import filter_by_threshold
        result = filter_by_threshold(self._candidates(), threshold=0.0)
        scores = [c.score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_filter_top_k(self):
        from puzzle_reconstruction.assembly.candidate_filter import filter_top_k
        result = filter_top_k(self._candidates(), k=2)
        assert result.n_kept == 2
        assert result.candidates[0].score >= result.candidates[1].score

    def test_filter_by_rank(self):
        from puzzle_reconstruction.assembly.candidate_filter import filter_by_rank
        result = filter_by_rank(self._candidates(), rank_threshold=0.5)
        assert result.n_kept >= 1

    def test_deduplicate_candidates(self):
        from puzzle_reconstruction.assembly.candidate_filter import (
            Candidate, deduplicate_candidates,
        )
        dupes = [
            Candidate(idx1=0, idx2=1, score=0.8),
            Candidate(idx1=1, idx2=0, score=0.6),  # duplicate pair, lower score
            Candidate(idx1=2, idx2=3, score=0.4),
        ]
        result = deduplicate_candidates(dupes)
        assert result.n_kept == 2
        # The kept score for pair (0,1) should be the max
        pair_scores = {(min(c.idx1, c.idx2), max(c.idx1, c.idx2)): c.score
                       for c in result.candidates}
        assert pair_scores[(0, 1)] == 0.8

    def test_normalize_scores(self):
        from puzzle_reconstruction.assembly.candidate_filter import (
            Candidate, normalize_scores,
        )
        cands = [
            Candidate(idx1=0, idx2=1, score=0.2),
            Candidate(idx1=1, idx2=2, score=0.6),
            Candidate(idx1=2, idx2=3, score=1.0),
        ]
        normed = normalize_scores(cands)
        assert normed[0].score == pytest.approx(0.0)
        assert normed[-1].score == pytest.approx(1.0)

    def test_merge_candidate_lists(self):
        from puzzle_reconstruction.assembly.candidate_filter import (
            Candidate, merge_candidate_lists,
        )
        list1 = [Candidate(idx1=0, idx2=1, score=0.9)]
        list2 = [Candidate(idx1=2, idx2=3, score=0.5)]
        merged = merge_candidate_lists([list1, list2])
        assert len(merged) == 2

    def test_batch_filter(self):
        from puzzle_reconstruction.assembly.candidate_filter import (
            Candidate, batch_filter,
        )
        lists = [
            [Candidate(idx1=0, idx2=1, score=0.9),
             Candidate(idx1=1, idx2=2, score=0.2)],
            [Candidate(idx1=2, idx2=3, score=0.8)],
        ]
        results = batch_filter(lists, threshold=0.5)
        assert len(results) == 2
        assert results[0].n_kept == 1
        assert results[1].n_kept == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 3. canvas_builder.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanvasBuilder:
    def _make_placement(self, fid, x, y, h=20, w=20):
        from puzzle_reconstruction.assembly.canvas_builder import FragmentPlacement
        img = np.full((h, w, 3), fill_value=128, dtype=np.uint8)
        return FragmentPlacement(fragment_id=fid, image=img, x=x, y=y)

    def test_build_canvas_basic(self):
        from puzzle_reconstruction.assembly.canvas_builder import build_canvas
        ps = [self._make_placement(0, 0, 0), self._make_placement(1, 30, 0)]
        result = build_canvas(ps)
        assert result.n_placed == 2
        assert result.canvas.shape[2] == 3

    def test_build_canvas_coverage_between_0_1(self):
        from puzzle_reconstruction.assembly.canvas_builder import build_canvas
        ps = [self._make_placement(0, 0, 0)]
        result = build_canvas(ps, canvas_w=100, canvas_h=100)
        assert 0.0 <= result.coverage <= 1.0

    def test_compute_canvas_size(self):
        from puzzle_reconstruction.assembly.canvas_builder import (
            compute_canvas_size, FragmentPlacement,
        )
        img = np.zeros((10, 15, 3), dtype=np.uint8)
        ps = [FragmentPlacement(fragment_id=0, image=img, x=5, y=5)]
        w, h = compute_canvas_size(ps, padding=0)
        assert w == 20  # 5+15
        assert h == 15  # 5+10

    def test_make_empty_canvas(self):
        from puzzle_reconstruction.assembly.canvas_builder import (
            make_empty_canvas, CanvasConfig,
        )
        cfg = CanvasConfig(bg_color=(0, 0, 0))
        canvas = make_empty_canvas(50, 40, cfg)
        assert canvas.shape == (40, 50, 3)
        assert canvas.sum() == 0

    def test_place_fragment_overwrites(self):
        from puzzle_reconstruction.assembly.canvas_builder import (
            make_empty_canvas, place_fragment, FragmentPlacement,
        )
        canvas = make_empty_canvas(30, 30)
        img = np.full((10, 10, 3), 99, dtype=np.uint8)
        p = FragmentPlacement(fragment_id=0, image=img, x=0, y=0)
        canvas = place_fragment(canvas, p, blend_mode="overwrite")
        assert canvas[0, 0, 0] == 99

    def test_crop_to_content(self):
        from puzzle_reconstruction.assembly.canvas_builder import (
            build_canvas, crop_to_content,
        )
        ps = [self._make_placement(0, 10, 10)]
        result = build_canvas(ps, canvas_w=50, canvas_h=50)
        cropped = crop_to_content(result)
        assert cropped.shape[0] <= 50
        assert cropped.shape[1] <= 50

    def test_batch_build_canvases(self):
        from puzzle_reconstruction.assembly.canvas_builder import batch_build_canvases
        ps1 = [self._make_placement(0, 0, 0)]
        ps2 = [self._make_placement(0, 0, 0), self._make_placement(1, 25, 0)]
        results = batch_build_canvases([ps1, ps2])
        assert len(results) == 2
        assert results[1].n_placed == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 4. collision_detector.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestCollisionDetector:
    def _rect(self, fid, x, y, w=10, h=10):
        from puzzle_reconstruction.assembly.collision_detector import PlacedRect
        return PlacedRect(fragment_id=fid, x=x, y=y, width=w, height=h)

    def test_aabb_overlap_true(self):
        from puzzle_reconstruction.assembly.collision_detector import aabb_overlap
        a = self._rect(0, 0, 0, 10, 10)
        b = self._rect(1, 5, 5, 10, 10)
        assert aabb_overlap(a, b) is True

    def test_aabb_overlap_false(self):
        from puzzle_reconstruction.assembly.collision_detector import aabb_overlap
        a = self._rect(0, 0, 0, 10, 10)
        b = self._rect(1, 20, 20, 10, 10)
        assert aabb_overlap(a, b) is False

    def test_compute_overlap_returns_collision_info(self):
        from puzzle_reconstruction.assembly.collision_detector import compute_overlap
        a = self._rect(0, 0, 0, 10, 10)
        b = self._rect(1, 5, 0, 10, 10)
        info = compute_overlap(a, b)
        assert info is not None
        assert info.overlap_area > 0

    def test_compute_overlap_no_collision(self):
        from puzzle_reconstruction.assembly.collision_detector import compute_overlap
        a = self._rect(0, 0, 0, 10, 10)
        b = self._rect(1, 20, 0, 10, 10)
        assert compute_overlap(a, b) is None

    def test_detect_collisions_finds_overlap(self):
        from puzzle_reconstruction.assembly.collision_detector import detect_collisions
        rects = [self._rect(0, 0, 0), self._rect(1, 5, 5), self._rect(2, 50, 50)]
        colls = detect_collisions(rects)
        assert len(colls) >= 1

    def test_is_collision_free_true(self):
        from puzzle_reconstruction.assembly.collision_detector import is_collision_free
        rects = [self._rect(0, 0, 0), self._rect(1, 20, 0)]
        assert is_collision_free(rects) is True

    def test_is_collision_free_false(self):
        from puzzle_reconstruction.assembly.collision_detector import is_collision_free
        rects = [self._rect(0, 0, 0), self._rect(1, 5, 0)]
        assert is_collision_free(rects) is False

    def test_total_overlap_area(self):
        from puzzle_reconstruction.assembly.collision_detector import (
            detect_collisions, total_overlap_area,
        )
        rects = [self._rect(0, 0, 0), self._rect(1, 5, 5)]
        colls = detect_collisions(rects)
        area = total_overlap_area(colls)
        assert area >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. fragment_arranger.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestFragmentArranger:
    def test_arrange_grid_basic(self):
        from puzzle_reconstruction.assembly.fragment_arranger import arrange_grid
        sizes = [(20, 20)] * 6
        placements = arrange_grid(sizes, cols=3, gap=5)
        assert len(placements) == 6

    def test_arrange_grid_no_negative_coords(self):
        from puzzle_reconstruction.assembly.fragment_arranger import arrange_grid
        sizes = [(10, 10), (15, 20), (8, 8)]
        placements = arrange_grid(sizes, cols=2, gap=2)
        for p in placements:
            assert p.x >= 0
            assert p.y >= 0

    def test_arrange_strip_basic(self):
        from puzzle_reconstruction.assembly.fragment_arranger import arrange_strip
        sizes = [(30, 20), (30, 20), (30, 20)]
        placements = arrange_strip(sizes, canvas_w=70, gap=5)
        assert len(placements) == 3

    def test_arrange_strip_wraps_rows(self):
        from puzzle_reconstruction.assembly.fragment_arranger import arrange_strip
        # Each fragment is 40px wide, canvas 50px → wraps after first
        sizes = [(40, 10), (40, 10), (40, 10)]
        placements = arrange_strip(sizes, canvas_w=50, gap=0)
        # Second and third fragment should start at x=0 (new row)
        assert placements[1].x == 0
        assert placements[2].x == 0

    def test_center_placements(self):
        from puzzle_reconstruction.assembly.fragment_arranger import (
            arrange_grid, center_placements,
        )
        sizes = [(20, 20)] * 4
        placements = arrange_grid(sizes, cols=2, gap=0)
        centered = center_placements(placements, canvas_w=200, canvas_h=200)
        assert len(centered) == 4

    def test_group_bbox(self):
        from puzzle_reconstruction.assembly.fragment_arranger import (
            arrange_grid, group_bbox,
        )
        sizes = [(20, 20)] * 4
        placements = arrange_grid(sizes, cols=2, gap=0)
        x, y, w, h = group_bbox(placements)
        assert w > 0 and h > 0

    def test_shift_placements(self):
        from puzzle_reconstruction.assembly.fragment_arranger import (
            arrange_grid, shift_placements,
        )
        sizes = [(10, 10)]
        placements = arrange_grid(sizes, cols=1, gap=0)
        shifted = shift_placements(placements, dx=5, dy=3)
        assert shifted[0].x == placements[0].x + 5
        assert shifted[0].y == placements[0].y + 3

    def test_arrange_dispatches_strategies(self):
        from puzzle_reconstruction.assembly.fragment_arranger import (
            arrange, ArrangementParams,
        )
        sizes = [(10, 10)] * 4
        for strategy in ("grid", "strip", "center"):
            params = ArrangementParams(strategy=strategy, canvas_w=100, canvas_h=100)
            result = arrange(sizes, params)
            assert len(result) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# 6. fragment_mapper.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestFragmentMapper:
    def test_build_fragment_map_basic(self):
        from puzzle_reconstruction.assembly.fragment_mapper import (
            build_fragment_map, MapConfig,
        )
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        ids = [0, 1, 2]
        pos = [(10, 10), (60, 10), (10, 60)]
        result = build_fragment_map(ids, pos, cfg)
        assert result.n_fragments == 3
        assert result.n_assigned == 3

    def test_assign_to_zone_clamp(self):
        from puzzle_reconstruction.assembly.fragment_mapper import (
            assign_to_zone, MapConfig,
        )
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=4, n_zones_y=4)
        zx, zy = assign_to_zone(200, 200, cfg)
        assert zx == 3  # clamped to max
        assert zy == 3

    def test_compute_zone_grid(self):
        from puzzle_reconstruction.assembly.fragment_mapper import (
            compute_zone_grid, MapConfig,
        )
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        zones = compute_zone_grid(cfg)
        assert len(zones) == 4

    def test_map_result_by_fragment(self):
        from puzzle_reconstruction.assembly.fragment_mapper import build_fragment_map
        result = build_fragment_map([0, 1], [(0, 0), (50, 50)])
        d = result.by_fragment
        assert 0 in d and 1 in d

    def test_score_mapping_nonnegative(self):
        from puzzle_reconstruction.assembly.fragment_mapper import (
            build_fragment_map, score_mapping,
        )
        result = build_fragment_map([0, 1, 2], [(0, 0), (256, 0), (256, 256)])
        s = score_mapping(result)
        assert 0.0 <= s <= 1.0

    def test_remap_fragments(self):
        from puzzle_reconstruction.assembly.fragment_mapper import (
            build_fragment_map, remap_fragments,
        )
        result = build_fragment_map([0, 1], [(0, 0), (100, 100)])
        remapped = remap_fragments(result, {0: 10, 1: 20})
        fids = [fz.fragment_id for fz in remapped.assignments]
        assert 10 in fids and 20 in fids


# ═══════════════════════════════════════════════════════════════════════════════
# 7. fragment_sequencer.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestFragmentSequencer:
    def _matrix(self, n=4):
        np.random.seed(0)
        m = np.random.rand(n, n)
        np.fill_diagonal(m, 0)
        return (m + m.T) / 2  # symmetric

    def test_sequence_greedy_returns_full_order(self):
        from puzzle_reconstruction.assembly.fragment_sequencer import sequence_greedy
        mat = self._matrix(4)
        result = sequence_greedy(mat)
        assert len(result.order) == 4
        assert set(result.order) == {0, 1, 2, 3}

    def test_sequence_greedy_with_start(self):
        from puzzle_reconstruction.assembly.fragment_sequencer import sequence_greedy
        mat = self._matrix(4)
        result = sequence_greedy(mat, start=2)
        assert result.order[0] == 2

    def test_sequence_by_score(self):
        from puzzle_reconstruction.assembly.fragment_sequencer import sequence_by_score
        scores = [0.3, 0.9, 0.1, 0.7]
        result = sequence_by_score(scores, descending=True)
        assert result.order[0] == 1  # highest score first

    def test_compute_sequence_score(self):
        from puzzle_reconstruction.assembly.fragment_sequencer import (
            compute_sequence_score,
        )
        mat = np.array([[0, 0.5], [0.5, 0]], dtype=float)
        score = compute_sequence_score([0, 1], mat)
        assert score == pytest.approx(0.5)

    def test_reverse_sequence(self):
        from puzzle_reconstruction.assembly.fragment_sequencer import (
            sequence_greedy, reverse_sequence,
        )
        mat = self._matrix(4)
        result = sequence_greedy(mat)
        rev = reverse_sequence(result)
        assert rev.order == list(reversed(result.order))

    def test_rotate_sequence(self):
        from puzzle_reconstruction.assembly.fragment_sequencer import (
            sequence_greedy, rotate_sequence,
        )
        mat = self._matrix(4)
        result = sequence_greedy(mat, start=0)
        second_elem = result.order[1]
        rotated = rotate_sequence(result, second_elem)
        assert rotated.order[0] == second_elem

    def test_sequence_to_pairs(self):
        from puzzle_reconstruction.assembly.fragment_sequencer import (
            SequenceResult, sequence_to_pairs,
        )
        sr = SequenceResult(order=[0, 1, 2, 3], total_score=1.0)
        pairs = sequence_to_pairs(sr)
        assert pairs == [(0, 1), (1, 2), (2, 3)]

    def test_batch_sequence(self):
        from puzzle_reconstruction.assembly.fragment_sequencer import batch_sequence
        mats = [self._matrix(3) for _ in range(4)]
        results = batch_sequence(mats)
        assert len(results) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# 8. fragment_sorter.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestFragmentSorter:
    def _frags(self):
        from puzzle_reconstruction.assembly.fragment_sorter import FragmentSortInfo
        return [
            FragmentSortInfo(fragment_id=3, area=100.0, score=0.5),
            FragmentSortInfo(fragment_id=1, area=50.0,  score=0.9),
            FragmentSortInfo(fragment_id=2, area=200.0, score=0.2),
        ]

    def test_sort_by_id_ascending(self):
        from puzzle_reconstruction.assembly.fragment_sorter import sort_by_id
        result = sort_by_id(self._frags(), reverse=False)
        ids = [f.fragment_id for f in result]
        assert ids == [1, 2, 3]

    def test_sort_by_area_descending(self):
        from puzzle_reconstruction.assembly.fragment_sorter import sort_by_area
        result = sort_by_area(self._frags(), reverse=True)
        areas = [f.area for f in result]
        assert areas == sorted(areas, reverse=True)

    def test_sort_by_score_descending(self):
        from puzzle_reconstruction.assembly.fragment_sorter import sort_by_score
        result = sort_by_score(self._frags(), reverse=True)
        scores = [f.score for f in result]
        assert scores == sorted(scores, reverse=True)

    def test_sort_random_reproducible(self):
        from puzzle_reconstruction.assembly.fragment_sorter import sort_random
        r1 = [f.fragment_id for f in sort_random(self._frags(), seed=42)]
        r2 = [f.fragment_id for f in sort_random(self._frags(), seed=42)]
        assert r1 == r2

    def test_assign_positions(self):
        from puzzle_reconstruction.assembly.fragment_sorter import (
            assign_positions,
        )
        frags = self._frags()
        result = assign_positions(frags)
        for i, sf in enumerate(result):
            assert sf.position == i

    def test_sort_fragments_via_config(self):
        from puzzle_reconstruction.assembly.fragment_sorter import (
            sort_fragments, SortConfig,
        )
        cfg = SortConfig(strategy="area", reverse=True)
        result = sort_fragments(self._frags(), cfg)
        areas = [f.area for f in result]
        assert areas[0] >= areas[-1]

    def test_batch_sort(self):
        from puzzle_reconstruction.assembly.fragment_sorter import batch_sort
        lists = [self._frags(), self._frags()]
        results = batch_sort(lists)
        assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 9. gap_analyzer.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestGapAnalyzer:
    def _bounds(self, fid, x, y, w=20.0, h=20.0):
        from puzzle_reconstruction.assembly.gap_analyzer import FragmentBounds
        return FragmentBounds(fragment_id=fid, x=x, y=y, width=w, height=h)

    def test_compute_gap_far(self):
        from puzzle_reconstruction.assembly.gap_analyzer import compute_gap
        a = self._bounds(0, 0, 0)
        b = self._bounds(1, 100, 100)
        info = compute_gap(a, b)
        assert info.category == "far"
        assert info.distance > 0

    def test_compute_gap_overlap(self):
        from puzzle_reconstruction.assembly.gap_analyzer import compute_gap
        a = self._bounds(0, 0, 0)
        b = self._bounds(1, 5, 5)
        info = compute_gap(a, b)
        assert info.category == "overlap"
        assert info.is_overlapping

    def test_analyze_all_gaps_count(self):
        from puzzle_reconstruction.assembly.gap_analyzer import analyze_all_gaps
        frags = [self._bounds(i, i * 30, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        assert len(gaps) == 6  # C(4,2)

    def test_gap_histogram(self):
        from puzzle_reconstruction.assembly.gap_analyzer import (
            analyze_all_gaps, gap_histogram,
        )
        frags = [self._bounds(i, i * 50, 0) for i in range(3)]
        gaps = analyze_all_gaps(frags)
        counts, edges = gap_histogram(gaps, bins=5)
        assert len(counts) == 5
        assert len(edges) == 6

    def test_classify_gaps(self):
        from puzzle_reconstruction.assembly.gap_analyzer import (
            analyze_all_gaps, classify_gaps,
        )
        frags = [self._bounds(0, 0, 0), self._bounds(1, 5, 5), self._bounds(2, 100, 0)]
        gaps = analyze_all_gaps(frags)
        classified = classify_gaps(gaps)
        assert set(classified.keys()) == {"overlap", "touching", "near", "far"}

    def test_summarize(self):
        from puzzle_reconstruction.assembly.gap_analyzer import (
            analyze_all_gaps, summarize,
        )
        frags = [self._bounds(i, i * 40, 0) for i in range(3)]
        gaps = analyze_all_gaps(frags)
        stats = summarize(gaps)
        assert stats.n_pairs == 3
        assert stats.mean_distance >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. hierarchical.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestHierarchical:
    def _make_inputs(self, n=3):
        frags = [_make_fragment(i) for i in range(n)]
        entries = []
        for i in range(n):
            for j in range(i + 1, n):
                ei = frags[i].edges[0]
                ej = frags[j].edges[0]
                entries.append(_make_compat_entry(ei, ej, score=0.7))
        return frags, entries

    def test_hierarchical_basic(self):
        from puzzle_reconstruction.assembly.hierarchical import hierarchical_assembly
        frags, entries = self._make_inputs(3)
        result = hierarchical_assembly(frags, entries)
        assert result.method == "hierarchical"
        assert len(result.placements) == 3

    def test_hierarchical_empty(self):
        from puzzle_reconstruction.assembly.hierarchical import hierarchical_assembly
        result = hierarchical_assembly([], [])
        assert result.placements == {}

    def test_hierarchical_single_linkage(self):
        from puzzle_reconstruction.assembly.hierarchical import (
            hierarchical_assembly, HierarchicalConfig,
        )
        frags, entries = self._make_inputs(3)
        cfg = HierarchicalConfig(linkage="single")
        result = hierarchical_assembly(frags, entries, cfg)
        assert result.total_score >= 0.0

    def test_hierarchical_complete_linkage(self):
        from puzzle_reconstruction.assembly.hierarchical import (
            hierarchical_assembly, HierarchicalConfig,
        )
        frags, entries = self._make_inputs(3)
        cfg = HierarchicalConfig(linkage="complete")
        result = hierarchical_assembly(frags, entries, cfg)
        assert result.method == "hierarchical"

    def test_linkage_helpers(self):
        from puzzle_reconstruction.assembly.hierarchical import (
            single_linkage_score, average_linkage_score, complete_linkage_score,
        )
        scores = [0.2, 0.8, 0.5]
        assert single_linkage_score(scores) == 0.8
        assert complete_linkage_score(scores) == 0.2
        assert average_linkage_score(scores) == pytest.approx(0.5)

    def test_linkage_helpers_empty(self):
        from puzzle_reconstruction.assembly.hierarchical import (
            single_linkage_score, average_linkage_score, complete_linkage_score,
        )
        assert single_linkage_score([]) == 0.0
        assert average_linkage_score([]) == 0.0
        assert complete_linkage_score([]) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 11. layout_builder.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestLayoutBuilder:
    def test_create_layout(self):
        from puzzle_reconstruction.assembly.layout_builder import create_layout
        layout = create_layout(canvas_w=200, canvas_h=100, version="1.0")
        assert layout.canvas_w == 200.0
        assert layout.canvas_h == 100.0
        assert layout.params.get("version") == "1.0"

    def test_add_and_remove_cell(self):
        from puzzle_reconstruction.assembly.layout_builder import (
            create_layout, add_cell, remove_cell,
        )
        layout = create_layout()
        layout = add_cell(layout, fragment_idx=0, x=10, y=20, width=50, height=40)
        assert len(layout.cells) == 1
        layout = remove_cell(layout, fragment_idx=0)
        assert len(layout.cells) == 0

    def test_add_cell_replaces_existing(self):
        from puzzle_reconstruction.assembly.layout_builder import (
            create_layout, add_cell,
        )
        layout = create_layout()
        layout = add_cell(layout, 0, x=0, y=0, width=10, height=10)
        layout = add_cell(layout, 0, x=5, y=5, width=20, height=20)
        assert len(layout.cells) == 1
        assert layout.cells[0].x == 5.0

    def test_compute_bounding_box(self):
        from puzzle_reconstruction.assembly.layout_builder import (
            create_layout, add_cell, compute_bounding_box,
        )
        layout = create_layout()
        layout = add_cell(layout, 0, x=0, y=0, width=30, height=20)
        layout = add_cell(layout, 1, x=30, y=20, width=10, height=10)
        x, y, w, h = compute_bounding_box(layout)
        assert w == 40.0
        assert h == 30.0

    def test_snap_to_grid(self):
        from puzzle_reconstruction.assembly.layout_builder import (
            create_layout, add_cell, snap_to_grid,
        )
        layout = create_layout()
        layout = add_cell(layout, 0, x=7.3, y=2.8, width=10, height=10)
        snap_to_grid(layout, grid_size=5.0)
        assert layout.cells[0].x % 5.0 == pytest.approx(0.0)

    def test_render_layout_image(self):
        from puzzle_reconstruction.assembly.layout_builder import (
            create_layout, add_cell, render_layout_image,
        )
        layout = create_layout()
        layout = add_cell(layout, 0, x=0, y=0, width=40, height=30)
        img = render_layout_image(layout)
        assert img.ndim == 2
        assert img.dtype == np.uint8

    def test_layout_serialization_roundtrip(self):
        from puzzle_reconstruction.assembly.layout_builder import (
            create_layout, add_cell, layout_to_dict, dict_to_layout,
        )
        layout = create_layout(canvas_w=100, canvas_h=80)
        layout = add_cell(layout, 0, x=5, y=5, width=20, height=20, rotation=45.0)
        d = layout_to_dict(layout)
        restored = dict_to_layout(d)
        assert len(restored.cells) == 1
        assert restored.cells[0].rotation == 45.0


# ═══════════════════════════════════════════════════════════════════════════════
# 12. layout_refiner.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestLayoutRefiner:
    def _positions(self):
        from puzzle_reconstruction.assembly.layout_refiner import FragmentPosition
        return {
            0: FragmentPosition(fragment_id=0, x=0.0, y=0.0),
            1: FragmentPosition(fragment_id=1, x=10.0, y=0.0),
        }

    def test_compute_layout_score(self):
        from puzzle_reconstruction.assembly.layout_refiner import compute_layout_score
        positions = self._positions()
        adjacency = {(0, 1): 0.8}
        score = compute_layout_score(positions, adjacency, target_gap=10.0)
        assert score >= 0.0

    def test_refine_layout_runs(self):
        from puzzle_reconstruction.assembly.layout_refiner import (
            refine_layout, RefineConfig,
        )
        positions = self._positions()
        adjacency = {(0, 1): 0.9}
        cfg = RefineConfig(max_iter=5, step_size=1.0)
        result = refine_layout(positions, adjacency, cfg=cfg, target_gap=10.0)
        assert result.n_iter <= 5
        assert len(result.positions) == 2

    def test_refine_layout_converges(self):
        from puzzle_reconstruction.assembly.layout_refiner import (
            refine_layout, RefineConfig,
        )
        positions = self._positions()
        adjacency = {(0, 1): 1.0}
        cfg = RefineConfig(max_iter=50, convergence_eps=0.0001)
        result = refine_layout(positions, adjacency, cfg=cfg)
        # Either converged or exhausted iterations
        assert result.n_iter >= 1

    def test_apply_offset(self):
        from puzzle_reconstruction.assembly.layout_refiner import apply_offset
        positions = self._positions()
        shifted = apply_offset(positions, dx=5.0, dy=3.0)
        assert shifted[0].x == pytest.approx(5.0)
        assert shifted[1].x == pytest.approx(15.0)

    def test_compare_layouts(self):
        from puzzle_reconstruction.assembly.layout_refiner import (
            apply_offset, compare_layouts,
        )
        before = self._positions()
        after = apply_offset(before, dx=2.0, dy=0.0)
        comparison = compare_layouts(before, after)
        assert comparison["mean_shift"] == pytest.approx(2.0)
        assert comparison["n_moved"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 13. overlap_resolver.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverlapResolver:
    def _bbox(self, fid, x, y, w=20.0, h=20.0):
        from puzzle_reconstruction.assembly.overlap_resolver import BBox
        return BBox(fragment_id=fid, x=x, y=y, w=w, h=h)

    def test_compute_overlap_with_overlap(self):
        from puzzle_reconstruction.assembly.overlap_resolver import compute_overlap
        a = self._bbox(0, 0, 0)
        b = self._bbox(1, 10, 0)
        ov = compute_overlap(a, b)
        assert ov.has_overlap
        assert ov.area > 0

    def test_compute_overlap_no_overlap(self):
        from puzzle_reconstruction.assembly.overlap_resolver import compute_overlap
        a = self._bbox(0, 0, 0)
        b = self._bbox(1, 50, 0)
        ov = compute_overlap(a, b)
        assert not ov.has_overlap

    def test_detect_overlaps(self):
        from puzzle_reconstruction.assembly.overlap_resolver import detect_overlaps
        boxes = {
            0: self._bbox(0, 0, 0),
            1: self._bbox(1, 5, 5),
            2: self._bbox(2, 100, 100),
        }
        overlaps = detect_overlaps(boxes)
        assert len(overlaps) >= 1

    def test_resolve_overlaps_reduces_overlap(self):
        from puzzle_reconstruction.assembly.overlap_resolver import (
            resolve_overlaps, compute_total_overlap,
        )
        boxes = {0: self._bbox(0, 0, 0), 1: self._bbox(1, 5, 5)}
        before = compute_total_overlap(boxes)
        result = resolve_overlaps(boxes)
        after = compute_total_overlap(result.boxes)
        assert after <= before

    def test_resolve_overlaps_result_structure(self):
        from puzzle_reconstruction.assembly.overlap_resolver import resolve_overlaps
        boxes = {0: self._bbox(0, 0, 0), 1: self._bbox(1, 100, 100)}
        result = resolve_overlaps(boxes)
        assert result.n_iter >= 1
        assert isinstance(result.resolved, bool)

    def test_overlap_ratio(self):
        from puzzle_reconstruction.assembly.overlap_resolver import overlap_ratio
        boxes = {0: self._bbox(0, 0, 0), 1: self._bbox(1, 10, 0)}
        ratio = overlap_ratio(boxes)
        assert 0.0 <= ratio <= 1.0

    def test_bbox_translate(self):
        from puzzle_reconstruction.assembly.overlap_resolver import BBox
        b = BBox(fragment_id=0, x=5.0, y=5.0, w=10.0, h=10.0)
        b2 = b.translate(3.0, 2.0)
        assert b2.x == 8.0
        assert b2.y == 7.0
        # Original unmodified
        assert b.x == 5.0


# ═══════════════════════════════════════════════════════════════════════════════
# 14. placement_optimizer.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlacementOptimizer:
    def _score_matrix(self, n=4):
        np.random.seed(7)
        m = np.random.rand(n, n)
        np.fill_diagonal(m, 0)
        return (m + m.T) / 2

    def test_greedy_place_basic(self):
        from puzzle_reconstruction.assembly.placement_optimizer import greedy_place
        mat = self._score_matrix(4)
        result = greedy_place(4, mat, root=0)
        assert result.n_placed == 4
        assert result.score >= 0.0

    def test_greedy_place_history_length(self):
        from puzzle_reconstruction.assembly.placement_optimizer import greedy_place
        mat = self._score_matrix(4)
        result = greedy_place(4, mat, root=0)
        assert len(result.history) == 4

    def test_score_placement(self):
        from puzzle_reconstruction.assembly.placement_optimizer import (
            greedy_place, score_placement,
        )
        mat = self._score_matrix(4)
        result = greedy_place(4, mat, root=0)
        s = score_placement(result.state, mat)
        assert s >= 0.0

    def test_find_best_next(self):
        from puzzle_reconstruction.assembly.placement_optimizer import (
            find_best_next,
        )
        from puzzle_reconstruction.assembly.assembly_state import (
            create_state, place_fragment,
        )
        mat = self._score_matrix(4)
        state = create_state(4)
        state = place_fragment(state, 0, position=(0.0, 0.0))
        best_idx, gain = find_best_next(state, mat)
        assert best_idx in {1, 2, 3}

    def test_remove_worst_placed(self):
        from puzzle_reconstruction.assembly.placement_optimizer import (
            greedy_place, remove_worst_placed,
        )
        mat = self._score_matrix(4)
        result = greedy_place(4, mat, root=0)
        trimmed = remove_worst_placed(result, mat)
        assert trimmed.n_placed == 3

    def test_iterative_place(self):
        from puzzle_reconstruction.assembly.placement_optimizer import iterative_place
        mat = self._score_matrix(4)
        result = iterative_place(4, mat, root=0, max_iter=3, patience=2)
        assert result.n_placed == 4
