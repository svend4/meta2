"""
Integration tests for assembly modules:
    astar.py, bridge.py, fragment_scorer.py, gamma_optimizer.py, hierarchical.py

At least 12 tests per module (~60+ total).
Uses numpy seeded arrays and synthetic Fragment / CompatEntry objects.
No mocks – all tests exercise real code paths and verify computed values.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from puzzle_reconstruction.models import (
    Fragment,
    CompatEntry,
    Assembly,
    EdgeSignature,
    EdgeSide,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_edge(edge_id: int, n_pts: int = 10, seed: int = 0) -> EdgeSignature:
    rng = np.random.RandomState(seed)
    curve = rng.rand(n_pts, 2).astype(np.float32)
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=curve,
        fd=1.5,
        css_vec=rng.rand(8).astype(np.float32),
        ifs_coeffs=rng.rand(4).astype(np.float32),
        length=float(n_pts),
    )


def _make_fragment(fragment_id: int, n_edges: int = 2, seed: int = 0) -> Fragment:
    rng = np.random.RandomState(seed + fragment_id * 17)
    image = rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    edges = [_make_edge(edge_id=fragment_id * 10 + i, n_pts=10, seed=seed + i)
             for i in range(n_edges)]
    return Fragment(fragment_id=fragment_id, image=image, edges=edges)


def _make_compat_entries(fragments, rng: np.random.RandomState):
    """Create synthetic CompatEntry records between every pair of fragments."""
    entries = []
    for i, fi in enumerate(fragments):
        for j, fj in enumerate(fragments):
            if i >= j:
                continue
            score = float(rng.rand())
            entry = CompatEntry(
                edge_i=fi.edges[0],
                edge_j=fj.edges[0],
                score=score,
                dtw_dist=float(rng.rand() * 5 + 0.1),
                css_sim=float(rng.rand()),
                fd_diff=float(rng.rand() * 0.5),
                text_score=0.0,
            )
            entries.append(entry)
    entries.sort(key=lambda e: e.score, reverse=True)
    return entries


# ---------------------------------------------------------------------------
# TestAstar
# ---------------------------------------------------------------------------

class TestAstar:
    """Tests for puzzle_reconstruction.assembly.astar"""

    @pytest.fixture(autouse=True)
    def _imports(self):
        from puzzle_reconstruction.assembly.astar import (
            astar_assembly,
            _build_edge_to_frag,
            _build_best_score_per_frag,
            _score_for_placement,
            _heuristic,
            _place_new_fragment,
            _AStarState,
        )
        self.astar_assembly = astar_assembly
        self._build_edge_to_frag = _build_edge_to_frag
        self._build_best_score_per_frag = _build_best_score_per_frag
        self._score_for_placement = _score_for_placement
        self._heuristic = _heuristic
        self._place_new_fragment = _place_new_fragment
        self._AStarState = _AStarState

    def _frags_and_entries(self, n: int = 4, seed: int = 42):
        rng = np.random.RandomState(seed)
        frags = [_make_fragment(i, n_edges=2, seed=seed) for i in range(n)]
        entries = _make_compat_entries(frags, rng)
        return frags, entries

    # ── 1. Empty fragments returns empty Assembly ──────────────────────────
    def test_empty_fragments(self):
        result = self.astar_assembly([], [])
        assert isinstance(result, Assembly)
        assert result.placements == {}

    # ── 2. Single fragment is placed at origin ─────────────────────────────
    def test_single_fragment_placed_at_origin(self):
        frags = [_make_fragment(0)]
        result = self.astar_assembly(frags, [])
        assert 0 in result.placements
        pos, rot = result.placements[0]
        np.testing.assert_array_equal(pos, [0.0, 0.0])
        assert rot == 0.0

    # ── 3. All fragments are placed ────────────────────────────────────────
    def test_all_fragments_placed(self):
        frags, entries = self._frags_and_entries(4)
        result = self.astar_assembly(frags, entries)
        assert len(result.placements) == 4
        for f in frags:
            assert f.fragment_id in result.placements

    # ── 4. Method tag is "astar" ───────────────────────────────────────────
    def test_method_tag(self):
        frags, entries = self._frags_and_entries(3)
        result = self.astar_assembly(frags, entries)
        assert result.method == "astar"

    # ── 5. total_score is a finite float ──────────────────────────────────
    def test_total_score_finite(self):
        frags, entries = self._frags_and_entries(4)
        result = self.astar_assembly(frags, entries)
        assert math.isfinite(result.total_score)

    # ── 6. total_score >= 0 (scores are non-negative) ─────────────────────
    def test_total_score_non_negative(self):
        frags, entries = self._frags_and_entries(4)
        result = self.astar_assembly(frags, entries)
        assert result.total_score >= 0.0

    # ── 7. _build_edge_to_frag maps every edge_id to its parent ───────────
    def test_build_edge_to_frag(self):
        frags = [_make_fragment(i, n_edges=2) for i in range(3)]
        mapping = self._build_edge_to_frag(frags)
        for frag in frags:
            for edge in frag.edges:
                assert edge.edge_id in mapping
                assert mapping[edge.edge_id].fragment_id == frag.fragment_id

    # ── 8. _build_best_score_per_frag returns correct upper bounds ─────────
    def test_build_best_score_per_frag_values(self):
        frags, entries = self._frags_and_entries(3)
        e2f = self._build_edge_to_frag(frags)
        best = self._build_best_score_per_frag(frags, entries, e2f)
        # Each value must be >= 0
        for frag in frags:
            fid = frag.fragment_id
            assert fid in best
            assert best[fid] >= 0.0

    # ── 9. _heuristic sums best scores for unplaced frags ─────────────────
    def test_heuristic_sum(self):
        best = {0: 0.9, 1: 0.5, 2: 0.3}
        unplaced = frozenset([1, 2])
        h = self._heuristic(unplaced, best)
        assert abs(h - 0.8) < 1e-9

    # ── 10. _heuristic is zero for empty unplaced set ─────────────────────
    def test_heuristic_empty(self):
        h = self._heuristic(frozenset(), {0: 1.0})
        assert h == 0.0

    # ── 11. _score_for_placement returns 0 when no placed frags share entries
    def test_score_for_placement_no_overlap(self):
        frags, entries = self._frags_and_entries(4)
        e2f = self._build_edge_to_frag(frags)
        # placed set contains only fragment id 99 (not in entries)
        score = self._score_for_placement(frags[0], frozenset([99]), entries, e2f)
        assert score == 0.0

    # ── 12. _score_for_placement is >= 0 ──────────────────────────────────
    def test_score_for_placement_non_negative(self):
        frags, entries = self._frags_and_entries(4)
        e2f = self._build_edge_to_frag(frags)
        placed = frozenset([frags[1].fragment_id, frags[2].fragment_id])
        score = self._score_for_placement(frags[0], placed, entries, e2f)
        assert score >= 0.0

    # ── 13. _AStarState f_score is negated (g+h) ──────────────────────────
    def test_astar_state_f_score(self):
        state = self._AStarState(frozenset([0]), {}, g_score=3.0, h_score=2.0)
        assert state.f_score == -(3.0 + 2.0)

    # ── 14. Placement positions are 2-element arrays ──────────────────────
    def test_placement_positions_shape(self):
        frags, entries = self._frags_and_entries(3)
        result = self.astar_assembly(frags, entries)
        for fid, (pos, rot) in result.placements.items():
            assert pos.shape == (2,)

    # ── 15. max_states=1 still returns a valid Assembly ───────────────────
    def test_max_states_one(self):
        frags, entries = self._frags_and_entries(3)
        result = self.astar_assembly(frags, entries, max_states=1)
        assert isinstance(result, Assembly)
        assert len(result.placements) >= 1

    # ── 16. beam_width=1 returns valid Assembly ───────────────────────────
    def test_beam_width_one(self):
        frags, entries = self._frags_and_entries(3)
        result = self.astar_assembly(frags, entries, beam_width=1)
        assert isinstance(result, Assembly)


# ---------------------------------------------------------------------------
# TestAssemblyBridge
# ---------------------------------------------------------------------------

class TestAssemblyBridge:
    """Tests for puzzle_reconstruction.assembly.bridge"""

    @pytest.fixture(autouse=True)
    def _imports(self):
        from puzzle_reconstruction.assembly.bridge import (
            build_assembly_registry,
            list_assembly_fns,
            get_assembly_fn,
            get_assembly_category,
            ASSEMBLY_CATEGORIES,
        )
        self.build_assembly_registry = build_assembly_registry
        self.list_assembly_fns = list_assembly_fns
        self.get_assembly_fn = get_assembly_fn
        self.get_assembly_category = get_assembly_category
        self.ASSEMBLY_CATEGORIES = ASSEMBLY_CATEGORIES

    # ── 1. build_assembly_registry returns a dict ─────────────────────────
    def test_registry_is_dict(self):
        reg = self.build_assembly_registry()
        assert isinstance(reg, dict)

    # ── 2. Registry values are all callable ───────────────────────────────
    def test_registry_values_callable(self):
        reg = self.build_assembly_registry()
        for name, fn in reg.items():
            assert callable(fn), f"{name} is not callable"

    # ── 3. list_assembly_fns returns a sorted list ────────────────────────
    def test_list_fns_sorted(self):
        fns = self.list_assembly_fns()
        assert fns == sorted(fns)

    # ── 4. list_assembly_fns with valid category returns subset ───────────
    def test_list_fns_by_category(self):
        for cat in self.ASSEMBLY_CATEGORIES:
            fns = self.list_assembly_fns(cat)
            assert isinstance(fns, list)
            cat_set = set(self.ASSEMBLY_CATEGORIES[cat])
            for name in fns:
                assert name in cat_set

    # ── 5. list_assembly_fns unknown category returns empty ───────────────
    def test_list_fns_unknown_category(self):
        fns = self.list_assembly_fns("nonexistent_category_xyz")
        assert fns == []

    # ── 6. get_assembly_fn returns callable for known functions ───────────
    def test_get_assembly_fn_known(self):
        reg = self.build_assembly_registry()
        for name in list(reg.keys())[:3]:
            fn = self.get_assembly_fn(name)
            assert callable(fn)

    # ── 7. get_assembly_fn returns None for unknown name ──────────────────
    def test_get_assembly_fn_unknown(self):
        result = self.get_assembly_fn("__nonexistent_fn_xyz__")
        assert result is None

    # ── 8. get_assembly_category returns correct category ─────────────────
    def test_get_assembly_category_correct(self):
        for cat, names in self.ASSEMBLY_CATEGORIES.items():
            for name in names:
                returned = self.get_assembly_category(name)
                assert returned == cat
                break  # one sample per category

    # ── 9. get_assembly_category returns None for unknown ─────────────────
    def test_get_assembly_category_unknown(self):
        assert self.get_assembly_category("__no_such_fn__") is None

    # ── 10. ASSEMBLY_CATEGORIES has exactly 8 categories ──────────────────
    def test_categories_count(self):
        expected = {"state", "filter", "geometry", "cost",
                    "layout", "scoring", "sequencing", "tracking"}
        assert set(self.ASSEMBLY_CATEGORIES.keys()) == expected

    # ── 11. All category values are non-empty lists ────────────────────────
    def test_category_values_are_lists(self):
        for cat, names in self.ASSEMBLY_CATEGORIES.items():
            assert isinstance(names, list), f"{cat} value is not a list"
            assert len(names) > 0, f"{cat} is empty"

    # ── 12. list_assembly_fns result is a subset of ASSEMBLY_CATEGORIES ───
    def test_all_listed_fns_in_categories(self):
        all_cats = {name for names in self.ASSEMBLY_CATEGORIES.values()
                    for name in names}
        for fn_name in self.list_assembly_fns():
            assert fn_name in all_cats

    # ── 13. Registry is idempotent (call twice -> same keys) ──────────────
    def test_registry_idempotent(self):
        reg1 = self.build_assembly_registry()
        reg2 = self.build_assembly_registry()
        assert set(reg1.keys()) == set(reg2.keys())

    # ── 14. "geometry" category contains geometry-related names ───────────
    def test_geometry_category_names(self):
        geom_names = set(self.ASSEMBLY_CATEGORIES["geometry"])
        assert "aabb_overlap" in geom_names
        assert "detect_collisions" in geom_names
        assert "analyze_all_gaps" in geom_names


# ---------------------------------------------------------------------------
# TestFragmentScorer
# ---------------------------------------------------------------------------

class TestFragmentScorer:
    """Tests for puzzle_reconstruction.assembly.fragment_scorer"""

    @pytest.fixture(autouse=True)
    def _imports(self):
        from puzzle_reconstruction.assembly.fragment_scorer import (
            ScoreConfig,
            FragmentScore,
            AssemblyScore,
            score_fragment,
            score_assembly,
            top_k_placed,
            bottom_k_placed,
            batch_score,
        )
        from puzzle_reconstruction.assembly.assembly_state import (
            create_state, place_fragment, add_adjacency,
        )
        from puzzle_reconstruction.assembly.cost_matrix import build_from_scores
        self.ScoreConfig = ScoreConfig
        self.FragmentScore = FragmentScore
        self.AssemblyScore = AssemblyScore
        self.score_fragment = score_fragment
        self.score_assembly = score_assembly
        self.top_k_placed = top_k_placed
        self.bottom_k_placed = bottom_k_placed
        self.batch_score = batch_score
        self.create_state = create_state
        self.place_fragment = place_fragment
        self.add_adjacency = add_adjacency
        self.build_from_scores = build_from_scores

    def _make_state_and_cm(self, n: int = 4, seed: int = 7):
        rng = np.random.RandomState(seed)
        scores = rng.rand(n, n).astype(np.float32)
        np.fill_diagonal(scores, 0.0)
        cm = self.build_from_scores(scores)
        state = self.create_state(n)
        for i in range(n):
            state = self.place_fragment(state, i, (float(i * 10), 0.0))
        # connect 0-1, 1-2, 2-3
        for i in range(n - 1):
            state = self.add_adjacency(state, i, i + 1)
        return state, cm

    # ── 1. ScoreConfig default weights sum to 1.0 ─────────────────────────
    def test_score_config_default_total_weight(self):
        cfg = self.ScoreConfig()
        assert abs(cfg.total_weight - 1.0) < 1e-9

    # ── 2. ScoreConfig rejects negative neighbor_weight ───────────────────
    def test_score_config_negative_neighbor_weight(self):
        with pytest.raises(ValueError):
            self.ScoreConfig(neighbor_weight=-0.1)

    # ── 3. ScoreConfig rejects negative coverage_weight ──────────────────
    def test_score_config_negative_coverage_weight(self):
        with pytest.raises(ValueError):
            self.ScoreConfig(coverage_weight=-1.0)

    # ── 4. ScoreConfig rejects min_neighbors < 1 ──────────────────────────
    def test_score_config_min_neighbors_zero(self):
        with pytest.raises(ValueError):
            self.ScoreConfig(min_neighbors=0)

    # ── 5. FragmentScore local_score is in [0, 1] ─────────────────────────
    def test_fragment_score_local_in_range(self):
        state, cm = self._make_state_and_cm()
        fs = self.score_fragment(state, 0, cm)
        assert 0.0 <= fs.local_score <= 1.0

    # ── 6. Isolated fragment (no neighbours) gets local_score=0.5 ─────────
    def test_fragment_score_isolated_neutral(self):
        state = self.create_state(3)
        rng = np.random.RandomState(1)
        scores = rng.rand(3, 3).astype(np.float32)
        cm = self.build_from_scores(scores)
        state = self.place_fragment(state, 0, (0.0, 0.0))
        fs = self.score_fragment(state, 0, cm)
        assert fs.local_score == 0.5
        assert fs.n_neighbors == 0
        assert not fs.is_reliable

    # ── 7. score_fragment raises when fragment not placed ─────────────────
    def test_score_fragment_unplaced_raises(self):
        state, cm = self._make_state_and_cm(3)
        state2 = self.create_state(3)
        state2 = self.place_fragment(state2, 0, (0.0, 0.0))
        with pytest.raises(ValueError):
            self.score_fragment(state2, 2, cm)

    # ── 8. score_fragment raises on cm/state n_fragments mismatch ─────────
    def test_score_fragment_mismatch_raises(self):
        state, cm = self._make_state_and_cm(4)
        state3 = self.create_state(3)
        state3 = self.place_fragment(state3, 0, (0.0, 0.0))
        with pytest.raises(ValueError):
            self.score_fragment(state3, 0, cm)

    # ── 9. score_assembly coverage matches placed/total ───────────────────
    def test_assembly_score_coverage(self):
        state, cm = self._make_state_and_cm(4)
        asm = self.score_assembly(state, cm)
        assert abs(asm.coverage - 1.0) < 1e-9

    # ── 10. score_assembly global_score in [0, 1] ─────────────────────────
    def test_assembly_score_global_in_range(self):
        state, cm = self._make_state_and_cm(4)
        asm = self.score_assembly(state, cm)
        assert 0.0 <= asm.global_score <= 1.0

    # ── 11. score_assembly n_placed matches placed count ──────────────────
    def test_assembly_score_n_placed(self):
        state, cm = self._make_state_and_cm(4)
        asm = self.score_assembly(state, cm)
        assert asm.n_placed == 4

    # ── 12. top_k_placed returns at most k items sorted ascending ─────────
    def test_top_k_placed_sorted_ascending(self):
        state, cm = self._make_state_and_cm(4)
        asm = self.score_assembly(state, cm)
        top = self.top_k_placed(asm, k=3)
        assert len(top) == 3
        scores_only = [s for _, s in top]
        assert scores_only == sorted(scores_only)

    # ── 13. bottom_k_placed returns at most k items sorted descending ──────
    def test_bottom_k_placed_sorted_descending(self):
        state, cm = self._make_state_and_cm(4)
        asm = self.score_assembly(state, cm)
        bottom = self.bottom_k_placed(asm, k=3)
        assert len(bottom) == 3
        scores_only = [s for _, s in bottom]
        assert scores_only == sorted(scores_only, reverse=True)

    # ── 14. top_k_placed raises when k < 1 ────────────────────────────────
    def test_top_k_placed_invalid_k(self):
        state, cm = self._make_state_and_cm(3)
        asm = self.score_assembly(state, cm)
        with pytest.raises(ValueError):
            self.top_k_placed(asm, k=0)

    # ── 15. bottom_k_placed raises when k < 1 ─────────────────────────────
    def test_bottom_k_placed_invalid_k(self):
        state, cm = self._make_state_and_cm(3)
        asm = self.score_assembly(state, cm)
        with pytest.raises(ValueError):
            self.bottom_k_placed(asm, k=0)

    # ── 16. batch_score returns same length as input ───────────────────────
    def test_batch_score_length(self):
        state, cm = self._make_state_and_cm(4)
        results = self.batch_score([state, state, state], cm)
        assert len(results) == 3

    # ── 17. AssemblyScore summary string contains key fields ──────────────
    def test_assembly_score_summary_string(self):
        state, cm = self._make_state_and_cm(4)
        asm = self.score_assembly(state, cm)
        s = asm.summary()
        assert "global=" in s
        assert "coverage=" in s
        assert "placed=" in s

    # ── 18. FragmentScore raises on negative fragment_idx ─────────────────
    def test_fragment_score_invalid_idx(self):
        with pytest.raises(ValueError):
            self.FragmentScore(fragment_idx=-1, local_score=0.5, n_neighbors=0)

    # ── 19. FragmentScore raises on local_score > 1 ───────────────────────
    def test_fragment_score_out_of_range_local(self):
        with pytest.raises(ValueError):
            self.FragmentScore(fragment_idx=0, local_score=1.5, n_neighbors=0)


# ---------------------------------------------------------------------------
# TestGammaOptimizer
# ---------------------------------------------------------------------------

class TestGammaOptimizer:
    """Tests for puzzle_reconstruction.assembly.gamma_optimizer"""

    @pytest.fixture(autouse=True)
    def _imports(self):
        from puzzle_reconstruction.assembly.gamma_optimizer import (
            GammaEdgeModel,
            gamma_optimizer,
            _fit_gamma_model,
            _evaluate_ll,
            _rotate_curve,
        )
        self.GammaEdgeModel = GammaEdgeModel
        self.gamma_optimizer = gamma_optimizer
        self._fit_gamma_model = _fit_gamma_model
        self._evaluate_ll = _evaluate_ll
        self._rotate_curve = _rotate_curve

    def _frags_and_entries(self, n: int = 3, seed: int = 99):
        rng = np.random.RandomState(seed)
        frags = [_make_fragment(i, n_edges=2, seed=seed) for i in range(n)]
        entries = _make_compat_entries(frags, rng)
        return frags, entries

    # ── 1. GammaEdgeModel has default k and theta ──────────────────────────
    def test_default_params(self):
        m = self.GammaEdgeModel()
        assert m.k == 2.0
        assert m.theta == 0.5

    # ── 2. fit updates k and theta ────────────────────────────────────────
    def test_fit_updates_params(self):
        rng = np.random.RandomState(0)
        deviations = rng.gamma(shape=3.0, scale=0.8, size=200)
        m = self.GammaEdgeModel()
        m.fit(deviations)
        assert m.k > 0.0
        assert m.theta > 0.0

    # ── 3. fit with too few points keeps defaults ──────────────────────────
    def test_fit_too_few_points(self):
        m = self.GammaEdgeModel(k=2.0, theta=0.5)
        m.fit(np.array([0.1, 0.2]))  # < 5 points
        assert m.k == 2.0
        assert m.theta == 0.5

    # ── 4. log_likelihood returns finite float ─────────────────────────────
    def test_log_likelihood_finite(self):
        m = self.GammaEdgeModel()
        ll = m.log_likelihood(np.array([0.1, 0.5, 1.0, 2.0]))
        assert math.isfinite(ll)

    # ── 5. log_likelihood is <= 0 ─────────────────────────────────────────
    def test_log_likelihood_non_positive(self):
        m = self.GammaEdgeModel()
        ll = m.log_likelihood(np.array([0.1, 0.5, 1.0, 2.0]))
        assert ll <= 0.0

    # ── 6. pair_score returns finite value for compatible arrays ──────────
    def test_pair_score_finite(self):
        rng = np.random.RandomState(1)
        a = rng.rand(15, 2).astype(np.float32)
        b = rng.rand(15, 2).astype(np.float32)
        m = self.GammaEdgeModel()
        score = m.pair_score(a, b)
        assert math.isfinite(score)

    # ── 7. pair_score returns -inf for empty arrays ────────────────────────
    def test_pair_score_empty(self):
        m = self.GammaEdgeModel()
        score = m.pair_score(np.zeros((0, 2)), np.zeros((5, 2)))
        assert score == -np.inf

    # ── 8. gamma_optimizer returns Assembly ───────────────────────────────
    def test_returns_assembly(self):
        frags, entries = self._frags_and_entries(3)
        result = self.gamma_optimizer(frags, entries, n_iter=50, seed=42)
        assert isinstance(result, Assembly)

    # ── 9. gamma_optimizer covers all fragments ───────────────────────────
    def test_all_fragments_covered(self):
        frags, entries = self._frags_and_entries(3)
        result = self.gamma_optimizer(frags, entries, n_iter=50, seed=42)
        assert len(result.placements) == 3

    # ── 10. Empty fragments returns empty Assembly ─────────────────────────
    def test_empty_fragments(self):
        result = self.gamma_optimizer([], [], n_iter=10, seed=0)
        assert result.placements == {}

    # ── 11. _rotate_curve preserves point count ───────────────────────────
    def test_rotate_curve_shape(self):
        rng = np.random.RandomState(3)
        curve = rng.rand(20, 2).astype(np.float32)
        rotated = self._rotate_curve(curve, math.pi / 4)
        assert rotated.shape == (20, 2)

    # ── 12. _rotate_curve by 0 is identity ────────────────────────────────
    def test_rotate_curve_zero_angle(self):
        rng = np.random.RandomState(5)
        curve = rng.rand(10, 2).astype(np.float32)
        rotated = self._rotate_curve(curve, 0.0)
        np.testing.assert_allclose(rotated, curve, atol=1e-5)

    # ── 13. _rotate_curve by 2*pi is identity (within tolerance) ──────────
    def test_rotate_curve_full_rotation(self):
        rng = np.random.RandomState(6)
        curve = rng.rand(10, 2).astype(np.float32)
        rotated = self._rotate_curve(curve, 2 * math.pi)
        np.testing.assert_allclose(rotated, curve, atol=1e-5)

    # ── 14. _fit_gamma_model returns GammaEdgeModel ───────────────────────
    def test_fit_gamma_model_type(self):
        _, entries = self._frags_and_entries(3)
        model = self._fit_gamma_model(entries)
        assert isinstance(model, self.GammaEdgeModel)

    # ── 15. _fit_gamma_model with empty entries returns model with defaults
    def test_fit_gamma_model_empty(self):
        model = self._fit_gamma_model([])
        assert model.k == 2.0
        assert model.theta == 0.5

    # ── 16. gamma_optimizer total_score is finite ─────────────────────────
    def test_total_score_finite(self):
        frags, entries = self._frags_and_entries(3)
        result = self.gamma_optimizer(frags, entries, n_iter=50, seed=42)
        assert math.isfinite(result.total_score)

    # ── 17. Two calls with same seed yield same total_score ────────────────
    def test_deterministic_with_seed(self):
        frags, entries = self._frags_and_entries(3)
        r1 = self.gamma_optimizer(frags, entries, n_iter=50, seed=7)
        r2 = self.gamma_optimizer(frags, entries, n_iter=50, seed=7)
        assert abs(r1.total_score - r2.total_score) < 1e-9


# ---------------------------------------------------------------------------
# TestHierarchical
# ---------------------------------------------------------------------------

class TestHierarchical:
    """Tests for puzzle_reconstruction.assembly.hierarchical"""

    @pytest.fixture(autouse=True)
    def _imports(self):
        from puzzle_reconstruction.assembly.hierarchical import (
            hierarchical_assembly,
            HierarchicalConfig,
            Cluster,
            _inter_cluster_score,
            _merge_clusters,
            single_linkage_score,
            average_linkage_score,
            complete_linkage_score,
        )
        self.hierarchical_assembly = hierarchical_assembly
        self.HierarchicalConfig = HierarchicalConfig
        self.Cluster = Cluster
        self._inter_cluster_score = _inter_cluster_score
        self._merge_clusters = _merge_clusters
        self.single_linkage_score = single_linkage_score
        self.average_linkage_score = average_linkage_score
        self.complete_linkage_score = complete_linkage_score

    def _frags_and_entries(self, n: int = 4, seed: int = 21):
        rng = np.random.RandomState(seed)
        frags = [_make_fragment(i, n_edges=2, seed=seed) for i in range(n)]
        entries = _make_compat_entries(frags, rng)
        return frags, entries

    def _edge_to_frag(self, frags):
        from puzzle_reconstruction.assembly.astar import _build_edge_to_frag
        return _build_edge_to_frag(frags)

    # ── 1. Empty fragments returns empty Assembly ──────────────────────────
    def test_empty_fragments(self):
        result = self.hierarchical_assembly([], [])
        assert isinstance(result, Assembly)
        assert result.placements == {}

    # ── 2. Single fragment is placed ──────────────────────────────────────
    def test_single_fragment(self):
        frags = [_make_fragment(0)]
        result = self.hierarchical_assembly(frags, [])
        assert 0 in result.placements

    # ── 3. All fragments are placed ────────────────────────────────────────
    def test_all_fragments_placed(self):
        frags, entries = self._frags_and_entries(4)
        result = self.hierarchical_assembly(frags, entries)
        for f in frags:
            assert f.fragment_id in result.placements

    # ── 4. Method tag is "hierarchical" ───────────────────────────────────
    def test_method_tag(self):
        frags, entries = self._frags_and_entries(3)
        result = self.hierarchical_assembly(frags, entries)
        assert result.method == "hierarchical"

    # ── 5. total_score is non-negative ────────────────────────────────────
    def test_total_score_non_negative(self):
        frags, entries = self._frags_and_entries(4)
        result = self.hierarchical_assembly(frags, entries)
        assert result.total_score >= 0.0

    # ── 6. single linkage returns max of list ─────────────────────────────
    def test_single_linkage(self):
        scores = [0.1, 0.9, 0.4]
        assert self.single_linkage_score(scores) == pytest.approx(0.9)

    # ── 7. average linkage returns mean of list ────────────────────────────
    def test_average_linkage(self):
        scores = [0.0, 1.0]
        assert self.average_linkage_score(scores) == pytest.approx(0.5)

    # ── 8. complete linkage returns min of list ────────────────────────────
    def test_complete_linkage(self):
        scores = [0.1, 0.9, 0.4]
        assert self.complete_linkage_score(scores) == pytest.approx(0.1)

    # ── 9. Linkage functions return 0.0 on empty list ─────────────────────
    def test_linkage_empty(self):
        assert self.single_linkage_score([]) == 0.0
        assert self.average_linkage_score([]) == 0.0
        assert self.complete_linkage_score([]) == 0.0

    # ── 10. HierarchicalConfig defaults ───────────────────────────────────
    def test_config_defaults(self):
        cfg = self.HierarchicalConfig()
        assert cfg.linkage == "average"
        assert cfg.min_merge_score == 0.0
        assert cfg.max_clusters == 1

    # ── 11. _inter_cluster_score returns -1 when no connecting entries ────
    def test_inter_cluster_no_entries(self):
        frags, entries = self._frags_and_entries(4)
        e2f = self._edge_to_frag(frags)
        ca = self.Cluster(0, {frags[0].fragment_id},
                          {frags[0].fragment_id: (np.array([0.0, 0.0]), 0.0)})
        cb = self.Cluster(1, {frags[1].fragment_id},
                          {frags[1].fragment_id: (np.array([100.0, 0.0]), 0.0)})
        score = self._inter_cluster_score(ca, cb, [], e2f)
        assert score == -1.0

    # ── 12. _inter_cluster_score with entries returns value in [-1, 1] ────
    def test_inter_cluster_with_entries(self):
        frags, entries = self._frags_and_entries(4)
        e2f = self._edge_to_frag(frags)
        ca = self.Cluster(0, {frags[0].fragment_id},
                          {frags[0].fragment_id: (np.array([0.0, 0.0]), 0.0)})
        cb = self.Cluster(1, {frags[1].fragment_id},
                          {frags[1].fragment_id: (np.array([100.0, 0.0]), 0.0)})
        score = self._inter_cluster_score(ca, cb, entries, e2f)
        assert -1.0 <= score <= 1.0

    # ── 13. _merge_clusters combines fragment ids ──────────────────────────
    def test_merge_clusters_ids(self):
        frags, entries = self._frags_and_entries(4)
        e2f = self._edge_to_frag(frags)
        ca = self.Cluster(0, {frags[0].fragment_id},
                          {frags[0].fragment_id: (np.array([0.0, 0.0]), 0.0)})
        cb = self.Cluster(1, {frags[1].fragment_id},
                          {frags[1].fragment_id: (np.array([100.0, 0.0]), 0.0)})
        merged = self._merge_clusters(ca, cb, entries, e2f,
                                      new_id=99, merge_score=0.5)
        assert frags[0].fragment_id in merged.fragment_ids
        assert frags[1].fragment_id in merged.fragment_ids
        assert merged.cluster_id == 99

    # ── 14. _merge_clusters total_score includes merge_score ──────────────
    def test_merge_clusters_score(self):
        frags, entries = self._frags_and_entries(4)
        e2f = self._edge_to_frag(frags)
        ca = self.Cluster(0, {frags[0].fragment_id},
                          {frags[0].fragment_id: (np.array([0.0, 0.0]), 0.0)},
                          total_score=1.0)
        cb = self.Cluster(1, {frags[1].fragment_id},
                          {frags[1].fragment_id: (np.array([100.0, 0.0]), 0.0)},
                          total_score=2.0)
        merged = self._merge_clusters(ca, cb, entries, e2f,
                                      new_id=99, merge_score=0.3)
        assert abs(merged.total_score - (1.0 + 2.0 + 0.3)) < 1e-9

    # ── 15. single- and average-linkage both produce valid assemblies ──────
    def test_single_vs_average_linkage(self):
        frags, entries = self._frags_and_entries(5)
        cfg_single = self.HierarchicalConfig(linkage="single")
        cfg_avg    = self.HierarchicalConfig(linkage="average")
        r1 = self.hierarchical_assembly(frags, entries, cfg=cfg_single)
        r2 = self.hierarchical_assembly(frags, entries, cfg=cfg_avg)
        assert len(r1.placements) == 5
        assert len(r2.placements) == 5

    # ── 16. max_clusters=2 still covers all fragment placements ───────────
    def test_max_clusters_two(self):
        frags, entries = self._frags_and_entries(4)
        cfg = self.HierarchicalConfig(max_clusters=2)
        result = self.hierarchical_assembly(frags, entries, cfg=cfg)
        assert len(result.placements) == 4

    # ── 17. min_merge_score above max score stops immediately ─────────────
    def test_min_merge_score_no_merges(self):
        frags, entries = self._frags_and_entries(3)
        cfg = self.HierarchicalConfig(min_merge_score=999.0)
        result = self.hierarchical_assembly(frags, entries, cfg=cfg)
        assert len(result.placements) == 3

    # ── 18. Placements have 2-element position arrays ─────────────────────
    def test_placement_position_shape(self):
        frags, entries = self._frags_and_entries(3)
        result = self.hierarchical_assembly(frags, entries)
        for fid, (pos, rot) in result.placements.items():
            assert pos.shape == (2,)
