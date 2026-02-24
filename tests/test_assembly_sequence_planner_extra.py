"""Extra tests for puzzle_reconstruction/assembly/sequence_planner.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.assembly.sequence_planner import (
    PlanConfig,
    PlacementStep,
    PlacementPlan,
    build_placement_plan,
    reorder_plan,
    filter_plan,
    export_plan,
    batch_build_plans,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _scores():
    """Scores dict for fragments 0,1,2."""
    return {(0, 1): 0.8, (0, 2): 0.5, (1, 2): 0.9}


# ─── PlanConfig ─────────────────────────────────────────────────────────────

class TestPlanConfigExtra:
    def test_defaults(self):
        c = PlanConfig()
        assert c.strategy == "greedy"
        assert c.anchor_id is None
        assert c.min_score == 0.0
        assert c.allow_revisit is False

    def test_valid_strategies(self):
        for s in ("greedy", "bfs"):
            PlanConfig(strategy=s)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            PlanConfig(strategy="bad")

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            PlanConfig(min_score=-0.1)


# ─── PlacementStep ──────────────────────────────────────────────────────────

class TestPlacementStepExtra:
    def test_valid(self):
        ps = PlacementStep(step=0, fragment_id=1, score=0.5)
        assert ps.is_anchor is True

    def test_not_anchor(self):
        ps = PlacementStep(step=1, fragment_id=2, score=0.8)
        assert ps.is_anchor is False

    def test_negative_step_raises(self):
        with pytest.raises(ValueError):
            PlacementStep(step=-1, fragment_id=0, score=0.0)

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            PlacementStep(step=0, fragment_id=-1, score=0.0)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            PlacementStep(step=0, fragment_id=0, score=-0.1)


# ─── PlacementPlan ──────────────────────────────────────────────────────────

class TestPlacementPlanExtra:
    def test_valid(self):
        steps = [PlacementStep(step=0, fragment_id=0, score=0.0)]
        pp = PlacementPlan(steps=steps, n_fragments=1, n_placed=1, strategy="greedy")
        assert pp.placement_order == [0]
        assert pp.coverage == pytest.approx(1.0)

    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            PlacementPlan(steps=[], n_fragments=-1, n_placed=0, strategy="greedy")

    def test_negative_n_placed_raises(self):
        with pytest.raises(ValueError):
            PlacementPlan(steps=[], n_fragments=0, n_placed=-1, strategy="greedy")

    def test_coverage_zero_fragments(self):
        pp = PlacementPlan(steps=[], n_fragments=0, n_placed=0, strategy="greedy")
        assert pp.coverage == pytest.approx(0.0)

    def test_mean_score(self):
        steps = [
            PlacementStep(step=0, fragment_id=0, score=0.0),
            PlacementStep(step=1, fragment_id=1, score=0.8),
            PlacementStep(step=2, fragment_id=2, score=0.6),
        ]
        pp = PlacementPlan(steps=steps, n_fragments=3, n_placed=3, strategy="greedy")
        assert pp.mean_score == pytest.approx(0.7)  # (0.8 + 0.6) / 2


# ─── build_placement_plan ──────────────────────────────────────────────────

class TestBuildPlacementPlanExtra:
    def test_empty(self):
        pp = build_placement_plan([], {})
        assert pp.n_fragments == 0
        assert pp.n_placed == 0

    def test_single(self):
        pp = build_placement_plan([0], {})
        assert pp.n_placed == 1
        assert pp.placement_order == [0]

    def test_greedy(self):
        pp = build_placement_plan([0, 1, 2], _scores())
        assert pp.n_placed == 3
        assert pp.strategy == "greedy"

    def test_bfs(self):
        cfg = PlanConfig(strategy="bfs")
        pp = build_placement_plan([0, 1, 2], _scores(), cfg)
        assert pp.n_placed == 3
        assert pp.strategy == "bfs"

    def test_custom_anchor(self):
        cfg = PlanConfig(anchor_id=2)
        pp = build_placement_plan([0, 1, 2], _scores(), cfg)
        assert pp.placement_order[0] == 2

    def test_coverage(self):
        pp = build_placement_plan([0, 1, 2], _scores())
        assert pp.coverage == pytest.approx(1.0)


# ─── reorder_plan ───────────────────────────────────────────────────────────

class TestReorderPlanExtra:
    def test_reorder(self):
        pp = build_placement_plan([0, 1, 2], _scores())
        reordered = reorder_plan(pp, priority=[2, 0])
        assert reordered.placement_order[0] == 2
        assert reordered.placement_order[1] == 0

    def test_noop(self):
        pp = build_placement_plan([0, 1], _scores())
        original_order = pp.placement_order.copy()
        reordered = reorder_plan(pp, priority=[])
        assert reordered.placement_order == original_order


# ─── filter_plan ────────────────────────────────────────────────────────────

class TestFilterPlanExtra:
    def test_keeps_anchor(self):
        pp = build_placement_plan([0, 1, 2], _scores())
        filtered = filter_plan(pp, min_score=999.0)
        assert len(filtered.steps) >= 1  # Anchor always kept
        assert filtered.steps[0].is_anchor

    def test_negative_min_score_raises(self):
        pp = build_placement_plan([0], {})
        with pytest.raises(ValueError):
            filter_plan(pp, min_score=-0.1)

    def test_keeps_all_high_scores(self):
        pp = build_placement_plan([0, 1, 2], _scores())
        filtered = filter_plan(pp, min_score=0.0)
        assert filtered.n_placed == pp.n_placed


# ─── export_plan ────────────────────────────────────────────────────────────

class TestExportPlanExtra:
    def test_basic(self):
        pp = build_placement_plan([0, 1], _scores())
        exported = export_plan(pp)
        assert len(exported) == 2
        assert "step" in exported[0]
        assert "fragment_id" in exported[0]
        assert "score" in exported[0]

    def test_empty(self):
        pp = PlacementPlan(steps=[], n_fragments=0, n_placed=0, strategy="greedy")
        assert export_plan(pp) == []


# ─── batch_build_plans ──────────────────────────────────────────────────────

class TestBatchBuildPlansExtra:
    def test_empty(self):
        assert batch_build_plans([], []) == []

    def test_length(self):
        results = batch_build_plans(
            [[0, 1], [2, 3]],
            [_scores(), {(2, 3): 0.7}],
        )
        assert len(results) == 2

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            batch_build_plans([[0]], [])

    def test_result_type(self):
        results = batch_build_plans([[0, 1]], [_scores()])
        assert isinstance(results[0], PlacementPlan)
