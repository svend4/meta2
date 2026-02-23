"""Extra tests for puzzle_reconstruction.assembly.sequence_planner."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.assembly.sequence_planner import (
    PlanConfig,
    PlacementPlan,
    PlacementStep,
    batch_build_plans,
    build_placement_plan,
    export_plan,
    filter_plan,
    reorder_plan,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _scores(*pairs):
    return {(min(a, b), max(a, b)): s for (a, b), s in pairs}


def _plan(n=4, strategy="greedy"):
    ids = list(range(n))
    sc = {(i, j): round(1.0 - abs(i - j) * 0.1, 1)
          for i in range(n) for j in range(i + 1, n)}
    return build_placement_plan(ids, sc, PlanConfig(strategy=strategy))


# ─── TestPlanConfigExtra ─────────────────────────────────────────────────────

class TestPlanConfigExtra:
    def test_default_strategy(self):
        assert PlanConfig().strategy == "greedy"

    def test_bfs_strategy(self):
        cfg = PlanConfig(strategy="bfs")
        assert cfg.strategy == "bfs"

    def test_anchor_id_none_default(self):
        assert PlanConfig().anchor_id is None

    def test_anchor_id_set(self):
        cfg = PlanConfig(anchor_id=5)
        assert cfg.anchor_id == 5

    def test_allow_revisit_default_false(self):
        assert PlanConfig().allow_revisit is False

    def test_allow_revisit_true(self):
        cfg = PlanConfig(allow_revisit=True)
        assert cfg.allow_revisit is True

    def test_min_score_half(self):
        cfg = PlanConfig(min_score=0.5)
        assert cfg.min_score == pytest.approx(0.5)

    def test_min_score_one_ok(self):
        cfg = PlanConfig(min_score=1.0)
        assert cfg.min_score == pytest.approx(1.0)


# ─── TestPlacementStepExtra ──────────────────────────────────────────────────

class TestPlacementStepExtra:
    def test_step_zero_is_anchor(self):
        s = PlacementStep(step=0, fragment_id=0, score=0.0)
        assert s.is_anchor is True

    def test_step_one_not_anchor(self):
        s = PlacementStep(step=1, fragment_id=1, score=0.8)
        assert s.is_anchor is False

    def test_high_step_not_anchor(self):
        s = PlacementStep(step=99, fragment_id=0, score=0.5)
        assert s.is_anchor is False

    def test_anchored_by_default_empty(self):
        s = PlacementStep(step=0, fragment_id=0, score=0.0)
        assert s.anchored_by == []

    def test_anchored_by_stored(self):
        s = PlacementStep(step=2, fragment_id=3, score=0.6, anchored_by=[0, 1])
        assert s.anchored_by == [0, 1]

    def test_score_stored(self):
        s = PlacementStep(step=1, fragment_id=2, score=0.77)
        assert s.score == pytest.approx(0.77)

    def test_fragment_id_stored(self):
        s = PlacementStep(step=1, fragment_id=42, score=0.5)
        assert s.fragment_id == 42


# ─── TestPlacementPlanExtra ──────────────────────────────────────────────────

class TestPlacementPlanExtra:
    def test_placement_order_length(self):
        plan = _plan(4)
        assert len(plan.placement_order) == 4

    def test_anchor_first(self):
        plan = _plan(4)
        assert plan.steps[0].is_anchor is True

    def test_coverage_full(self):
        plan = _plan(4)
        assert plan.coverage == pytest.approx(1.0)

    def test_mean_score_in_range(self):
        plan = _plan(4)
        assert 0.0 <= plan.mean_score <= 1.0

    def test_strategy_stored(self):
        plan = _plan(strategy="bfs")
        assert plan.strategy == "bfs"

    def test_steps_sequential(self):
        plan = _plan(4)
        for i, s in enumerate(plan.steps):
            assert s.step == i

    def test_n_fragments_stored(self):
        plan = _plan(5)
        assert plan.n_fragments == 5


# ─── TestBuildPlacementPlanExtra ─────────────────────────────────────────────

class TestBuildPlacementPlanExtra:
    def test_two_fragments(self):
        plan = build_placement_plan([0, 1], _scores(((0, 1), 0.9)))
        assert plan.n_placed == 2

    def test_five_fragments(self):
        ids = list(range(5))
        sc = {(i, j): 0.8 for i in range(5) for j in range(i + 1, 5)}
        plan = build_placement_plan(ids, sc)
        assert plan.n_placed == 5

    def test_bfs_strategy(self):
        plan = build_placement_plan([0, 1, 2],
                                    _scores(((0, 1), 0.9), ((1, 2), 0.8)),
                                    PlanConfig(strategy="bfs"))
        assert plan.strategy == "bfs"

    def test_anchor_id_respected(self):
        cfg = PlanConfig(anchor_id=2)
        plan = build_placement_plan([0, 1, 2],
                                    _scores(((0, 2), 0.9), ((1, 2), 0.8)),
                                    cfg)
        assert plan.steps[0].fragment_id == 2

    def test_invalid_anchor_fallback(self):
        cfg = PlanConfig(anchor_id=999)
        plan = build_placement_plan([0, 1], _scores(((0, 1), 0.8)), cfg)
        assert plan.steps[0].fragment_id == 0

    def test_empty_returns_empty(self):
        plan = build_placement_plan([], {})
        assert plan.n_placed == 0
        assert plan.steps == []

    def test_single_fragment(self):
        plan = build_placement_plan([7], {})
        assert plan.n_placed == 1
        assert plan.steps[0].fragment_id == 7
        assert plan.steps[0].is_anchor


# ─── TestReorderPlanExtra ────────────────────────────────────────────────────

class TestReorderPlanExtra:
    def test_priority_fragment_first(self):
        plan = _plan(4)
        last_id = plan.steps[-1].fragment_id
        reordered = reorder_plan(plan, [last_id])
        assert reordered.steps[0].fragment_id == last_id

    def test_preserves_all_ids(self):
        plan = _plan(5)
        reordered = reorder_plan(plan, [2, 4])
        ids_orig = set(plan.placement_order)
        ids_new = set(reordered.placement_order)
        assert ids_orig == ids_new

    def test_empty_priority_unchanged(self):
        plan = _plan(4)
        reordered = reorder_plan(plan, [])
        assert reordered.placement_order == plan.placement_order

    def test_steps_reindexed(self):
        plan = _plan(4)
        reordered = reorder_plan(plan, [plan.steps[-1].fragment_id])
        for i, s in enumerate(reordered.steps):
            assert s.step == i

    def test_unknown_priority_ignored(self):
        plan = _plan(4)
        reordered = reorder_plan(plan, [999])
        assert len(reordered.steps) == len(plan.steps)


# ─── TestFilterPlanExtra ─────────────────────────────────────────────────────

class TestFilterPlanExtra:
    def test_zero_threshold_all_kept(self):
        plan = _plan(4)
        filtered = filter_plan(plan, min_score=0.0)
        assert len(filtered.steps) == len(plan.steps)

    def test_anchor_always_kept(self):
        plan = _plan(4)
        filtered = filter_plan(plan, min_score=1.0)
        assert len(filtered.steps) >= 1
        assert filtered.steps[0].is_anchor

    def test_steps_reindexed_after_filter(self):
        plan = _plan(4)
        filtered = filter_plan(plan, min_score=0.7)
        for i, s in enumerate(filtered.steps):
            assert s.step == i

    def test_high_threshold_keeps_anchor(self):
        plan = _plan(4)
        filtered = filter_plan(plan, min_score=0.99)
        assert filtered.steps[0].is_anchor

    def test_no_filter_full_coverage(self):
        plan = _plan(4)
        filtered = filter_plan(plan, min_score=0.0)
        assert filtered.n_placed == plan.n_placed


# ─── TestExportPlanExtra ─────────────────────────────────────────────────────

class TestExportPlanExtra:
    def test_length_matches_steps(self):
        plan = _plan(4)
        exported = export_plan(plan)
        assert len(exported) == len(plan.steps)

    def test_keys_present(self):
        plan = _plan(4)
        for rec in export_plan(plan):
            assert "step" in rec
            assert "fragment_id" in rec
            assert "score" in rec
            assert "anchored_by" in rec

    def test_anchored_by_list(self):
        plan = _plan(4)
        for rec in export_plan(plan):
            assert isinstance(rec["anchored_by"], list)

    def test_step_values_sequential(self):
        plan = _plan(4)
        steps = [r["step"] for r in export_plan(plan)]
        assert steps == list(range(len(steps)))

    def test_fragment_ids_all_present(self):
        plan = _plan(4)
        exported_ids = {r["fragment_id"] for r in export_plan(plan)}
        plan_ids = set(plan.placement_order)
        assert exported_ids == plan_ids


# ─── TestBatchBuildPlansExtra ────────────────────────────────────────────────

class TestBatchBuildPlansExtra:
    def test_single_group(self):
        plans = batch_build_plans([[0, 1, 2]],
                                  [_scores(((0, 1), 0.9), ((1, 2), 0.7))])
        assert len(plans) == 1

    def test_three_groups(self):
        plans = batch_build_plans(
            [[0, 1], [2, 3], [4, 5]],
            [_scores(((0, 1), 0.8)),
             _scores(((2, 3), 0.7)),
             _scores(((4, 5), 0.6))],
        )
        assert len(plans) == 3

    def test_all_placed_minimum_one(self):
        plans = batch_build_plans(
            [[0, 1], [2, 3]],
            [{}, {}],
        )
        for plan in plans:
            assert plan.n_placed >= 1

    def test_custom_cfg_propagated(self):
        cfg = PlanConfig(strategy="bfs")
        plans = batch_build_plans([[0, 1]], [_scores(((0, 1), 0.9))], cfg)
        assert plans[0].strategy == "bfs"

    def test_empty_batch(self):
        assert batch_build_plans([], []) == []
