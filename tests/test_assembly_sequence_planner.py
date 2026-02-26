"""Tests for puzzle_reconstruction/assembly/sequence_planner.py."""
import pytest
import numpy as np

from puzzle_reconstruction.assembly.sequence_planner import (
    PlanConfig,
    PlacementStep,
    PlacementPlan,
    _best_candidate,
    _next_bfs,
    build_placement_plan,
    reorder_plan,
    filter_plan,
    export_plan,
    batch_build_plans,
)


# ─── PlanConfig ───────────────────────────────────────────────────────────────

class TestPlanConfig:
    def test_default_values(self):
        cfg = PlanConfig()
        assert cfg.strategy == "greedy"
        assert cfg.anchor_id is None
        assert cfg.min_score == 0.0
        assert cfg.allow_revisit is False

    def test_greedy_strategy_ok(self):
        cfg = PlanConfig(strategy="greedy")
        assert cfg.strategy == "greedy"

    def test_bfs_strategy_ok(self):
        cfg = PlanConfig(strategy="bfs")
        assert cfg.strategy == "bfs"

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            PlanConfig(strategy="unknown")

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            PlanConfig(min_score=-0.1)

    def test_zero_min_score_ok(self):
        cfg = PlanConfig(min_score=0.0)
        assert cfg.min_score == 0.0

    def test_anchor_id_set(self):
        cfg = PlanConfig(anchor_id=5)
        assert cfg.anchor_id == 5


# ─── PlacementStep ────────────────────────────────────────────────────────────

class TestPlacementStep:
    def test_basic_step(self):
        step = PlacementStep(step=0, fragment_id=1, score=0.5)
        assert step.step == 0
        assert step.fragment_id == 1
        assert abs(step.score - 0.5) < 1e-9

    def test_default_anchored_by_empty(self):
        step = PlacementStep(step=1, fragment_id=2, score=0.3)
        assert step.anchored_by == []

    def test_is_anchor_true_for_step_zero(self):
        step = PlacementStep(step=0, fragment_id=0, score=0.0)
        assert step.is_anchor is True

    def test_is_anchor_false_for_nonzero_step(self):
        step = PlacementStep(step=1, fragment_id=0, score=0.5)
        assert step.is_anchor is False

    def test_negative_step_raises(self):
        with pytest.raises(ValueError):
            PlacementStep(step=-1, fragment_id=0, score=0.0)

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            PlacementStep(step=0, fragment_id=-1, score=0.0)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            PlacementStep(step=0, fragment_id=0, score=-0.1)

    def test_anchored_by_stored(self):
        step = PlacementStep(step=1, fragment_id=3, score=0.7, anchored_by=[0, 1])
        assert step.anchored_by == [0, 1]


# ─── PlacementPlan ────────────────────────────────────────────────────────────

class TestPlacementPlan:
    def _make_plan(self):
        steps = [
            PlacementStep(step=0, fragment_id=0, score=0.0),
            PlacementStep(step=1, fragment_id=1, score=0.5),
            PlacementStep(step=2, fragment_id=2, score=0.8),
        ]
        return PlacementPlan(steps=steps, n_fragments=3, n_placed=3, strategy="greedy")

    def test_placement_order(self):
        plan = self._make_plan()
        assert plan.placement_order == [0, 1, 2]

    def test_coverage_full(self):
        plan = self._make_plan()
        assert plan.coverage == 1.0

    def test_coverage_zero_when_no_fragments(self):
        plan = PlacementPlan(steps=[], n_fragments=0, n_placed=0, strategy="greedy")
        assert plan.coverage == 0.0

    def test_mean_score_excludes_anchor(self):
        plan = self._make_plan()
        # steps 1 and 2 have scores 0.5 and 0.8
        assert abs(plan.mean_score - 0.65) < 1e-9

    def test_mean_score_zero_only_anchor(self):
        steps = [PlacementStep(step=0, fragment_id=0, score=0.0)]
        plan = PlacementPlan(steps=steps, n_fragments=1, n_placed=1, strategy="greedy")
        assert plan.mean_score == 0.0

    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            PlacementPlan(steps=[], n_fragments=-1, n_placed=0, strategy="greedy")

    def test_negative_n_placed_raises(self):
        with pytest.raises(ValueError):
            PlacementPlan(steps=[], n_fragments=0, n_placed=-1, strategy="greedy")


# ─── _best_candidate ──────────────────────────────────────────────────────────

class TestBestCandidate:
    def test_returns_highest_score_candidate(self):
        remaining = {1, 2, 3}
        placed = {0}
        scores = {(0, 1): 0.3, (0, 2): 0.8, (0, 3): 0.5}
        fid, sc, anchors = _best_candidate(remaining, placed, scores, min_score=0.0)
        assert fid == 2
        assert abs(sc - 0.8) < 1e-9

    def test_returns_none_when_no_candidates_above_min_score(self):
        remaining = {1}
        placed = {0}
        scores = {(0, 1): 0.1}
        fid, sc, anchors = _best_candidate(remaining, placed, scores, min_score=0.5)
        assert fid is None

    def test_empty_remaining(self):
        fid, sc, anchors = _best_candidate(set(), {0}, {}, min_score=0.0)
        assert fid is None

    def test_anchors_list_contains_placed_id(self):
        remaining = {1}
        placed = {0}
        scores = {(0, 1): 0.7}
        fid, sc, anchors = _best_candidate(remaining, placed, scores, min_score=0.0)
        assert 0 in anchors


# ─── _next_bfs ────────────────────────────────────────────────────────────────

class TestNextBfs:
    def test_returns_first_valid_candidate(self):
        remaining = {1, 2, 3}
        placed = {0}
        scores = {(0, 1): 0.6, (0, 2): 0.8}
        fid, sc, anchors = _next_bfs(remaining, placed, scores, min_score=0.0)
        # BFS returns lowest id with score >= min_score
        assert fid == 1

    def test_fallback_to_unconnected_fragment(self):
        remaining = {5}
        placed = {0}
        scores = {}
        fid, sc, anchors = _next_bfs(remaining, placed, scores, min_score=0.0)
        assert fid == 5
        assert sc == 0.0

    def test_empty_remaining_returns_none(self):
        fid, sc, anchors = _next_bfs(set(), {0}, {}, min_score=0.0)
        assert fid is None


# ─── build_placement_plan ─────────────────────────────────────────────────────

class TestBuildPlacementPlan:
    def test_empty_fragment_ids(self):
        plan = build_placement_plan([], {})
        assert isinstance(plan, PlacementPlan)
        assert plan.n_placed == 0

    def test_single_fragment(self):
        plan = build_placement_plan([42], {})
        assert len(plan.steps) == 1
        assert plan.steps[0].fragment_id == 42
        assert plan.steps[0].is_anchor

    def test_all_fragments_placed(self):
        ids = [0, 1, 2, 3]
        scores = {(0, 1): 0.5, (1, 2): 0.6, (2, 3): 0.7}
        plan = build_placement_plan(ids, scores)
        assert plan.n_placed == 4

    def test_anchor_is_first_step(self):
        ids = [0, 1, 2]
        plan = build_placement_plan(ids, {})
        assert plan.steps[0].is_anchor

    def test_custom_anchor_id(self):
        ids = [0, 1, 2, 3]
        cfg = PlanConfig(anchor_id=2)
        plan = build_placement_plan(ids, {}, cfg)
        assert plan.steps[0].fragment_id == 2

    def test_bfs_strategy(self):
        ids = [0, 1, 2]
        scores = {(0, 1): 0.5, (0, 2): 0.8}
        cfg = PlanConfig(strategy="bfs")
        plan = build_placement_plan(ids, scores, cfg)
        assert plan.strategy == "bfs"
        assert plan.n_placed == 3

    def test_greedy_strategy(self):
        ids = [0, 1, 2]
        scores = {(0, 1): 0.5, (0, 2): 0.8}
        cfg = PlanConfig(strategy="greedy")
        plan = build_placement_plan(ids, scores, cfg)
        assert plan.strategy == "greedy"

    def test_n_fragments_matches_input(self):
        ids = [10, 20, 30]
        plan = build_placement_plan(ids, {})
        assert plan.n_fragments == 3

    def test_min_score_filters_low_score(self):
        ids = [0, 1, 2]
        scores = {(0, 1): 0.1, (0, 2): 0.9}
        cfg = PlanConfig(min_score=0.5)
        plan = build_placement_plan(ids, scores, cfg)
        # Only fragment 2 passes min_score threshold
        assert any(s.fragment_id == 2 for s in plan.steps if not s.is_anchor)

    def test_coverage_full_when_all_placed(self):
        ids = [0, 1, 2]
        scores = {(0, 1): 0.5, (1, 2): 0.6}
        plan = build_placement_plan(ids, scores)
        assert plan.coverage == 1.0


# ─── reorder_plan ─────────────────────────────────────────────────────────────

class TestReorderPlan:
    def _make_plan(self):
        steps = [
            PlacementStep(step=0, fragment_id=0, score=0.0),
            PlacementStep(step=1, fragment_id=1, score=0.5),
            PlacementStep(step=2, fragment_id=2, score=0.8),
            PlacementStep(step=3, fragment_id=3, score=0.6),
        ]
        return PlacementPlan(steps=steps, n_fragments=4, n_placed=4, strategy="greedy")

    def test_priority_ids_go_first(self):
        plan = self._make_plan()
        new_plan = reorder_plan(plan, priority=[3, 2])
        assert new_plan.steps[0].fragment_id == 3
        assert new_plan.steps[1].fragment_id == 2

    def test_all_steps_preserved(self):
        plan = self._make_plan()
        new_plan = reorder_plan(plan, priority=[2])
        ids = {s.fragment_id for s in new_plan.steps}
        assert ids == {0, 1, 2, 3}

    def test_step_indices_renumbered(self):
        plan = self._make_plan()
        new_plan = reorder_plan(plan, priority=[3])
        for i, step in enumerate(new_plan.steps):
            assert step.step == i

    def test_empty_priority(self):
        plan = self._make_plan()
        new_plan = reorder_plan(plan, priority=[])
        assert [s.fragment_id for s in new_plan.steps] == [0, 1, 2, 3]


# ─── filter_plan ──────────────────────────────────────────────────────────────

class TestFilterPlan:
    def _make_plan(self):
        steps = [
            PlacementStep(step=0, fragment_id=0, score=0.0),
            PlacementStep(step=1, fragment_id=1, score=0.2),
            PlacementStep(step=2, fragment_id=2, score=0.8),
            PlacementStep(step=3, fragment_id=3, score=0.5),
        ]
        return PlacementPlan(steps=steps, n_fragments=4, n_placed=4, strategy="greedy")

    def test_anchor_always_kept(self):
        plan = self._make_plan()
        filtered = filter_plan(plan, min_score=0.9)
        assert filtered.steps[0].fragment_id == 0

    def test_low_score_steps_filtered(self):
        plan = self._make_plan()
        filtered = filter_plan(plan, min_score=0.5)
        ids = {s.fragment_id for s in filtered.steps}
        assert 1 not in ids  # score 0.2 < 0.5

    def test_high_score_steps_kept(self):
        plan = self._make_plan()
        filtered = filter_plan(plan, min_score=0.5)
        ids = {s.fragment_id for s in filtered.steps}
        assert 2 in ids  # score 0.8 >= 0.5

    def test_indices_renumbered(self):
        plan = self._make_plan()
        filtered = filter_plan(plan, min_score=0.5)
        for i, step in enumerate(filtered.steps):
            assert step.step == i

    def test_negative_min_score_raises(self):
        plan = self._make_plan()
        with pytest.raises(ValueError):
            filter_plan(plan, min_score=-0.1)


# ─── export_plan ──────────────────────────────────────────────────────────────

class TestExportPlan:
    def test_returns_list_of_dicts(self):
        steps = [PlacementStep(step=0, fragment_id=0, score=0.0)]
        plan = PlacementPlan(steps=steps, n_fragments=1, n_placed=1, strategy="greedy")
        result = export_plan(plan)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_dict_keys_present(self):
        steps = [PlacementStep(step=0, fragment_id=5, score=0.3, anchored_by=[1, 2])]
        plan = PlacementPlan(steps=steps, n_fragments=1, n_placed=1, strategy="greedy")
        result = export_plan(plan)
        d = result[0]
        assert "step" in d
        assert "fragment_id" in d
        assert "score" in d
        assert "anchored_by" in d

    def test_empty_plan(self):
        plan = PlacementPlan(steps=[], n_fragments=0, n_placed=0, strategy="greedy")
        result = export_plan(plan)
        assert result == []

    def test_values_correct(self):
        steps = [PlacementStep(step=2, fragment_id=7, score=0.6, anchored_by=[3])]
        plan = PlacementPlan(steps=steps, n_fragments=1, n_placed=1, strategy="greedy")
        result = export_plan(plan)
        assert result[0]["step"] == 2
        assert result[0]["fragment_id"] == 7
        assert abs(result[0]["score"] - 0.6) < 1e-9
        assert result[0]["anchored_by"] == [3]


# ─── batch_build_plans ────────────────────────────────────────────────────────

class TestBatchBuildPlans:
    def test_returns_list_of_plans(self):
        id_lists = [[0, 1, 2], [3, 4]]
        score_dicts = [{(0, 1): 0.5}, {(3, 4): 0.6}]
        plans = batch_build_plans(id_lists, score_dicts)
        assert len(plans) == 2
        assert all(isinstance(p, PlacementPlan) for p in plans)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            batch_build_plans([[0, 1]], [{}, {}])

    def test_empty_inputs(self):
        plans = batch_build_plans([], [])
        assert plans == []

    def test_each_plan_correct_n_fragments(self):
        id_lists = [[0, 1, 2], [5, 6]]
        score_dicts = [{}, {}]
        plans = batch_build_plans(id_lists, score_dicts)
        assert plans[0].n_fragments == 3
        assert plans[1].n_fragments == 2
