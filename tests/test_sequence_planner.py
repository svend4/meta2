"""Тесты для puzzle_reconstruction.assembly.sequence_planner."""
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

def _scores(*pairs):
    """Создать словарь оценок из кортежей ((a, b), score)."""
    return {(min(a, b), max(a, b)): s for (a, b), s in pairs}


def _simple_plan(strategy="greedy"):
    ids = [0, 1, 2, 3]
    scores = _scores(
        ((0, 1), 0.9), ((0, 2), 0.7), ((0, 3), 0.5),
        ((1, 2), 0.8), ((1, 3), 0.4),
        ((2, 3), 0.6),
    )
    return build_placement_plan(ids, scores, PlanConfig(strategy=strategy))


# ─── TestPlanConfig ───────────────────────────────────────────────────────────

class TestPlanConfig:
    def test_defaults(self):
        cfg = PlanConfig()
        assert cfg.strategy == "greedy"
        assert cfg.anchor_id is None
        assert cfg.min_score == 0.0
        assert cfg.allow_revisit is False

    def test_valid_greedy(self):
        cfg = PlanConfig(strategy="greedy")
        assert cfg.strategy == "greedy"

    def test_valid_bfs(self):
        cfg = PlanConfig(strategy="bfs")
        assert cfg.strategy == "bfs"

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            PlanConfig(strategy="random")

    def test_invalid_min_score_neg(self):
        with pytest.raises(ValueError):
            PlanConfig(min_score=-0.1)

    def test_min_score_zero_ok(self):
        cfg = PlanConfig(min_score=0.0)
        assert cfg.min_score == 0.0

    def test_anchor_id_set(self):
        cfg = PlanConfig(anchor_id=3)
        assert cfg.anchor_id == 3


# ─── TestPlacementStep ────────────────────────────────────────────────────────

class TestPlacementStep:
    def test_basic(self):
        s = PlacementStep(step=0, fragment_id=5, score=0.0)
        assert s.step == 0
        assert s.fragment_id == 5
        assert s.score == 0.0
        assert s.anchored_by == []

    def test_is_anchor_true(self):
        s = PlacementStep(step=0, fragment_id=0, score=0.0)
        assert s.is_anchor is True

    def test_is_anchor_false(self):
        s = PlacementStep(step=1, fragment_id=1, score=0.8)
        assert s.is_anchor is False

    def test_anchored_by(self):
        s = PlacementStep(step=2, fragment_id=3, score=0.7,
                          anchored_by=[0, 1])
        assert s.anchored_by == [0, 1]

    def test_invalid_step_neg(self):
        with pytest.raises(ValueError):
            PlacementStep(step=-1, fragment_id=0, score=0.0)

    def test_invalid_fragment_id_neg(self):
        with pytest.raises(ValueError):
            PlacementStep(step=0, fragment_id=-1, score=0.0)

    def test_invalid_score_neg(self):
        with pytest.raises(ValueError):
            PlacementStep(step=0, fragment_id=0, score=-0.1)


# ─── TestPlacementPlan ────────────────────────────────────────────────────────

class TestPlacementPlan:
    def test_placement_order(self):
        plan = _simple_plan()
        order = plan.placement_order
        assert len(order) == 4
        assert order[0] == 0  # anchor

    def test_coverage_full(self):
        plan = _simple_plan()
        assert abs(plan.coverage - 1.0) < 1e-9

    def test_coverage_partial(self):
        ids = [0, 1, 2]
        scores = _scores(((0, 1), 0.8))
        cfg = PlanConfig(min_score=0.9)  # 2 and 0 связаны score < threshold
        plan = build_placement_plan(ids, scores, cfg)
        assert 0.0 <= plan.coverage <= 1.0

    def test_coverage_zero_fragments(self):
        plan = PlacementPlan(steps=[], n_fragments=0, n_placed=0, strategy="greedy")
        assert plan.coverage == 0.0

    def test_mean_score_non_anchor(self):
        plan = _simple_plan()
        ms = plan.mean_score
        assert ms > 0.0
        assert ms <= 1.0

    def test_mean_score_anchor_only(self):
        plan = PlacementPlan(
            steps=[PlacementStep(0, 0, 0.0)],
            n_fragments=1, n_placed=1, strategy="greedy",
        )
        assert plan.mean_score == 0.0

    def test_invalid_n_fragments_neg(self):
        with pytest.raises(ValueError):
            PlacementPlan(steps=[], n_fragments=-1, n_placed=0, strategy="greedy")

    def test_invalid_n_placed_neg(self):
        with pytest.raises(ValueError):
            PlacementPlan(steps=[], n_fragments=0, n_placed=-1, strategy="greedy")


# ─── TestBuildPlacementPlan ───────────────────────────────────────────────────

class TestBuildPlacementPlan:
    def test_basic_greedy(self):
        plan = _simple_plan("greedy")
        assert plan.strategy == "greedy"
        assert plan.n_placed == 4

    def test_basic_bfs(self):
        plan = _simple_plan("bfs")
        assert plan.strategy == "bfs"
        assert plan.n_placed == 4

    def test_anchor_is_first(self):
        ids = [0, 1, 2]
        scores = _scores(((0, 1), 0.8), ((1, 2), 0.7))
        cfg = PlanConfig(anchor_id=0)
        plan = build_placement_plan(ids, scores, cfg)
        assert plan.steps[0].fragment_id == 0
        assert plan.steps[0].is_anchor is True

    def test_custom_anchor_id(self):
        ids = [0, 1, 2, 3]
        scores = _scores(((0, 2), 0.9), ((1, 2), 0.8), ((2, 3), 0.7))
        cfg = PlanConfig(anchor_id=2)
        plan = build_placement_plan(ids, scores, cfg)
        assert plan.steps[0].fragment_id == 2

    def test_invalid_anchor_falls_back_to_first(self):
        ids = [0, 1, 2]
        scores = _scores(((0, 1), 0.8))
        cfg = PlanConfig(anchor_id=99)
        plan = build_placement_plan(ids, scores, cfg)
        assert plan.steps[0].fragment_id == 0

    def test_min_score_filters(self):
        ids = [0, 1, 2]
        scores = _scores(((0, 1), 0.9), ((0, 2), 0.1))
        cfg = PlanConfig(min_score=0.5)
        plan = build_placement_plan(ids, scores, cfg)
        # Fragment 2 with score 0.1 may be skipped or placed with 0 score
        assert plan.steps[0].is_anchor

    def test_empty_fragments_returns_empty(self):
        plan = build_placement_plan([], {})
        assert plan.n_fragments == 0
        assert plan.n_placed == 0
        assert plan.steps == []

    def test_single_fragment(self):
        plan = build_placement_plan([5], {})
        assert plan.n_placed == 1
        assert plan.steps[0].fragment_id == 5
        assert plan.steps[0].is_anchor

    def test_no_scores_all_placed(self):
        ids = [0, 1, 2]
        plan = build_placement_plan(ids, {})
        assert plan.n_placed >= 1

    def test_all_fragments_placed_with_full_scores(self):
        ids = list(range(5))
        scores = {(i, j): 0.8 for i in range(5) for j in range(i + 1, 5)}
        plan = build_placement_plan(ids, scores)
        assert plan.n_placed == 5

    def test_step_indices_sequential(self):
        plan = _simple_plan()
        for i, s in enumerate(plan.steps):
            assert s.step == i


# ─── TestReorderPlan ──────────────────────────────────────────────────────────

class TestReorderPlan:
    def test_priority_first(self):
        plan = _simple_plan()
        reordered = reorder_plan(plan, [3, 2])
        assert reordered.steps[0].fragment_id == 3
        assert reordered.steps[1].fragment_id == 2

    def test_unknown_priority_ids_ignored(self):
        plan = _simple_plan()
        reordered = reorder_plan(plan, [99])
        assert len(reordered.steps) == len(plan.steps)

    def test_preserves_all_fragments(self):
        plan = _simple_plan()
        reordered = reorder_plan(plan, [2, 1])
        ids_orig = set(plan.placement_order)
        ids_new = set(reordered.placement_order)
        assert ids_orig == ids_new

    def test_step_indices_reindexed(self):
        plan = _simple_plan()
        reordered = reorder_plan(plan, [2, 0])
        for i, s in enumerate(reordered.steps):
            assert s.step == i

    def test_empty_priority(self):
        plan = _simple_plan()
        reordered = reorder_plan(plan, [])
        assert reordered.placement_order == plan.placement_order


# ─── TestFilterPlan ───────────────────────────────────────────────────────────

class TestFilterPlan:
    def test_basic_filter(self):
        ids = [0, 1, 2, 3]
        scores = _scores(
            ((0, 1), 0.9), ((0, 2), 0.3), ((0, 3), 0.1),
        )
        plan = build_placement_plan(ids, scores)
        filtered = filter_plan(plan, min_score=0.5)
        # Anchor always kept
        assert filtered.steps[0].is_anchor

    def test_anchor_always_kept(self):
        plan = _simple_plan()
        filtered = filter_plan(plan, min_score=1.0)
        assert len(filtered.steps) >= 1
        assert filtered.steps[0].is_anchor

    def test_no_filter_keeps_all(self):
        plan = _simple_plan()
        filtered = filter_plan(plan, min_score=0.0)
        assert len(filtered.steps) == len(plan.steps)

    def test_step_indices_reindexed(self):
        plan = _simple_plan()
        filtered = filter_plan(plan, min_score=0.7)
        for i, s in enumerate(filtered.steps):
            assert s.step == i

    def test_invalid_min_score_neg(self):
        plan = _simple_plan()
        with pytest.raises(ValueError):
            filter_plan(plan, min_score=-0.1)


# ─── TestExportPlan ───────────────────────────────────────────────────────────

class TestExportPlan:
    def test_basic_keys(self):
        plan = _simple_plan()
        exported = export_plan(plan)
        assert len(exported) == len(plan.steps)
        for rec in exported:
            assert "step" in rec
            assert "fragment_id" in rec
            assert "score" in rec
            assert "anchored_by" in rec

    def test_empty_plan(self):
        plan = PlacementPlan(steps=[], n_fragments=0, n_placed=0, strategy="greedy")
        assert export_plan(plan) == []

    def test_step_order_preserved(self):
        plan = _simple_plan()
        exported = export_plan(plan)
        steps = [r["step"] for r in exported]
        assert steps == list(range(len(steps)))

    def test_anchored_by_is_list(self):
        plan = _simple_plan()
        exported = export_plan(plan)
        for rec in exported:
            assert isinstance(rec["anchored_by"], list)


# ─── TestBatchBuildPlans ──────────────────────────────────────────────────────

class TestBatchBuildPlans:
    def test_basic(self):
        id_lists = [[0, 1, 2], [3, 4, 5]]
        score_dicts = [
            _scores(((0, 1), 0.8), ((1, 2), 0.7)),
            _scores(((3, 4), 0.9), ((4, 5), 0.6)),
        ]
        plans = batch_build_plans(id_lists, score_dicts)
        assert len(plans) == 2
        for plan in plans:
            assert plan.n_placed >= 1

    def test_mismatch_raises(self):
        with pytest.raises(ValueError):
            batch_build_plans([[0, 1]], [{}, {}])

    def test_empty_lists(self):
        assert batch_build_plans([], []) == []

    def test_custom_config(self):
        id_lists = [[0, 1, 2], [3, 4]]
        score_dicts = [
            _scores(((0, 1), 0.9), ((1, 2), 0.8)),
            _scores(((3, 4), 0.7)),
        ]
        cfg = PlanConfig(strategy="bfs")
        plans = batch_build_plans(id_lists, score_dicts, cfg)
        for plan in plans:
            assert plan.strategy == "bfs"
