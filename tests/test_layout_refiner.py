"""Тесты для puzzle_reconstruction.assembly.layout_refiner."""
import pytest
from puzzle_reconstruction.assembly.layout_refiner import (
    RefineConfig,
    FragmentPosition,
    RefineStep,
    RefineResult,
    compute_layout_score,
    refine_layout,
    apply_offset,
    compare_layouts,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pos(fid: int, x: float = 0.0, y: float = 0.0,
         rot: float = 0.0) -> FragmentPosition:
    return FragmentPosition(fragment_id=fid, x=x, y=y, rotation=rot)


def _simple_layout():
    """Три фрагмента в ряд с шагом 1.0."""
    return {
        0: _pos(0, 0.0, 0.0),
        1: _pos(1, 1.0, 0.0),
        2: _pos(2, 2.0, 0.0),
    }


def _simple_adj():
    return {(0, 1): 0.9, (1, 2): 0.8}


def _refine(positions=None, adjacency=None, cfg=None):
    pos = positions or _simple_layout()
    adj = adjacency or _simple_adj()
    return refine_layout(pos, adj, cfg)


# ─── TestRefineConfig ─────────────────────────────────────────────────────────

class TestRefineConfig:
    def test_defaults(self):
        cfg = RefineConfig()
        assert cfg.max_iter == 20
        assert cfg.step_size == pytest.approx(1.0)
        assert cfg.convergence_eps == pytest.approx(0.01)
        assert cfg.frozen_ids == []

    def test_valid_custom(self):
        cfg = RefineConfig(max_iter=5, step_size=0.5, convergence_eps=0.0,
                           frozen_ids=[0, 1])
        assert cfg.max_iter == 5
        assert cfg.frozen_ids == [0, 1]

    def test_max_iter_one_ok(self):
        cfg = RefineConfig(max_iter=1)
        assert cfg.max_iter == 1

    def test_max_iter_zero_raises(self):
        with pytest.raises(ValueError):
            RefineConfig(max_iter=0)

    def test_step_size_pos_ok(self):
        cfg = RefineConfig(step_size=0.1)
        assert cfg.step_size == pytest.approx(0.1)

    def test_step_size_zero_raises(self):
        with pytest.raises(ValueError):
            RefineConfig(step_size=0.0)

    def test_step_size_neg_raises(self):
        with pytest.raises(ValueError):
            RefineConfig(step_size=-1.0)

    def test_convergence_eps_zero_ok(self):
        cfg = RefineConfig(convergence_eps=0.0)
        assert cfg.convergence_eps == 0.0

    def test_convergence_eps_neg_raises(self):
        with pytest.raises(ValueError):
            RefineConfig(convergence_eps=-0.001)


# ─── TestFragmentPosition ─────────────────────────────────────────────────────

class TestFragmentPosition:
    def test_defaults(self):
        p = FragmentPosition(fragment_id=5)
        assert p.x == pytest.approx(0.0)
        assert p.y == pytest.approx(0.0)
        assert p.rotation == pytest.approx(0.0)

    def test_position_tuple(self):
        p = _pos(0, 3.0, 4.0)
        assert p.position == (pytest.approx(3.0), pytest.approx(4.0))

    def test_distance_to_same(self):
        p = _pos(0, 1.0, 1.0)
        assert p.distance_to(p) == pytest.approx(0.0)

    def test_distance_to_3_4(self):
        a = _pos(0, 0.0, 0.0)
        b = _pos(1, 3.0, 4.0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_distance_symmetric(self):
        a = _pos(0, 1.0, 2.0)
        b = _pos(1, 4.0, 6.0)
        assert a.distance_to(b) == pytest.approx(b.distance_to(a))

    def test_fragment_id_stored(self):
        p = _pos(7, 1.0, 2.0)
        assert p.fragment_id == 7


# ─── TestRefineStep ───────────────────────────────────────────────────────────

class TestRefineStep:
    def test_basic(self):
        s = RefineStep(iteration=0, total_shift=1.5, score_delta=0.2, n_moved=2)
        assert s.iteration == 0
        assert s.total_shift == pytest.approx(1.5)
        assert s.n_moved == 2

    def test_improved_true(self):
        s = RefineStep(iteration=0, total_shift=0.5, score_delta=0.1, n_moved=1)
        assert s.improved is True

    def test_improved_false(self):
        s = RefineStep(iteration=0, total_shift=0.5, score_delta=-0.1, n_moved=1)
        assert s.improved is False

    def test_improved_zero(self):
        s = RefineStep(iteration=0, total_shift=0.0, score_delta=0.0, n_moved=0)
        assert s.improved is False

    def test_invalid_iteration_neg(self):
        with pytest.raises(ValueError):
            RefineStep(iteration=-1, total_shift=0.0, score_delta=0.0, n_moved=0)

    def test_invalid_total_shift_neg(self):
        with pytest.raises(ValueError):
            RefineStep(iteration=0, total_shift=-0.1, score_delta=0.0, n_moved=0)


# ─── TestRefineResult ─────────────────────────────────────────────────────────

class TestRefineResult:
    def _make(self, n_steps=3, converged=True):
        steps = [
            RefineStep(iteration=i, total_shift=float(i + 1) * 0.5,
                       score_delta=0.1 if i < n_steps - 1 else -0.05,
                       n_moved=2)
            for i in range(n_steps)
        ]
        positions = {0: _pos(0, 1.0, 0.0), 1: _pos(1, 2.0, 0.0)}
        return RefineResult(positions=positions, history=steps,
                            n_iter=n_steps, converged=converged)

    def test_total_shift(self):
        r = self._make(3)
        expected = 0.5 + 1.0 + 1.5
        assert r.total_shift == pytest.approx(expected)

    def test_total_shift_empty_history(self):
        r = RefineResult(positions={}, history=[], n_iter=0, converged=True)
        assert r.total_shift == pytest.approx(0.0)

    def test_improved_iters(self):
        r = self._make(3)
        # First 2 have positive delta, last has negative
        assert r.improved_iters == 2

    def test_get_position_found(self):
        r = self._make()
        p = r.get_position(0)
        assert p is not None
        assert p.fragment_id == 0

    def test_get_position_not_found(self):
        r = self._make()
        assert r.get_position(99) is None

    def test_converged_flag(self):
        r = self._make(converged=True)
        assert r.converged is True

    def test_not_converged_flag(self):
        r = self._make(converged=False)
        assert r.converged is False

    def test_invalid_n_iter_neg(self):
        with pytest.raises(ValueError):
            RefineResult(positions={}, history=[], n_iter=-1, converged=True)


# ─── TestComputeLayoutScore ───────────────────────────────────────────────────

class TestComputeLayoutScore:
    def test_at_target_gap_high_score(self):
        pos = {0: _pos(0, 0.0, 0.0), 1: _pos(1, 1.0, 0.0)}
        adj = {(0, 1): 1.0}
        score = compute_layout_score(pos, adj, target_gap=1.0)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_far_from_target_low_score(self):
        pos = {0: _pos(0, 0.0, 0.0), 1: _pos(1, 100.0, 0.0)}
        adj = {(0, 1): 1.0}
        score = compute_layout_score(pos, adj, target_gap=1.0)
        assert score < 0.01

    def test_missing_fragment_skipped(self):
        pos = {0: _pos(0, 0.0, 0.0)}
        adj = {(0, 1): 0.9}  # fragment 1 missing
        score = compute_layout_score(pos, adj, target_gap=1.0)
        assert score == pytest.approx(0.0)

    def test_empty_adjacency(self):
        score = compute_layout_score(_simple_layout(), {})
        assert score == pytest.approx(0.0)

    def test_score_nonneg(self):
        score = compute_layout_score(_simple_layout(), _simple_adj())
        assert score >= 0.0

    def test_higher_adj_score_higher_layout_score(self):
        pos = {0: _pos(0, 0.0, 0.0), 1: _pos(1, 1.0, 0.0)}
        adj_lo = {(0, 1): 0.3}
        adj_hi = {(0, 1): 0.9}
        assert (compute_layout_score(pos, adj_hi) >
                compute_layout_score(pos, adj_lo))


# ─── TestRefineLayout ─────────────────────────────────────────────────────────

class TestRefineLayout:
    def test_returns_refine_result(self):
        r = _refine()
        assert isinstance(r, RefineResult)

    def test_all_fragments_in_result(self):
        r = _refine()
        assert set(r.positions.keys()) == {0, 1, 2}

    def test_n_iter_positive(self):
        r = _refine()
        assert r.n_iter >= 1

    def test_history_length_equals_n_iter(self):
        r = _refine()
        assert len(r.history) == r.n_iter

    def test_frozen_not_moved(self):
        initial = _simple_layout()
        cfg = RefineConfig(max_iter=5, frozen_ids=[0])
        r = refine_layout(initial, _simple_adj(), cfg)
        assert r.positions[0].x == pytest.approx(0.0)
        assert r.positions[0].y == pytest.approx(0.0)

    def test_max_iter_respected(self):
        cfg = RefineConfig(max_iter=3, convergence_eps=0.0)
        r = refine_layout(_simple_layout(), _simple_adj(), cfg)
        assert r.n_iter <= 3

    def test_convergence_on_optimal(self):
        # Already at target gap=1.0
        pos = {0: _pos(0, 0.0, 0.0), 1: _pos(1, 1.0, 0.0)}
        adj = {(0, 1): 1.0}
        cfg = RefineConfig(max_iter=10, step_size=1.0, convergence_eps=0.5)
        r = refine_layout(pos, adj, cfg, target_gap=1.0)
        assert isinstance(r, RefineResult)

    def test_does_not_mutate_input(self):
        initial = _simple_layout()
        orig_x = {fid: pos.x for fid, pos in initial.items()}
        _refine(initial)
        for fid, pos in initial.items():
            assert pos.x == pytest.approx(orig_x[fid])

    def test_empty_positions(self):
        r = refine_layout({}, {})
        assert r.n_iter >= 0
        assert r.positions == {}

    def test_single_fragment_no_adj(self):
        pos = {0: _pos(0, 0.0, 0.0)}
        r = refine_layout(pos, {})
        assert 0 in r.positions

    def test_step_results_iterations_sequential(self):
        r = _refine()
        for i, step in enumerate(r.history):
            assert step.iteration == i


# ─── TestApplyOffset ──────────────────────────────────────────────────────────

class TestApplyOffset:
    def test_all_shifted(self):
        pos = _simple_layout()
        result = apply_offset(pos, 10.0, 5.0)
        for fid, p in result.items():
            assert p.x == pytest.approx(pos[fid].x + 10.0)
            assert p.y == pytest.approx(pos[fid].y + 5.0)

    def test_partial_shift(self):
        pos = _simple_layout()
        result = apply_offset(pos, 3.0, 0.0, fragment_ids=[0])
        assert result[0].x == pytest.approx(3.0)
        assert result[1].x == pytest.approx(pos[1].x)
        assert result[2].x == pytest.approx(pos[2].x)

    def test_does_not_mutate_input(self):
        pos = _simple_layout()
        orig = {fid: (p.x, p.y) for fid, p in pos.items()}
        apply_offset(pos, 5.0, 5.0)
        for fid, p in pos.items():
            assert p.x == pytest.approx(orig[fid][0])

    def test_zero_offset_identity(self):
        pos = _simple_layout()
        result = apply_offset(pos, 0.0, 0.0)
        for fid in pos:
            assert result[fid].x == pytest.approx(pos[fid].x)
            assert result[fid].y == pytest.approx(pos[fid].y)

    def test_rotation_preserved(self):
        pos = {0: _pos(0, 0.0, 0.0, rot=45.0)}
        result = apply_offset(pos, 1.0, 1.0)
        assert result[0].rotation == pytest.approx(45.0)

    def test_empty_fragment_ids_shifts_none(self):
        pos = _simple_layout()
        result = apply_offset(pos, 10.0, 10.0, fragment_ids=[])
        for fid in pos:
            assert result[fid].x == pytest.approx(pos[fid].x)


# ─── TestCompareLayouts ───────────────────────────────────────────────────────

class TestCompareLayouts:
    def test_identical_layouts(self):
        pos = _simple_layout()
        cmp = compare_layouts(pos, pos)
        assert cmp["mean_shift"] == pytest.approx(0.0)
        assert cmp["max_shift"] == pytest.approx(0.0)
        assert cmp["n_moved"] == 0

    def test_shifted_layout(self):
        before = _simple_layout()
        after = apply_offset(before, 3.0, 4.0)
        cmp = compare_layouts(before, after)
        assert cmp["mean_shift"] == pytest.approx(5.0)
        assert cmp["max_shift"] == pytest.approx(5.0)
        assert cmp["n_moved"] == 3

    def test_partial_shift(self):
        before = _simple_layout()
        after = apply_offset(before, 1.0, 0.0, fragment_ids=[0])
        cmp = compare_layouts(before, after)
        assert cmp["n_moved"] == 1

    def test_empty_positions(self):
        cmp = compare_layouts({}, {})
        assert cmp["mean_shift"] == pytest.approx(0.0)
        assert cmp["n_moved"] == 0

    def test_no_common_fragments(self):
        before = {0: _pos(0)}
        after = {1: _pos(1)}
        cmp = compare_layouts(before, after)
        assert cmp["mean_shift"] == pytest.approx(0.0)
        assert cmp["n_moved"] == 0

    def test_max_shift_gte_mean_shift(self):
        before = _simple_layout()
        after = {
            0: _pos(0, 10.0, 0.0),
            1: _pos(1, 1.0, 0.0),
            2: _pos(2, 2.0, 0.0),
        }
        cmp = compare_layouts(before, after)
        assert cmp["max_shift"] >= cmp["mean_shift"]
