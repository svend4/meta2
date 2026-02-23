"""Extra tests for puzzle_reconstruction/assembly/position_estimator.py"""
import pytest

from puzzle_reconstruction.assembly.position_estimator import (
    FragmentPosition,
    PositionConfig,
    PositionEstimate,
    batch_estimate_positions,
    estimate_grid_positions,
    generate_position_candidates,
    refine_positions,
    snap_to_grid,
)


# ─── TestPositionConfigExtra ──────────────────────────────────────────────────

class TestPositionConfigExtra:
    def test_large_canvas(self):
        cfg = PositionConfig(canvas_w=4096, canvas_h=4096)
        assert cfg.canvas_w == 4096
        assert cfg.canvas_h == 4096

    def test_large_grid(self):
        cfg = PositionConfig(grid_cols=16, grid_rows=16)
        assert cfg.grid_cols == 16
        assert cfg.grid_rows == 16

    def test_large_padding(self):
        cfg = PositionConfig(padding=100)
        assert cfg.padding == 100

    def test_large_snap_size(self):
        cfg = PositionConfig(snap_size=64)
        assert cfg.snap_size == 64

    def test_snap_grid_true(self):
        cfg = PositionConfig(snap_grid=True)
        assert cfg.snap_grid is True

    def test_snap_grid_false_by_default(self):
        cfg = PositionConfig()
        assert cfg.snap_grid is False

    def test_canvas_w_1_valid(self):
        cfg = PositionConfig(canvas_w=1, canvas_h=1)
        assert cfg.canvas_w == 1

    def test_grid_1x1(self):
        cfg = PositionConfig(grid_cols=1, grid_rows=1)
        assert cfg.grid_cols == 1
        assert cfg.grid_rows == 1

    def test_snap_size_1_valid(self):
        cfg = PositionConfig(snap_size=1)
        assert cfg.snap_size == 1


# ─── TestFragmentPositionExtra ────────────────────────────────────────────────

class TestFragmentPositionExtra:
    def test_large_coords(self):
        p = FragmentPosition(fragment_id=0, x=10000, y=20000)
        assert p.x == 10000
        assert p.y == 20000

    def test_confidence_min(self):
        p = FragmentPosition(fragment_id=0, x=0, y=0, confidence=0.0)
        assert p.confidence == pytest.approx(0.0)

    def test_confidence_max(self):
        p = FragmentPosition(fragment_id=0, x=0, y=0, confidence=1.0)
        assert p.confidence == pytest.approx(1.0)

    def test_method_refined(self):
        p = FragmentPosition(fragment_id=0, x=0, y=0, method="refined")
        assert p.method == "refined"

    def test_method_candidate(self):
        p = FragmentPosition(fragment_id=0, x=0, y=0, method="candidate")
        assert p.method == "candidate"

    def test_coords_large(self):
        p = FragmentPosition(fragment_id=5, x=100, y=200)
        assert p.coords == (100, 200)

    def test_large_fragment_id(self):
        p = FragmentPosition(fragment_id=999, x=0, y=0)
        assert p.fragment_id == 999

    def test_zero_coords(self):
        p = FragmentPosition(fragment_id=0, x=0, y=0)
        assert p.coords == (0, 0)


# ─── TestPositionEstimateExtra ────────────────────────────────────────────────

class TestPositionEstimateExtra:
    def _make(self, n=4):
        positions = [FragmentPosition(fragment_id=i, x=i * 10, y=0) for i in range(n)]
        return PositionEstimate(positions=positions, n_fragments=n,
                                canvas_w=512, canvas_h=512, mean_conf=1.0)

    def test_ten_fragments(self):
        e = self._make(10)
        assert e.n_fragments == 10
        assert len(e.by_id) == 10

    def test_by_id_all_keys(self):
        e = self._make(5)
        assert set(e.by_id.keys()) == {0, 1, 2, 3, 4}

    def test_mean_conf_stored(self):
        e = self._make(3)
        assert e.mean_conf == pytest.approx(1.0)

    def test_canvas_dimensions_stored(self):
        pos = [FragmentPosition(fragment_id=0, x=0, y=0)]
        e = PositionEstimate(positions=pos, n_fragments=1,
                             canvas_w=800, canvas_h=600, mean_conf=1.0)
        assert e.canvas_w == 800
        assert e.canvas_h == 600

    def test_by_id_lookup(self):
        e = self._make(3)
        assert e.by_id[2].x == 20


# ─── TestSnapToGridExtra ──────────────────────────────────────────────────────

class TestSnapToGridExtra:
    def test_snap_size_16(self):
        sx, sy = snap_to_grid(17, 33, 16)
        assert sx % 16 == 0
        assert sy % 16 == 0

    def test_snap_size_32(self):
        sx, sy = snap_to_grid(45, 65, 32)
        assert sx % 32 == 0
        assert sy % 32 == 0

    def test_exact_multiple_unchanged(self):
        sx, sy = snap_to_grid(64, 128, 32)
        assert sx == 64
        assert sy == 128

    def test_result_nonnegative(self):
        sx, sy = snap_to_grid(1, 1, 16)
        assert sx >= 0
        assert sy >= 0

    def test_large_input(self):
        sx, sy = snap_to_grid(1000, 2000, 8)
        assert sx % 8 == 0
        assert sy % 8 == 0

    def test_snap_size_2(self):
        sx, sy = snap_to_grid(3, 5, 2)
        assert sx % 2 == 0
        assert sy % 2 == 0


# ─── TestEstimateGridPositionsExtra ───────────────────────────────────────────

class TestEstimateGridPositionsExtra:
    def test_single_fragment(self):
        r = estimate_grid_positions([42], 32, 32)
        assert r.n_fragments == 1
        assert r.positions[0].fragment_id == 42

    def test_ten_fragments(self):
        r = estimate_grid_positions(list(range(10)), 32, 32)
        assert r.n_fragments == 10

    def test_large_fragment_size(self):
        cfg = PositionConfig(canvas_w=2000, canvas_h=2000)
        r = estimate_grid_positions(list(range(4)), 200, 200, cfg)
        assert r.n_fragments == 4

    def test_y_increases_with_rows(self):
        cfg = PositionConfig(canvas_w=512, canvas_h=512, grid_cols=1)
        r = estimate_grid_positions([0, 1, 2], 32, 32, cfg)
        # all in one column, y should increase
        assert r.positions[1].y >= r.positions[0].y

    def test_x_increases_with_cols(self):
        cfg = PositionConfig(canvas_w=512, canvas_h=512, grid_rows=1)
        r = estimate_grid_positions([0, 1, 2], 32, 32, cfg)
        # all in one row, x should increase
        assert r.positions[1].x >= r.positions[0].x

    def test_snap_aligns_positions(self):
        cfg = PositionConfig(snap_grid=True, snap_size=8)
        r = estimate_grid_positions([0, 1, 2, 3], 20, 20, cfg)
        for p in r.positions:
            assert p.x % 8 == 0 or p.x == 0

    def test_non_square_fragments(self):
        r = estimate_grid_positions([0, 1], 32, 64)
        assert r.n_fragments == 2

    def test_positions_all_nonneg(self):
        r = estimate_grid_positions(list(range(6)), 32, 32)
        for p in r.positions:
            assert p.x >= 0
            assert p.y >= 0


# ─── TestRefinePositionsExtra ─────────────────────────────────────────────────

class TestRefinePositionsExtra:
    def _est(self, n=4):
        positions = [FragmentPosition(fragment_id=i, x=i * 10, y=0) for i in range(n)]
        return PositionEstimate(positions=positions, n_fragments=n,
                                canvas_w=512, canvas_h=512, mean_conf=1.0)

    def test_zero_offsets_unchanged_coords(self):
        e = self._est(3)
        r = refine_positions(e, [(0, 0)] * 3)
        for orig, new in zip(e.positions, r.positions):
            assert new.x == orig.x
            assert new.y == orig.y

    def test_positive_offsets_increase_coords(self):
        e = self._est(2)
        r = refine_positions(e, [(5, 10), (3, 7)])
        assert r.positions[0].x == e.positions[0].x + 5
        assert r.positions[0].y == e.positions[0].y + 10

    def test_large_negative_offset_clips_to_zero(self):
        e = self._est(1)
        r = refine_positions(e, [(-9999, -9999)])
        assert r.positions[0].x == 0
        assert r.positions[0].y == 0

    def test_canvas_dimensions_preserved(self):
        e = self._est(2)
        r = refine_positions(e, [(0, 0), (0, 0)])
        assert r.canvas_w == e.canvas_w
        assert r.canvas_h == e.canvas_h

    def test_method_all_refined(self):
        e = self._est(4)
        r = refine_positions(e, [(1, 1)] * 4)
        assert all(p.method == "refined" for p in r.positions)

    def test_mean_conf_with_all_same(self):
        e = self._est(3)
        r = refine_positions(e, [(0, 0)] * 3, confidences=[0.6, 0.6, 0.6])
        assert r.mean_conf == pytest.approx(0.6)

    def test_returns_new_position_estimate(self):
        e = self._est(2)
        r = refine_positions(e, [(0, 0), (0, 0)])
        assert r is not e


# ─── TestGeneratePositionCandidatesExtra ──────────────────────────────────────

class TestGeneratePositionCandidatesExtra:
    def test_radius_2_step_1_count(self):
        result = generate_position_candidates(10, 10, 2, step=1)
        # -2,-1,0,1,2 × -2,-1,0,1,2 clipped to >=0 → 5×5 = 25
        assert len(result) > 0

    def test_all_coords_are_tuples(self):
        result = generate_position_candidates(5, 5, 2)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in result)

    def test_original_coords_included(self):
        result = generate_position_candidates(10, 10, 2)
        assert (10, 10) in result

    def test_radius_0_step_1(self):
        result = generate_position_candidates(5, 5, 0, step=1)
        assert result == [(5, 5)]

    def test_step_larger_than_radius(self):
        result = generate_position_candidates(10, 10, 4, step=8)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_large_x_y(self):
        result = generate_position_candidates(500, 500, 3)
        assert all(x >= 0 and y >= 0 for x, y in result)

    def test_no_negative_coords(self):
        result = generate_position_candidates(0, 0, 5)
        assert all(x >= 0 and y >= 0 for x, y in result)


# ─── TestBatchEstimatePositionsExtra ──────────────────────────────────────────

class TestBatchEstimatePositionsExtra:
    def test_five_groups(self):
        groups = [list(range(i * 3, (i + 1) * 3)) for i in range(5)]
        r = batch_estimate_positions(groups, 32, 32)
        assert len(r) == 5

    def test_each_has_correct_n_fragments(self):
        groups = [[0, 1], [2, 3, 4], [5]]
        r = batch_estimate_positions(groups, 32, 32)
        assert r[0].n_fragments == 2
        assert r[1].n_fragments == 3
        assert r[2].n_fragments == 1

    def test_single_group_single_fragment(self):
        r = batch_estimate_positions([[99]], 32, 32)
        assert len(r) == 1
        assert r[0].n_fragments == 1

    def test_custom_config_applied_all(self):
        cfg = PositionConfig(canvas_w=200, canvas_h=100)
        groups = [[0, 1], [2, 3]]
        r = batch_estimate_positions(groups, 20, 20, cfg)
        assert all(e.canvas_w == 200 for e in r)
        assert all(e.canvas_h == 100 for e in r)

    def test_returns_position_estimate_instances(self):
        r = batch_estimate_positions([[0, 1], [2]], 32, 32)
        assert all(isinstance(e, PositionEstimate) for e in r)

    def test_empty_group_included(self):
        r = batch_estimate_positions([[], [0, 1]], 32, 32)
        assert r[0].n_fragments == 0
        assert r[1].n_fragments == 2
