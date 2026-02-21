"""Тесты для puzzle_reconstruction.assembly.position_estimator."""
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


# ─── TestPositionConfig ───────────────────────────────────────────────────────

class TestPositionConfig:
    def test_defaults(self):
        cfg = PositionConfig()
        assert cfg.canvas_w == 512
        assert cfg.canvas_h == 512
        assert cfg.grid_cols == 4
        assert cfg.grid_rows == 4
        assert cfg.padding == 0
        assert cfg.snap_grid is False
        assert cfg.snap_size == 8

    def test_canvas_w_lt_1_raises(self):
        with pytest.raises(ValueError):
            PositionConfig(canvas_w=0)

    def test_canvas_h_lt_1_raises(self):
        with pytest.raises(ValueError):
            PositionConfig(canvas_h=0)

    def test_grid_cols_lt_1_raises(self):
        with pytest.raises(ValueError):
            PositionConfig(grid_cols=0)

    def test_grid_rows_lt_1_raises(self):
        with pytest.raises(ValueError):
            PositionConfig(grid_rows=0)

    def test_negative_padding_raises(self):
        with pytest.raises(ValueError):
            PositionConfig(padding=-1)

    def test_snap_size_lt_1_raises(self):
        with pytest.raises(ValueError):
            PositionConfig(snap_size=0)

    def test_custom_valid(self):
        cfg = PositionConfig(canvas_w=200, canvas_h=300, grid_cols=2,
                             grid_rows=3, padding=5, snap_grid=True, snap_size=16)
        assert cfg.grid_cols == 2
        assert cfg.padding == 5


# ─── TestFragmentPosition ─────────────────────────────────────────────────────

class TestFragmentPosition:
    def test_basic_construction(self):
        p = FragmentPosition(fragment_id=0, x=10, y=20)
        assert p.fragment_id == 0
        assert p.x == 10
        assert p.y == 20

    def test_defaults(self):
        p = FragmentPosition(fragment_id=0, x=0, y=0)
        assert p.confidence == 1.0
        assert p.method == "grid"

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            FragmentPosition(fragment_id=-1, x=0, y=0)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            FragmentPosition(fragment_id=0, x=-1, y=0)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            FragmentPosition(fragment_id=0, x=0, y=-1)

    def test_confidence_below_0_raises(self):
        with pytest.raises(ValueError):
            FragmentPosition(fragment_id=0, x=0, y=0, confidence=-0.1)

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValueError):
            FragmentPosition(fragment_id=0, x=0, y=0, confidence=1.1)

    def test_empty_method_raises(self):
        with pytest.raises(ValueError):
            FragmentPosition(fragment_id=0, x=0, y=0, method="")

    def test_coords_property(self):
        p = FragmentPosition(fragment_id=0, x=5, y=7)
        assert p.coords == (5, 7)


# ─── TestPositionEstimate ─────────────────────────────────────────────────────

class TestPositionEstimate:
    def _make(self, n=3):
        positions = [FragmentPosition(fragment_id=i, x=i * 10, y=0)
                     for i in range(n)]
        return PositionEstimate(positions=positions, n_fragments=n,
                                canvas_w=512, canvas_h=512, mean_conf=1.0)

    def test_basic_construction(self):
        e = self._make()
        assert e.n_fragments == 3

    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            PositionEstimate(positions=[], n_fragments=-1,
                             canvas_w=512, canvas_h=512, mean_conf=0.0)

    def test_canvas_w_lt_1_raises(self):
        with pytest.raises(ValueError):
            PositionEstimate(positions=[], n_fragments=0,
                             canvas_w=0, canvas_h=512, mean_conf=0.0)

    def test_canvas_h_lt_1_raises(self):
        with pytest.raises(ValueError):
            PositionEstimate(positions=[], n_fragments=0,
                             canvas_w=512, canvas_h=0, mean_conf=0.0)

    def test_negative_mean_conf_raises(self):
        with pytest.raises(ValueError):
            PositionEstimate(positions=[], n_fragments=0,
                             canvas_w=512, canvas_h=512, mean_conf=-0.1)

    def test_by_id_property(self):
        e = self._make(n=3)
        d = e.by_id
        assert 0 in d and 1 in d and 2 in d
        assert d[0].fragment_id == 0

    def test_by_id_empty(self):
        e = PositionEstimate(positions=[], n_fragments=0,
                             canvas_w=512, canvas_h=512, mean_conf=0.0)
        assert e.by_id == {}


# ─── TestSnapToGrid ───────────────────────────────────────────────────────────

class TestSnapToGrid:
    def test_already_aligned(self):
        sx, sy = snap_to_grid(16, 32, 8)
        assert sx == 16
        assert sy == 32

    def test_rounds_to_nearest(self):
        sx, sy = snap_to_grid(5, 5, 8)
        assert sx == 8
        assert sy == 8

    def test_rounds_down(self):
        sx, sy = snap_to_grid(3, 3, 8)
        assert sx == 0
        assert sy == 0

    def test_never_negative(self):
        sx, sy = snap_to_grid(0, 0, 8)
        assert sx >= 0
        assert sy >= 0

    def test_invalid_snap_size_raises(self):
        with pytest.raises(ValueError):
            snap_to_grid(10, 10, 0)

    def test_snap_size_1(self):
        sx, sy = snap_to_grid(7, 13, 1)
        assert sx == 7
        assert sy == 13


# ─── TestEstimateGridPositions ────────────────────────────────────────────────

class TestEstimateGridPositions:
    def test_returns_position_estimate(self):
        r = estimate_grid_positions([0, 1, 2], 32, 32)
        assert isinstance(r, PositionEstimate)

    def test_n_fragments_correct(self):
        r = estimate_grid_positions([0, 1, 2, 3], 32, 32)
        assert r.n_fragments == 4

    def test_empty_ids_returns_empty(self):
        r = estimate_grid_positions([], 32, 32)
        assert r.n_fragments == 0

    def test_frag_w_lt_1_raises(self):
        with pytest.raises(ValueError):
            estimate_grid_positions([0], 0, 32)

    def test_frag_h_lt_1_raises(self):
        with pytest.raises(ValueError):
            estimate_grid_positions([0], 32, 0)

    def test_positions_in_canvas(self):
        cfg = PositionConfig(canvas_w=200, canvas_h=200)
        r = estimate_grid_positions(list(range(8)), 40, 40, cfg)
        for p in r.positions:
            assert p.x >= 0
            assert p.y >= 0
            assert p.x < 200
            assert p.y < 200

    def test_ids_preserved(self):
        ids = [10, 20, 30]
        r = estimate_grid_positions(ids, 32, 32)
        result_ids = [p.fragment_id for p in r.positions]
        assert result_ids == ids

    def test_mean_conf_is_one(self):
        r = estimate_grid_positions([0, 1], 32, 32)
        assert abs(r.mean_conf - 1.0) < 1e-9

    def test_wraps_to_next_row(self):
        cfg = PositionConfig(canvas_w=512, canvas_h=512, grid_cols=2)
        r = estimate_grid_positions([0, 1, 2, 3], 32, 32, cfg)
        # fragments 0 and 2 should have same x (col=0)
        assert r.positions[0].x == r.positions[2].x
        # fragments 1 and 3 have x > 0 (col=1)
        assert r.positions[1].x > 0 and r.positions[3].x > 0

    def test_snap_grid_option(self):
        cfg = PositionConfig(canvas_w=512, canvas_h=512,
                             snap_grid=True, snap_size=16)
        r = estimate_grid_positions([0, 1, 2], 20, 20, cfg)
        for p in r.positions:
            assert p.x % 16 == 0 or p.x == 0
            assert p.y % 16 == 0 or p.y == 0

    def test_canvas_dimensions_stored(self):
        cfg = PositionConfig(canvas_w=100, canvas_h=200)
        r = estimate_grid_positions([0], 10, 10, cfg)
        assert r.canvas_w == 100
        assert r.canvas_h == 200


# ─── TestRefinePositions ──────────────────────────────────────────────────────

class TestRefinePositions:
    def _estimate(self, n=3):
        positions = [FragmentPosition(fragment_id=i, x=i * 10, y=0)
                     for i in range(n)]
        return PositionEstimate(positions=positions, n_fragments=n,
                                canvas_w=512, canvas_h=512, mean_conf=1.0)

    def test_returns_position_estimate(self):
        e = self._estimate(2)
        r = refine_positions(e, [(1, 2), (-1, 0)])
        assert isinstance(r, PositionEstimate)

    def test_offset_applied(self):
        e = self._estimate(1)
        r = refine_positions(e, [(5, 3)])
        assert r.positions[0].x == 5
        assert r.positions[0].y == 3

    def test_negative_result_clipped_to_zero(self):
        e = self._estimate(1)
        r = refine_positions(e, [(-100, -100)])
        assert r.positions[0].x == 0
        assert r.positions[0].y == 0

    def test_offset_length_mismatch_raises(self):
        e = self._estimate(2)
        with pytest.raises(ValueError):
            refine_positions(e, [(1, 0)])

    def test_confidences_applied(self):
        e = self._estimate(2)
        r = refine_positions(e, [(0, 0), (0, 0)], confidences=[0.3, 0.7])
        assert abs(r.positions[0].confidence - 0.3) < 1e-9
        assert abs(r.positions[1].confidence - 0.7) < 1e-9

    def test_confidence_length_mismatch_raises(self):
        e = self._estimate(2)
        with pytest.raises(ValueError):
            refine_positions(e, [(0, 0), (0, 0)], confidences=[0.5])

    def test_method_set_to_refined(self):
        e = self._estimate(1)
        r = refine_positions(e, [(0, 0)])
        assert r.positions[0].method == "refined"

    def test_mean_conf_updated(self):
        e = self._estimate(2)
        r = refine_positions(e, [(0, 0), (0, 0)], confidences=[0.4, 0.6])
        assert abs(r.mean_conf - 0.5) < 1e-9

    def test_n_fragments_preserved(self):
        e = self._estimate(3)
        r = refine_positions(e, [(0, 0)] * 3)
        assert r.n_fragments == 3


# ─── TestGeneratePositionCandidates ───────────────────────────────────────────

class TestGeneratePositionCandidates:
    def test_returns_list(self):
        result = generate_position_candidates(10, 10, 2)
        assert isinstance(result, list)

    def test_radius_zero_single_point(self):
        result = generate_position_candidates(10, 10, 0)
        assert len(result) == 1
        assert result[0] == (10, 10)

    def test_all_nonnegative(self):
        result = generate_position_candidates(2, 2, 5)
        for x, y in result:
            assert x >= 0 and y >= 0

    def test_radius_1_step_1(self):
        result = generate_position_candidates(5, 5, 1)
        # -1,0,+1 × -1,0,+1 = 9 candidates
        assert len(result) == 9

    def test_step_2_reduces_count(self):
        r1 = generate_position_candidates(10, 10, 4, step=1)
        r2 = generate_position_candidates(10, 10, 4, step=2)
        assert len(r2) < len(r1)

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError):
            generate_position_candidates(0, 0, -1)

    def test_step_lt_1_raises(self):
        with pytest.raises(ValueError):
            generate_position_candidates(0, 0, 2, step=0)

    def test_no_duplicates_basic(self):
        result = generate_position_candidates(10, 10, 2)
        assert len(result) == len(set(result))


# ─── TestBatchEstimatePositions ───────────────────────────────────────────────

class TestBatchEstimatePositions:
    def test_returns_list(self):
        r = batch_estimate_positions([[0, 1], [2, 3]], 32, 32)
        assert isinstance(r, list)

    def test_length_matches(self):
        r = batch_estimate_positions([[0], [1], [2, 3]], 32, 32)
        assert len(r) == 3

    def test_each_is_position_estimate(self):
        r = batch_estimate_positions([[0, 1]], 32, 32)
        assert all(isinstance(e, PositionEstimate) for e in r)

    def test_empty_list(self):
        r = batch_estimate_positions([], 32, 32)
        assert r == []

    def test_custom_config(self):
        cfg = PositionConfig(canvas_w=100, canvas_h=100)
        r = batch_estimate_positions([[0, 1]], 20, 20, cfg)
        assert r[0].canvas_w == 100
