"""Тесты для puzzle_reconstruction/assembly/fragment_scorer.py."""
import pytest
import numpy as np

from puzzle_reconstruction.assembly.assembly_state import (
    AssemblyState,
    create_state,
    place_fragment,
    add_adjacency,
)
from puzzle_reconstruction.assembly.cost_matrix import CostMatrix, build_from_scores
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_cm(n: int, fill: float = 0.5) -> CostMatrix:
    """Создать матрицу стоимостей N×N."""
    m = np.full((n, n), fill, dtype=np.float32)
    np.fill_diagonal(m, 0.0)
    return CostMatrix(matrix=m, n_fragments=n, method="test")


def _make_state(*placed_ids, n: int = 4) -> AssemblyState:
    """Создать состояние с размещёнными фрагментами."""
    s = create_state(n)
    for idx in placed_ids:
        s = place_fragment(s, idx, position=(float(idx), 0.0))
    return s


def _make_adj_state(n: int = 4, edges=()) -> AssemblyState:
    """Состояние со всеми фрагментами и набором рёбер смежности."""
    s = _make_state(*range(n), n=n)
    for i, j in edges:
        s = add_adjacency(s, i, j)
    return s


# ─── ScoreConfig ──────────────────────────────────────────────────────────────

class TestScoreConfig:
    def test_defaults(self):
        cfg = ScoreConfig()
        assert cfg.neighbor_weight == pytest.approx(0.7)
        assert cfg.coverage_weight == pytest.approx(0.3)
        assert cfg.min_neighbors == 1

    def test_negative_neighbor_weight_raises(self):
        with pytest.raises(ValueError, match="neighbor_weight"):
            ScoreConfig(neighbor_weight=-0.1)

    def test_negative_coverage_weight_raises(self):
        with pytest.raises(ValueError, match="coverage_weight"):
            ScoreConfig(coverage_weight=-0.1)

    def test_min_neighbors_zero_raises(self):
        with pytest.raises(ValueError, match="min_neighbors"):
            ScoreConfig(min_neighbors=0)

    def test_min_neighbors_negative_raises(self):
        with pytest.raises(ValueError, match="min_neighbors"):
            ScoreConfig(min_neighbors=-1)

    def test_total_weight(self):
        cfg = ScoreConfig(neighbor_weight=0.6, coverage_weight=0.4)
        assert cfg.total_weight == pytest.approx(1.0)

    def test_zero_weights_valid(self):
        cfg = ScoreConfig(neighbor_weight=0.0, coverage_weight=0.0)
        assert cfg.total_weight == pytest.approx(0.0)


# ─── FragmentScore ────────────────────────────────────────────────────────────

class TestFragmentScore:
    def test_creation(self):
        fs = FragmentScore(fragment_idx=0, local_score=0.5, n_neighbors=2)
        assert fs.fragment_idx == 0
        assert fs.local_score == pytest.approx(0.5)
        assert fs.n_neighbors == 2
        assert fs.is_reliable is False

    def test_negative_idx_raises(self):
        with pytest.raises(ValueError, match="fragment_idx"):
            FragmentScore(fragment_idx=-1, local_score=0.5, n_neighbors=0)

    def test_local_score_below_zero_raises(self):
        with pytest.raises(ValueError, match="local_score"):
            FragmentScore(fragment_idx=0, local_score=-0.1, n_neighbors=0)

    def test_local_score_above_1_raises(self):
        with pytest.raises(ValueError, match="local_score"):
            FragmentScore(fragment_idx=0, local_score=1.1, n_neighbors=0)

    def test_negative_n_neighbors_raises(self):
        with pytest.raises(ValueError, match="n_neighbors"):
            FragmentScore(fragment_idx=0, local_score=0.5, n_neighbors=-1)

    def test_is_reliable_true(self):
        fs = FragmentScore(fragment_idx=0, local_score=0.5, n_neighbors=3, is_reliable=True)
        assert fs.is_reliable is True

    def test_boundary_scores_valid(self):
        fs0 = FragmentScore(fragment_idx=0, local_score=0.0, n_neighbors=0)
        fs1 = FragmentScore(fragment_idx=0, local_score=1.0, n_neighbors=0)
        assert fs0.local_score == pytest.approx(0.0)
        assert fs1.local_score == pytest.approx(1.0)


# ─── AssemblyScore ────────────────────────────────────────────────────────────

class TestAssemblyScore:
    def test_creation(self):
        asm = AssemblyScore(global_score=0.8, coverage=0.5, mean_local=0.4)
        assert asm.global_score == pytest.approx(0.8)
        assert asm.coverage == pytest.approx(0.5)
        assert asm.mean_local == pytest.approx(0.4)
        assert asm.n_reliable == 0

    def test_global_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="global_score"):
            AssemblyScore(global_score=1.1, coverage=0.5, mean_local=0.5)

    def test_coverage_out_of_range_raises(self):
        with pytest.raises(ValueError, match="coverage"):
            AssemblyScore(global_score=0.5, coverage=-0.1, mean_local=0.5)

    def test_n_placed_empty(self):
        asm = AssemblyScore(global_score=0.5, coverage=0.5, mean_local=0.5)
        assert asm.n_placed == 0

    def test_n_placed_with_scores(self):
        fscores = {
            0: FragmentScore(0, 0.5, 1),
            1: FragmentScore(1, 0.3, 1),
        }
        asm = AssemblyScore(global_score=0.5, coverage=0.5, mean_local=0.5,
                            fragment_scores=fscores)
        assert asm.n_placed == 2

    def test_summary_returns_string(self):
        asm = AssemblyScore(global_score=0.6, coverage=0.5, mean_local=0.4)
        s = asm.summary()
        assert isinstance(s, str)
        assert "0.600" in s


# ─── score_fragment ───────────────────────────────────────────────────────────

class TestScoreFragment:
    def test_returns_fragment_score(self):
        s = _make_state(0, n=3)
        cm = _make_cm(3)
        result = score_fragment(s, 0, cm)
        assert isinstance(result, FragmentScore)

    def test_no_neighbors_neutral_score(self):
        s = _make_state(0, n=3)
        cm = _make_cm(3)
        result = score_fragment(s, 0, cm)
        assert result.local_score == pytest.approx(0.5)
        assert result.n_neighbors == 0

    def test_with_neighbor_uses_cost(self):
        s = _make_adj_state(n=4, edges=[(0, 1)])
        # cost matrix with known value at [0,1]
        m = np.full((4, 4), 0.0, dtype=np.float32)
        m[0, 1] = 0.8
        m[1, 0] = 0.8
        cm = CostMatrix(matrix=m, n_fragments=4, method="test")
        result = score_fragment(s, 0, cm)
        assert result.local_score == pytest.approx(0.8)
        assert result.n_neighbors == 1

    def test_not_placed_raises(self):
        s = _make_state(0, n=3)  # only 0 placed
        cm = _make_cm(3)
        with pytest.raises(ValueError):
            score_fragment(s, 1, cm)

    def test_mismatched_n_fragments_raises(self):
        s = _make_state(0, n=3)
        cm = _make_cm(4)  # mismatch
        with pytest.raises(ValueError, match="n_fragments"):
            score_fragment(s, 0, cm)

    def test_is_reliable_false_when_no_neighbors(self):
        s = _make_state(0, n=2)
        cm = _make_cm(2)
        result = score_fragment(s, 0, cm)
        assert result.is_reliable is False

    def test_is_reliable_true_with_enough_neighbors(self):
        cfg = ScoreConfig(min_neighbors=1)
        s = _make_adj_state(n=3, edges=[(0, 1)])
        cm = _make_cm(3, fill=0.4)
        result = score_fragment(s, 0, cm, cfg=cfg)
        assert result.is_reliable is True

    def test_local_score_clamped_to_0_1(self):
        s = _make_adj_state(n=3, edges=[(0, 1)])
        m = np.full((3, 3), 2.0, dtype=np.float32)  # out of [0,1]
        np.fill_diagonal(m, 0.0)
        cm = CostMatrix(matrix=m, n_fragments=3, method="test")
        result = score_fragment(s, 0, cm)
        assert 0.0 <= result.local_score <= 1.0

    def test_fragment_idx_stored(self):
        s = _make_state(2, n=4)
        cm = _make_cm(4)
        result = score_fragment(s, 2, cm)
        assert result.fragment_idx == 2


# ─── score_assembly ───────────────────────────────────────────────────────────

class TestScoreAssembly:
    def test_returns_assembly_score(self):
        s = _make_adj_state(n=3, edges=[(0, 1)])
        cm = _make_cm(3)
        result = score_assembly(s, cm)
        assert isinstance(result, AssemblyScore)

    def test_coverage_all_placed(self):
        s = _make_state(*range(4), n=4)
        cm = _make_cm(4)
        result = score_assembly(s, cm)
        assert result.coverage == pytest.approx(1.0)

    def test_coverage_partial(self):
        s = _make_state(0, 1, n=4)
        cm = _make_cm(4)
        result = score_assembly(s, cm)
        assert result.coverage == pytest.approx(0.5)

    def test_empty_state_coverage_zero(self):
        s = create_state(4)
        cm = _make_cm(4)
        result = score_assembly(s, cm)
        assert result.coverage == pytest.approx(0.0)

    def test_global_score_in_range(self):
        s = _make_adj_state(n=4, edges=[(0, 1), (1, 2)])
        cm = _make_cm(4)
        result = score_assembly(s, cm)
        assert 0.0 <= result.global_score <= 1.0

    def test_fragment_scores_keys(self):
        s = _make_state(0, 1, 2, n=4)
        cm = _make_cm(4)
        result = score_assembly(s, cm)
        assert set(result.fragment_scores.keys()) == {0, 1, 2}

    def test_n_reliable_count(self):
        cfg = ScoreConfig(min_neighbors=1)
        s = _make_adj_state(n=3, edges=[(0, 1)])
        cm = _make_cm(3)
        result = score_assembly(s, cm, cfg=cfg)
        assert result.n_reliable <= len(result.fragment_scores)

    def test_mismatched_cm_raises(self):
        s = _make_state(0, n=3)
        cm = _make_cm(5)
        with pytest.raises(ValueError, match="n_fragments"):
            score_assembly(s, cm)

    def test_zero_total_weight_global_score_zero(self):
        cfg = ScoreConfig(neighbor_weight=0.0, coverage_weight=0.0)
        s = _make_state(0, n=2)
        cm = _make_cm(2)
        result = score_assembly(s, cm, cfg=cfg)
        assert result.global_score == pytest.approx(0.0)


# ─── top_k_placed ─────────────────────────────────────────────────────────────

class TestTopKPlaced:
    def _make_assembly_score_with_fscores(self):
        fscores = {
            0: FragmentScore(0, 0.2, 1),  # best (lowest cost)
            1: FragmentScore(1, 0.5, 1),
            2: FragmentScore(2, 0.8, 1),  # worst (highest cost)
        }
        return AssemblyScore(global_score=0.5, coverage=1.0, mean_local=0.5,
                             fragment_scores=fscores)

    def test_returns_list(self):
        asm = self._make_assembly_score_with_fscores()
        assert isinstance(top_k_placed(asm, k=2), list)

    def test_k1_returns_one(self):
        asm = self._make_assembly_score_with_fscores()
        result = top_k_placed(asm, k=1)
        assert len(result) == 1

    def test_sorted_ascending(self):
        asm = self._make_assembly_score_with_fscores()
        result = top_k_placed(asm, k=3)
        scores = [r[1] for r in result]
        assert scores == sorted(scores)

    def test_k_exceeds_placed(self):
        asm = self._make_assembly_score_with_fscores()
        result = top_k_placed(asm, k=10)
        assert len(result) == 3  # only 3 placed

    def test_k_zero_raises(self):
        asm = self._make_assembly_score_with_fscores()
        with pytest.raises(ValueError):
            top_k_placed(asm, k=0)

    def test_tuple_format(self):
        asm = self._make_assembly_score_with_fscores()
        result = top_k_placed(asm, k=1)
        idx, score = result[0]
        assert isinstance(idx, int)
        assert isinstance(score, float)


# ─── bottom_k_placed ──────────────────────────────────────────────────────────

class TestBottomKPlaced:
    def _make_assembly_score_with_fscores(self):
        fscores = {
            0: FragmentScore(0, 0.2, 1),
            1: FragmentScore(1, 0.5, 1),
            2: FragmentScore(2, 0.8, 1),
        }
        return AssemblyScore(global_score=0.5, coverage=1.0, mean_local=0.5,
                             fragment_scores=fscores)

    def test_sorted_descending(self):
        asm = self._make_assembly_score_with_fscores()
        result = bottom_k_placed(asm, k=3)
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_k1_worst_fragment(self):
        asm = self._make_assembly_score_with_fscores()
        result = bottom_k_placed(asm, k=1)
        assert result[0][0] == 2  # idx=2 has highest local_score=0.8

    def test_k_zero_raises(self):
        asm = self._make_assembly_score_with_fscores()
        with pytest.raises(ValueError):
            bottom_k_placed(asm, k=0)

    def test_length_capped_at_n_placed(self):
        asm = self._make_assembly_score_with_fscores()
        result = bottom_k_placed(asm, k=100)
        assert len(result) == 3


# ─── batch_score ──────────────────────────────────────────────────────────────

class TestBatchScore:
    def test_returns_list(self):
        cm = _make_cm(3)
        states = [_make_state(0, n=3), _make_state(0, 1, n=3)]
        result = batch_score(states, cm)
        assert isinstance(result, list)

    def test_length_matches(self):
        cm = _make_cm(3)
        states = [_make_state(0, n=3), _make_state(1, n=3), _make_state(2, n=3)]
        result = batch_score(states, cm)
        assert len(result) == 3

    def test_all_assembly_scores(self):
        cm = _make_cm(3)
        states = [_make_state(0, n=3)]
        result = batch_score(states, cm)
        assert isinstance(result[0], AssemblyScore)

    def test_empty_states_empty_result(self):
        cm = _make_cm(3)
        result = batch_score([], cm)
        assert result == []
