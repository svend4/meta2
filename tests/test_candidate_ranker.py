"""Тесты для puzzle_reconstruction/matching/candidate_ranker.py."""
import numpy as np
import pytest

from puzzle_reconstruction.matching.candidate_ranker import (
    CandidatePair,
    score_pair,
    rank_pairs,
    filter_by_score,
    top_k,
    deduplicate_pairs,
    batch_rank,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _pair(idx1, idx2, score, **meta):
    return CandidatePair(idx1=idx1, idx2=idx2, score=float(score), meta=dict(meta))


def _pairs(*scores):
    """Создаёт список пар (i, i+1) с заданными оценками."""
    return [_pair(i, i + 1, s) for i, s in enumerate(scores)]


# ─── CandidatePair ────────────────────────────────────────────────────────────

class TestCandidatePair:
    def test_fields(self):
        p = _pair(2, 5, 0.75)
        assert p.idx1 == 2
        assert p.idx2 == 5
        assert p.score == pytest.approx(0.75)

    def test_score_is_float(self):
        p = _pair(0, 1, 0.5)
        assert isinstance(p.score, float)

    def test_meta_default_empty(self):
        p = _pair(0, 1, 0.5)
        assert isinstance(p.meta, dict)

    def test_meta_stored(self):
        p = _pair(0, 1, 0.8, method="phase", channel="color")
        assert p.meta["method"] == "phase"
        assert p.meta["channel"] == "color"

    def test_repr_contains_class(self):
        assert "CandidatePair" in repr(_pair(0, 1, 0.5))

    def test_repr_contains_score(self):
        r = repr(_pair(0, 1, 0.1234))
        assert "0.12" in r or "score" in r.lower()

    def test_repr_contains_indices(self):
        r = repr(_pair(3, 7, 0.5))
        assert "3" in r and "7" in r

    def test_lt_higher_score_first(self):
        p_low  = _pair(0, 1, 0.2)
        p_high = _pair(2, 3, 0.8)
        # При сортировке sorted() высокая оценка должна идти первой
        ordered = sorted([p_low, p_high])
        assert ordered[0].score > ordered[1].score

    def test_lt_equal_scores(self):
        p1 = _pair(0, 1, 0.5)
        p2 = _pair(2, 3, 0.5)
        # Не должно бросать исключений
        assert not (p1 < p2 and p2 < p1)

    def test_score_zero(self):
        p = _pair(0, 1, 0.0)
        assert p.score == pytest.approx(0.0)

    def test_score_one(self):
        p = _pair(0, 1, 1.0)
        assert p.score == pytest.approx(1.0)


# ─── score_pair ───────────────────────────────────────────────────────────────

class TestScorePair:
    def test_returns_candidate_pair(self):
        assert isinstance(score_pair(0, 1, 0.5), CandidatePair)

    def test_idx1_stored(self):
        assert score_pair(3, 7, 0.5).idx1 == 3

    def test_idx2_stored(self):
        assert score_pair(3, 7, 0.5).idx2 == 7

    def test_score_stored(self):
        assert score_pair(0, 1, 0.82).score == pytest.approx(0.82)

    def test_score_is_float(self):
        assert isinstance(score_pair(0, 1, 1).score, float)

    def test_meta_kwargs_stored(self):
        p = score_pair(0, 1, 0.6, side=2, channel="rgb")
        assert p.meta.get("side") == 2
        assert p.meta.get("channel") == "rgb"

    def test_meta_empty_by_default(self):
        p = score_pair(0, 1, 0.5)
        assert isinstance(p.meta, dict)

    def test_zero_score(self):
        assert score_pair(0, 1, 0.0).score == pytest.approx(0.0)


# ─── rank_pairs ───────────────────────────────────────────────────────────────

class TestRankPairs:
    def test_returns_list(self):
        assert isinstance(rank_pairs(_pairs(0.3, 0.7)), list)

    def test_same_length(self):
        p = _pairs(0.2, 0.8, 0.5)
        assert len(rank_pairs(p)) == 3

    def test_empty_returns_empty(self):
        assert rank_pairs([]) == []

    def test_sorted_desc(self):
        p = _pairs(0.2, 0.9, 0.5)
        r = rank_pairs(p)
        assert r[0].score == pytest.approx(0.9)
        assert r[1].score == pytest.approx(0.5)
        assert r[2].score == pytest.approx(0.2)

    def test_each_is_candidate_pair(self):
        for p in rank_pairs(_pairs(0.3, 0.7)):
            assert isinstance(p, CandidatePair)

    def test_does_not_modify_input(self):
        pairs = _pairs(0.1, 0.9, 0.5)
        original_order = [p.score for p in pairs]
        rank_pairs(pairs)
        assert [p.score for p in pairs] == original_order

    def test_all_same_score(self):
        p = _pairs(0.5, 0.5, 0.5)
        r = rank_pairs(p)
        assert len(r) == 3
        for pair in r:
            assert pair.score == pytest.approx(0.5)

    def test_single_pair(self):
        p = [_pair(0, 1, 0.7)]
        r = rank_pairs(p)
        assert len(r) == 1
        assert r[0].score == pytest.approx(0.7)


# ─── filter_by_score ──────────────────────────────────────────────────────────

class TestFilterByScore:
    def test_returns_list(self):
        assert isinstance(filter_by_score(_pairs(0.3, 0.7)), list)

    def test_empty_returns_empty(self):
        assert filter_by_score([]) == []

    def test_all_above_threshold(self):
        p = _pairs(0.8, 0.9, 1.0)
        r = filter_by_score(p, threshold=0.5)
        assert len(r) == 3

    def test_all_below_threshold(self):
        p = _pairs(0.1, 0.2, 0.3)
        r = filter_by_score(p, threshold=0.5)
        assert len(r) == 0

    def test_boundary_exclusive(self):
        p = [_pair(0, 1, 0.5)]
        # score > threshold (strict), 0.5 не превышает 0.5
        assert filter_by_score(p, threshold=0.5) == []

    def test_boundary_above(self):
        p = [_pair(0, 1, 0.51)]
        assert len(filter_by_score(p, threshold=0.5)) == 1

    def test_mixed_filtering(self):
        p = _pairs(0.3, 0.8, 0.6)
        r = filter_by_score(p, threshold=0.5)
        scores = [x.score for x in r]
        assert 0.8 in [pytest.approx(s) for s in scores]
        assert 0.6 in [pytest.approx(s) for s in scores]
        assert len(r) == 2

    def test_result_sorted_desc(self):
        p = _pairs(0.3, 0.9, 0.6)
        r = filter_by_score(p, threshold=0.4)
        assert r[0].score >= r[1].score

    def test_each_is_candidate_pair(self):
        p = _pairs(0.7, 0.8)
        for x in filter_by_score(p, threshold=0.5):
            assert isinstance(x, CandidatePair)

    def test_default_threshold_05(self):
        p = _pairs(0.3, 0.7)
        r = filter_by_score(p)
        assert len(r) == 1
        assert r[0].score == pytest.approx(0.7)


# ─── top_k ────────────────────────────────────────────────────────────────────

class TestTopK:
    def test_k_zero_empty(self):
        assert top_k(_pairs(0.5, 0.8), k=0) == []

    def test_k_greater_than_n_clips(self):
        p = _pairs(0.5, 0.8)
        r = top_k(p, k=100)
        assert len(r) == 2

    def test_k1_is_best(self):
        p = _pairs(0.3, 0.9, 0.6)
        r = top_k(p, k=1)
        assert r[0].score == pytest.approx(0.9)

    def test_k2_sorted_desc(self):
        p = _pairs(0.3, 0.9, 0.6)
        r = top_k(p, k=2)
        assert r[0].score == pytest.approx(0.9)
        assert r[1].score == pytest.approx(0.6)

    def test_empty_returns_empty(self):
        assert top_k([], k=5) == []

    def test_each_is_candidate_pair(self):
        for x in top_k(_pairs(0.4, 0.7), k=2):
            assert isinstance(x, CandidatePair)

    def test_no_deduplicate_allows_repeat_idx(self):
        # Без дедупликации: пары могут делить индексы
        pairs = [_pair(0, 1, 0.9), _pair(0, 2, 0.8)]
        r = top_k(pairs, k=2, deduplicate=False)
        assert len(r) == 2

    def test_deduplicate_no_repeat_idx(self):
        # Пара (0,1) занимает индексы 0 и 1;
        # Пара (0,2) конфликтует по idx=0 → исключается
        pairs = [_pair(0, 1, 0.9), _pair(0, 2, 0.8)]
        r = top_k(pairs, k=2, deduplicate=True)
        indices = set()
        for p in r:
            assert p.idx1 not in indices
            assert p.idx2 not in indices
            indices.add(p.idx1)
            indices.add(p.idx2)

    def test_deduplicate_picks_best(self):
        pairs = [_pair(1, 2, 0.9), _pair(1, 3, 0.5)]
        r = top_k(pairs, k=2, deduplicate=True)
        assert r[0].score == pytest.approx(0.9)
        assert len(r) == 1   # (1,3) конфликтует по idx=1

    def test_returns_list(self):
        assert isinstance(top_k(_pairs(0.5), k=1), list)


# ─── deduplicate_pairs ────────────────────────────────────────────────────────

class TestDeduplicatePairs:
    def test_returns_list(self):
        assert isinstance(deduplicate_pairs([]), list)

    def test_empty_returns_empty(self):
        assert deduplicate_pairs([]) == []

    def test_single_pair_included(self):
        r = deduplicate_pairs([_pair(0, 1, 0.7)])
        assert len(r) == 1
        assert r[0].idx1 == 0

    def test_no_shared_indices_all_included(self):
        pairs = [_pair(0, 1, 0.9), _pair(2, 3, 0.7)]
        r = deduplicate_pairs(pairs)
        assert len(r) == 2

    def test_conflict_keeps_best(self):
        pairs = [_pair(0, 1, 0.3), _pair(0, 2, 0.9)]
        r = deduplicate_pairs(pairs)
        assert len(r) == 1
        assert r[0].score == pytest.approx(0.9)

    def test_no_index_repeated_in_output(self):
        pairs = [_pair(0, 1, 0.8), _pair(1, 2, 0.7), _pair(2, 3, 0.6)]
        r = deduplicate_pairs(pairs)
        seen = set()
        for p in r:
            assert p.idx1 not in seen
            assert p.idx2 not in seen
            seen.add(p.idx1)
            seen.add(p.idx2)

    def test_greedy_order_by_score(self):
        # Лучшая пара должна быть в результате
        pairs = [_pair(5, 6, 0.99), _pair(5, 7, 0.01)]
        r = deduplicate_pairs(pairs)
        assert r[0].score == pytest.approx(0.99)

    def test_each_is_candidate_pair(self):
        pairs = [_pair(0, 1, 0.8), _pair(2, 3, 0.5)]
        for p in deduplicate_pairs(pairs):
            assert isinstance(p, CandidatePair)


# ─── batch_rank ───────────────────────────────────────────────────────────────

class TestBatchRank:
    def _mat(self, n=4, seed=7):
        rng = np.random.default_rng(seed)
        m = rng.uniform(0.1, 1.0, (n, n)).astype(np.float32)
        np.fill_diagonal(m, 0.0)
        return m

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            batch_rank(np.array([0.5, 0.8]))

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            batch_rank(np.ones((3, 4)))

    def test_returns_list(self):
        assert isinstance(batch_rank(self._mat()), list)

    def test_each_is_candidate_pair(self):
        for p in batch_rank(self._mat(3)):
            assert isinstance(p, CandidatePair)

    def test_sorted_desc(self):
        r = batch_rank(self._mat(4))
        for i in range(len(r) - 1):
            assert r[i].score >= r[i + 1].score

    def test_symmetric_upper_triangle(self):
        m = self._mat(4)
        r = batch_rank(m, symmetric=True)
        for p in r:
            assert p.idx1 < p.idx2

    def test_threshold_filters(self):
        m = np.full((3, 3), 0.3, dtype=np.float32)
        np.fill_diagonal(m, 0.0)
        r = batch_rank(m, threshold=0.5)
        assert r == []

    def test_score_values_correct(self):
        m = np.zeros((2, 2), dtype=np.float32)
        m[0, 1] = 0.75
        r = batch_rank(m, symmetric=True)
        assert len(r) == 1
        assert r[0].score == pytest.approx(0.75)

    def test_empty_2x2_diagonal(self):
        m = np.eye(2, dtype=np.float32)
        # Диагональ отфильтруется, off-diagonal = 0 ≤ threshold=0
        r = batch_rank(m, threshold=0.0)
        assert isinstance(r, list)

    def test_1x1_matrix(self):
        r = batch_rank(np.array([[0.9]], dtype=np.float32))
        assert r == []   # только i==j, пропускается

    def test_idx_bounds(self):
        m = self._mat(5)
        for p in batch_rank(m):
            assert 0 <= p.idx1 < 5
            assert 0 <= p.idx2 < 5
