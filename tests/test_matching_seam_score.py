"""Тесты для puzzle_reconstruction.matching.seam_score."""
import pytest
import numpy as np
from puzzle_reconstruction.matching.seam_score import (
    SeamScoreResult,
    compute_seam_score,
    seam_score_matrix,
    normalize_seam_scores,
    rank_candidates,
    batch_seam_scores,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _rand_gray(h=64, w=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rand_rgb(h=64, w=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _seam_result(score=0.7) -> SeamScoreResult:
    return SeamScoreResult(
        score=score,
        component_scores={"profile": 0.8, "color": 0.7, "texture": 0.6, "gradient": 0.7},
        side1=1,
        side2=3,
    )


# ─── TestSeamScoreResult ──────────────────────────────────────────────────────

class TestSeamScoreResult:
    def test_score_stored(self):
        r = _seam_result(0.75)
        assert r.score == pytest.approx(0.75)

    def test_component_scores_stored(self):
        r = _seam_result()
        assert "profile" in r.component_scores
        assert "color" in r.component_scores
        assert "texture" in r.component_scores
        assert "gradient" in r.component_scores

    def test_sides_stored(self):
        r = _seam_result()
        assert r.side1 == 1
        assert r.side2 == 3

    def test_method_seam(self):
        r = _seam_result()
        assert r.method == "seam"

    def test_params_default_empty(self):
        r = _seam_result()
        assert isinstance(r.params, dict)

    def test_custom_sides(self):
        r = SeamScoreResult(score=0.5, component_scores={}, side1=0, side2=2)
        assert r.side1 == 0
        assert r.side2 == 2

    def test_repr_ok(self):
        r = _seam_result()
        s = repr(r)
        assert "SeamScoreResult" in s


# ─── TestComputeSeamScore ─────────────────────────────────────────────────────

class TestComputeSeamScore:
    def test_returns_seam_score_result(self):
        r = compute_seam_score(_rand_gray(), _rand_gray())
        assert isinstance(r, SeamScoreResult)

    def test_score_in_range(self):
        r = compute_seam_score(_rand_gray(seed=0), _rand_gray(seed=1))
        assert 0.0 <= r.score <= 1.0

    def test_method_seam(self):
        r = compute_seam_score(_rand_gray(), _rand_gray())
        assert r.method == "seam"

    def test_four_components(self):
        r = compute_seam_score(_rand_gray(), _rand_gray())
        assert set(r.component_scores.keys()) == {"profile", "color", "texture", "gradient"}

    def test_components_in_range(self):
        r = compute_seam_score(_rand_gray(seed=2), _rand_gray(seed=3))
        for k, v in r.component_scores.items():
            assert 0.0 <= v <= 1.0, f"component {k} = {v} out of range"

    def test_sides_stored(self):
        r = compute_seam_score(_rand_gray(), _rand_gray(), side1=0, side2=2)
        assert r.side1 == 0
        assert r.side2 == 2

    def test_default_sides(self):
        r = compute_seam_score(_rand_gray(), _rand_gray())
        assert r.side1 == 1
        assert r.side2 == 3

    def test_identical_images_high_score(self):
        img = _rand_gray(seed=42)
        r = compute_seam_score(img, img)
        assert r.score > 0.5

    def test_custom_weights(self):
        weights = {"profile": 1.0, "color": 0.0, "texture": 0.0, "gradient": 0.0}
        r = compute_seam_score(_rand_gray(), _rand_gray(), weights=weights)
        assert isinstance(r, SeamScoreResult)

    def test_rgb_input_ok(self):
        r = compute_seam_score(_rand_rgb(), _rand_rgb())
        assert 0.0 <= r.score <= 1.0

    def test_params_has_weights(self):
        r = compute_seam_score(_rand_gray(), _rand_gray())
        assert "weights" in r.params

    def test_params_has_n_samples(self):
        r = compute_seam_score(_rand_gray(), _rand_gray(), n_samples=32)
        assert r.params["n_samples"] == 32

    def test_all_sides_ok(self):
        img1 = _rand_gray()
        img2 = _rand_gray()
        for s1 in range(4):
            for s2 in range(4):
                r = compute_seam_score(img1, img2, side1=s1, side2=s2)
                assert 0.0 <= r.score <= 1.0


# ─── TestSeamScoreMatrix ──────────────────────────────────────────────────────

class TestSeamScoreMatrix:
    def test_returns_list(self):
        imgs = [_rand_gray(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3)]
        result = seam_score_matrix(imgs, pairs)
        assert isinstance(result, list)

    def test_length_matches_pairs(self):
        imgs = [_rand_gray(seed=i) for i in range(4)]
        pairs = [(0, 1, 1, 3), (1, 1, 2, 3), (0, 2, 3, 0)]
        result = seam_score_matrix(imgs, pairs)
        assert len(result) == 3

    def test_empty_pairs(self):
        imgs = [_rand_gray()]
        result = seam_score_matrix(imgs, [])
        assert result == []

    def test_all_seam_score_results(self):
        imgs = [_rand_gray(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (1, 0, 2, 2)]
        for r in seam_score_matrix(imgs, pairs):
            assert isinstance(r, SeamScoreResult)

    def test_scores_in_range(self):
        imgs = [_rand_gray(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (0, 2, 2, 0)]
        for r in seam_score_matrix(imgs, pairs):
            assert 0.0 <= r.score <= 1.0


# ─── TestNormalizeSeamScores ──────────────────────────────────────────────────

class TestNormalizeSeamScores:
    def test_empty_list(self):
        assert normalize_seam_scores([]) == []

    def test_returns_list(self):
        result = normalize_seam_scores([0.3, 0.7, 0.5])
        assert isinstance(result, list)

    def test_length_preserved(self):
        scores = [0.2, 0.5, 0.8]
        assert len(normalize_seam_scores(scores)) == 3

    def test_range_zero_to_one(self):
        scores = [0.1, 0.5, 0.9, 0.3]
        normalized = normalize_seam_scores(scores)
        assert min(normalized) == pytest.approx(0.0, abs=1e-9)
        assert max(normalized) == pytest.approx(1.0, abs=1e-9)

    def test_all_same_returns_ones(self):
        scores = [0.5, 0.5, 0.5]
        normalized = normalize_seam_scores(scores)
        assert all(v == pytest.approx(1.0) for v in normalized)

    def test_single_value(self):
        normalized = normalize_seam_scores([0.7])
        assert normalized == [pytest.approx(1.0)]

    def test_monotone_order_preserved(self):
        scores = [0.1, 0.3, 0.7, 0.9]
        normalized = normalize_seam_scores(scores)
        for i in range(len(normalized) - 1):
            assert normalized[i] <= normalized[i + 1]

    def test_values_in_range(self):
        scores = [0.0, 0.5, 1.0]
        normalized = normalize_seam_scores(scores)
        for v in normalized:
            assert 0.0 <= v <= 1.0


# ─── TestRankCandidates ───────────────────────────────────────────────────────

class TestRankCandidates:
    def test_returns_list(self):
        result = rank_candidates([(0, 0.8), (1, 0.5)])
        assert isinstance(result, list)

    def test_sorted_descending(self):
        scores = [(0, 0.3), (1, 0.9), (2, 0.6)]
        result = rank_candidates(scores)
        vals = [v for _, v in result]
        assert vals == sorted(vals, reverse=True)

    def test_empty_list(self):
        assert rank_candidates([]) == []

    def test_length_preserved(self):
        scores = [(i, float(i) * 0.1) for i in range(5)]
        assert len(rank_candidates(scores)) == 5

    def test_ids_preserved(self):
        scores = [(42, 0.7), (7, 0.9)]
        result = rank_candidates(scores)
        ids = [i for i, _ in result]
        assert set(ids) == {42, 7}

    def test_single_candidate(self):
        result = rank_candidates([(5, 0.6)])
        assert len(result) == 1
        assert result[0][0] == 5

    def test_equal_scores_all_kept(self):
        scores = [(i, 0.5) for i in range(4)]
        result = rank_candidates(scores)
        assert len(result) == 4


# ─── TestBatchSeamScores ──────────────────────────────────────────────────────

class TestBatchSeamScores:
    def test_returns_list(self):
        imgs = [_rand_gray(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3)]
        result = batch_seam_scores(imgs, pairs)
        assert isinstance(result, list)

    def test_same_as_seam_score_matrix(self):
        imgs = [_rand_gray(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (1, 0, 2, 2)]
        r1 = seam_score_matrix(imgs, pairs)
        r2 = batch_seam_scores(imgs, pairs)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.score == pytest.approx(b.score, abs=1e-5)

    def test_empty_pairs(self):
        imgs = [_rand_gray()]
        result = batch_seam_scores(imgs, [])
        assert result == []

    def test_all_seam_score_results(self):
        imgs = [_rand_gray(seed=i) for i in range(2)]
        pairs = [(0, 1, 1, 3)]
        for r in batch_seam_scores(imgs, pairs):
            assert isinstance(r, SeamScoreResult)

    def test_custom_n_samples(self):
        imgs = [_rand_gray(seed=i) for i in range(2)]
        pairs = [(0, 1, 1, 3)]
        result = batch_seam_scores(imgs, pairs, n_samples=32)
        assert result[0].params["n_samples"] == 32
