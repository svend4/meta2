"""Extra tests for puzzle_reconstruction.matching.seam_score."""
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


def _gray(h=64, w=64, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _rand_gray(h=64, w=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rand_rgb(h=64, w=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── SeamScoreResult extras ───────────────────────────────────────────────────

class TestSeamScoreResultExtra:
    def test_score_zero(self):
        r = SeamScoreResult(score=0.0, component_scores={}, side1=0, side2=0)
        assert r.score == pytest.approx(0.0)

    def test_score_one(self):
        r = SeamScoreResult(score=1.0, component_scores={}, side1=0, side2=0)
        assert r.score == pytest.approx(1.0)

    def test_empty_component_scores(self):
        r = SeamScoreResult(score=0.5, component_scores={}, side1=0, side2=2)
        assert r.component_scores == {}

    def test_component_values_stored(self):
        cs = {"profile": 0.1, "color": 0.2, "texture": 0.3, "gradient": 0.4}
        r = SeamScoreResult(score=0.5, component_scores=cs, side1=1, side2=3)
        assert r.component_scores["texture"] == pytest.approx(0.3)

    def test_side_pairs_all_combinations(self):
        for s1 in range(4):
            for s2 in range(4):
                r = SeamScoreResult(score=0.5, component_scores={}, side1=s1, side2=s2)
                assert r.side1 == s1
                assert r.side2 == s2

    def test_params_is_dict(self):
        r = SeamScoreResult(score=0.5, component_scores={}, side1=0, side2=0)
        assert isinstance(r.params, dict)

    def test_repr_contains_score(self):
        r = SeamScoreResult(score=0.42, component_scores={}, side1=0, side2=0)
        assert "0.42" in repr(r) or "SeamScoreResult" in repr(r)


# ─── compute_seam_score extras ────────────────────────────────────────────────

class TestComputeSeamScoreExtra:
    def test_uniform_images_score_in_range(self):
        img = _gray()
        r = compute_seam_score(img, img)
        assert 0.0 <= r.score <= 1.0

    def test_completely_different_images(self):
        img1 = _gray(val=0)
        img2 = _gray(val=255)
        r = compute_seam_score(img1, img2)
        assert 0.0 <= r.score <= 1.0

    def test_non_square_images(self):
        img1 = _rand_gray(h=32, w=64)
        img2 = _rand_gray(h=64, w=32)
        r = compute_seam_score(img1, img2)
        assert isinstance(r, SeamScoreResult)

    def test_small_images(self):
        img1 = _rand_gray(h=8, w=8)
        img2 = _rand_gray(h=8, w=8)
        r = compute_seam_score(img1, img2)
        assert 0.0 <= r.score <= 1.0

    def test_large_images(self):
        img1 = _rand_gray(h=128, w=128, seed=10)
        img2 = _rand_gray(h=128, w=128, seed=11)
        r = compute_seam_score(img1, img2)
        assert 0.0 <= r.score <= 1.0

    def test_custom_n_samples_stored(self):
        r = compute_seam_score(_rand_gray(), _rand_gray(), n_samples=16)
        assert r.params.get("n_samples") == 16

    def test_weights_all_texture(self):
        weights = {"profile": 0.0, "color": 0.0, "texture": 1.0, "gradient": 0.0}
        r = compute_seam_score(_rand_gray(), _rand_gray(), weights=weights)
        assert isinstance(r.score, float)

    def test_weights_all_gradient(self):
        weights = {"profile": 0.0, "color": 0.0, "texture": 0.0, "gradient": 1.0}
        r = compute_seam_score(_rand_gray(), _rand_gray(), weights=weights)
        assert isinstance(r.score, float)

    def test_side_0_3(self):
        r = compute_seam_score(_rand_gray(), _rand_gray(), side1=0, side2=3)
        assert r.side1 == 0
        assert r.side2 == 3

    def test_side_2_1(self):
        r = compute_seam_score(_rand_gray(), _rand_gray(), side1=2, side2=1)
        assert r.side1 == 2
        assert r.side2 == 1

    def test_rgb_all_sides_ok(self):
        img1 = _rand_rgb()
        img2 = _rand_rgb()
        for s in range(4):
            r = compute_seam_score(img1, img2, side1=s, side2=(s + 2) % 4)
            assert 0.0 <= r.score <= 1.0


# ─── seam_score_matrix extras ─────────────────────────────────────────────────

class TestSeamScoreMatrixExtra:
    def test_single_pair(self):
        imgs = [_rand_gray(seed=0), _rand_gray(seed=1)]
        result = seam_score_matrix(imgs, [(0, 1, 1, 3)])
        assert len(result) == 1

    def test_five_pairs(self):
        imgs = [_rand_gray(seed=i) for i in range(5)]
        pairs = [(i, (i + 1) % 5, 1, 3) for i in range(5)]
        result = seam_score_matrix(imgs, pairs)
        assert len(result) == 5

    def test_all_results_in_range(self):
        imgs = [_rand_gray(seed=i) for i in range(4)]
        pairs = [(0, 1, 1, 3), (1, 2, 1, 3), (2, 3, 1, 3)]
        for r in seam_score_matrix(imgs, pairs):
            assert 0.0 <= r.score <= 1.0

    def test_same_image_pair_high_score(self):
        img = _rand_gray(seed=7)
        result = seam_score_matrix([img, img], [(0, 1, 1, 3)])
        assert result[0].score > 0.5

    def test_rgb_pairs_ok(self):
        imgs = [_rand_rgb(seed=i) for i in range(3)]
        pairs = [(0, 1, 0, 2), (1, 2, 1, 3)]
        result = seam_score_matrix(imgs, pairs)
        assert len(result) == 2


# ─── normalize_seam_scores extras ────────────────────────────────────────────

class TestNormalizeSeamScoresExtra:
    def test_two_values(self):
        result = normalize_seam_scores([0.2, 0.8])
        assert min(result) == pytest.approx(0.0, abs=1e-9)
        assert max(result) == pytest.approx(1.0, abs=1e-9)

    def test_already_zero_one(self):
        result = normalize_seam_scores([0.0, 1.0])
        assert result[0] == pytest.approx(0.0, abs=1e-9)
        assert result[1] == pytest.approx(1.0, abs=1e-9)

    def test_all_zeros(self):
        result = normalize_seam_scores([0.0, 0.0, 0.0])
        for v in result:
            assert v == pytest.approx(1.0) or v == pytest.approx(0.0) or 0.0 <= v <= 1.0

    def test_large_list(self):
        scores = list(np.linspace(0.0, 1.0, 100))
        result = normalize_seam_scores(scores)
        assert len(result) == 100
        assert min(result) == pytest.approx(0.0, abs=1e-9)
        assert max(result) == pytest.approx(1.0, abs=1e-9)

    def test_negative_values_handled(self):
        result = normalize_seam_scores([-1.0, 0.0, 1.0])
        assert min(result) == pytest.approx(0.0, abs=1e-9)
        assert max(result) == pytest.approx(1.0, abs=1e-9)


# ─── rank_candidates extras ───────────────────────────────────────────────────

class TestRankCandidatesExtra:
    def test_single_entry(self):
        result = rank_candidates([(3, 0.7)])
        assert len(result) == 1
        assert result[0][0] == 3

    def test_two_entries_correct_order(self):
        result = rank_candidates([(0, 0.4), (1, 0.9)])
        assert result[0][1] == pytest.approx(0.9)
        assert result[1][1] == pytest.approx(0.4)

    def test_large_list_sorted(self):
        import random
        rng = random.Random(0)
        scores = [(i, rng.random()) for i in range(50)]
        result = rank_candidates(scores)
        vals = [v for _, v in result]
        assert vals == sorted(vals, reverse=True)

    def test_all_same_score_all_present(self):
        pairs = [(i, 0.5) for i in range(5)]
        result = rank_candidates(pairs)
        assert {i for i, _ in result} == {0, 1, 2, 3, 4}

    def test_negative_score(self):
        result = rank_candidates([(0, -0.5), (1, 0.5)])
        assert result[0][1] == pytest.approx(0.5)

    def test_zero_score(self):
        result = rank_candidates([(0, 0.0), (1, 1.0)])
        assert result[0][1] == pytest.approx(1.0)


# ─── batch_seam_scores extras ─────────────────────────────────────────────────

class TestBatchSeamScoresExtra:
    def test_single_pair_result(self):
        imgs = [_rand_gray(seed=0), _rand_gray(seed=1)]
        result = batch_seam_scores(imgs, [(0, 1, 1, 3)])
        assert len(result) == 1

    def test_scores_in_range(self):
        imgs = [_rand_gray(seed=i) for i in range(4)]
        pairs = [(0, 1, 1, 3), (2, 3, 0, 2)]
        for r in batch_seam_scores(imgs, pairs):
            assert 0.0 <= r.score <= 1.0

    def test_sides_stored_in_results(self):
        # pair format: (img_i, side1, img_j, side2)
        imgs = [_rand_gray(seed=i) for i in range(2)]
        result = batch_seam_scores(imgs, [(0, 2, 1, 0)])
        assert result[0].side1 == 2
        assert result[0].side2 == 0

    def test_n_samples_forwarded(self):
        imgs = [_rand_gray(seed=i) for i in range(2)]
        result = batch_seam_scores(imgs, [(0, 1, 1, 3)], n_samples=8)
        assert result[0].params.get("n_samples") == 8

    def test_rgb_input(self):
        imgs = [_rand_rgb(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (1, 2, 0, 2)]
        result = batch_seam_scores(imgs, pairs)
        assert len(result) == 2
        for r in result:
            assert isinstance(r, SeamScoreResult)
