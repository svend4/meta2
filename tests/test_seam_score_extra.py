"""Extra tests for puzzle_reconstruction/matching/seam_score.py."""
import numpy as np
import pytest

from puzzle_reconstruction.matching.seam_score import (
    SeamScoreResult,
    compute_seam_score,
    seam_score_matrix,
    normalize_seam_scores,
    rank_candidates,
    batch_seam_scores,
)


def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=64, w=64, seed=5):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── SeamScoreResult extras ───────────────────────────────────────────────────

class TestSeamScoreResultExtra:
    def test_score_float(self):
        r = SeamScoreResult(score=0.5, component_scores={}, side1=0, side2=0)
        assert isinstance(r.score, float)

    def test_method_is_seam(self):
        r = SeamScoreResult(score=0.3, component_scores={}, side1=1, side2=2)
        assert r.method == "seam"

    def test_params_is_dict(self):
        r = SeamScoreResult(score=0.3, component_scores={}, side1=0, side2=0)
        assert isinstance(r.params, dict)

    def test_score_boundaries(self):
        for score in (0.0, 0.5, 1.0):
            r = SeamScoreResult(score=score, component_scores={}, side1=0, side2=0)
            assert r.score == pytest.approx(score)

    def test_all_side_combos(self):
        for s1 in range(4):
            for s2 in range(4):
                r = SeamScoreResult(score=0.5, component_scores={}, side1=s1, side2=s2)
                assert r.side1 == s1 and r.side2 == s2

    def test_component_scores_keys_stored(self):
        cs = {"profile": 0.4, "color": 0.6, "texture": 0.5, "gradient": 0.7}
        r = SeamScoreResult(score=0.5, component_scores=cs, side1=0, side2=0)
        assert set(r.component_scores.keys()) == {"profile", "color", "texture", "gradient"}

    def test_repr_string(self):
        r = SeamScoreResult(score=0.7, component_scores={}, side1=0, side2=0)
        assert isinstance(repr(r), str)


# ─── compute_seam_score extras ────────────────────────────────────────────────

class TestComputeSeamScoreExtra:
    def test_uniform_image_returns_result(self):
        img = _gray()
        r = compute_seam_score(img, img)
        assert isinstance(r, SeamScoreResult)

    def test_score_zero_one_range_noisy(self):
        for seed in range(5):
            r = compute_seam_score(_noisy(seed=seed), _noisy(seed=seed + 5))
            assert 0.0 <= r.score <= 1.0

    def test_identical_images_high_score(self):
        img = _noisy(seed=3)
        r = compute_seam_score(img, img)
        assert r.score > 0.5

    def test_all_four_sides_pair(self):
        img1 = _noisy(seed=0)
        img2 = _noisy(seed=1)
        for s in range(4):
            r = compute_seam_score(img1, img2, side1=s, side2=(s + 2) % 4)
            assert 0.0 <= r.score <= 1.0

    def test_border_frac_stored(self):
        r = compute_seam_score(_noisy(), _noisy(), border_frac=0.15)
        assert r.params.get("border_frac") == pytest.approx(0.15)

    def test_small_8x8_image(self):
        img = np.full((8, 8), 50, dtype=np.uint8)
        r = compute_seam_score(img, img)
        assert isinstance(r, SeamScoreResult)

    def test_non_square_images(self):
        img1 = _noisy(h=32, w=64, seed=0)
        img2 = _noisy(h=64, w=32, seed=1)
        r = compute_seam_score(img1, img2)
        assert isinstance(r, SeamScoreResult)

    def test_bgr_components_in_range(self):
        img = _bgr()
        r = compute_seam_score(img, img)
        for v in r.component_scores.values():
            assert 0.0 <= v <= 1.0

    def test_zero_weight_component_profile(self):
        w = {"profile": 0.0, "color": 0.5, "texture": 0.3, "gradient": 0.2}
        r = compute_seam_score(_noisy(), _noisy(), weights=w)
        assert 0.0 <= r.score <= 1.0

    def test_n_samples_16(self):
        r = compute_seam_score(_noisy(), _noisy(), n_samples=16)
        assert r.params.get("n_samples") == 16

    def test_side_stored_correctly(self):
        r = compute_seam_score(_noisy(), _noisy(), side1=3, side2=1)
        assert r.side1 == 3
        assert r.side2 == 1


# ─── seam_score_matrix extras ─────────────────────────────────────────────────

class TestSeamScoreMatrixExtra:
    def test_single_pair_length_1(self):
        imgs = [_noisy(seed=0), _noisy(seed=1)]
        result = seam_score_matrix(imgs, [(0, 1, 1, 3)])
        assert len(result) == 1

    def test_six_pairs(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        pairs = [(i, 1, j, 3) for i in range(4) for j in range(i + 1, 4)]
        result = seam_score_matrix(imgs, pairs)
        assert len(result) == 6

    def test_all_results_seam_method(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (1, 0, 2, 2)]
        for r in seam_score_matrix(imgs, pairs):
            assert r.method == "seam"

    def test_results_score_in_range(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (0, 2, 2, 0)]
        for r in seam_score_matrix(imgs, pairs):
            assert 0.0 <= r.score <= 1.0

    def test_bgr_images(self):
        imgs = [_bgr(seed=i) for i in range(2)]
        result = seam_score_matrix(imgs, [(0, 1, 1, 3)])
        assert len(result) == 1
        assert isinstance(result[0], SeamScoreResult)


# ─── normalize_seam_scores extras ────────────────────────────────────────────

class TestNormalizeSeamScoresExtra:
    def test_single_value_is_one(self):
        result = normalize_seam_scores([0.3])
        assert result[0] == pytest.approx(1.0)

    def test_two_distinct_values(self):
        result = normalize_seam_scores([0.2, 0.8])
        assert min(result) == pytest.approx(0.0, abs=1e-9)
        assert max(result) == pytest.approx(1.0, abs=1e-9)

    def test_order_preserved(self):
        scores = [0.1, 0.4, 0.6, 0.9]
        result = normalize_seam_scores(scores)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]

    def test_large_input(self):
        scores = list(np.linspace(0.0, 1.0, 100))
        result = normalize_seam_scores(scores)
        assert len(result) == 100
        assert result[0] == pytest.approx(0.0, abs=1e-9)
        assert result[-1] == pytest.approx(1.0, abs=1e-9)

    def test_all_same_normalized_to_one(self):
        result = normalize_seam_scores([0.7, 0.7, 0.7])
        assert all(v == pytest.approx(1.0) for v in result)

    def test_values_in_0_1(self):
        scores = [0.1, 0.9, 0.3, 0.5, 0.7]
        for v in normalize_seam_scores(scores):
            assert 0.0 <= v <= 1.0


# ─── rank_candidates extras ───────────────────────────────────────────────────

class TestRankCandidatesExtra:
    def test_two_items_sorted(self):
        result = rank_candidates([(0, 0.3), (1, 0.9)])
        assert result[0][1] == pytest.approx(0.9)

    def test_tie_handling(self):
        result = rank_candidates([(0, 0.5), (1, 0.5), (2, 0.5)])
        assert len(result) == 3
        assert all(v == pytest.approx(0.5) for _, v in result)

    def test_large_list_descending(self):
        pairs = [(i, float(i) * 0.01) for i in range(100)]
        result = rank_candidates(pairs)
        vals = [v for _, v in result]
        assert vals == sorted(vals, reverse=True)

    def test_ids_all_present(self):
        pairs = [(i, float(i)) for i in range(10)]
        result = rank_candidates(pairs)
        assert {i for i, _ in result} == set(range(10))

    def test_negative_score_included(self):
        result = rank_candidates([(-1, -0.5), (0, 0.1)])
        assert result[0][1] == pytest.approx(0.1)


# ─── batch_seam_scores extras ─────────────────────────────────────────────────

class TestBatchSeamScoresExtra:
    def test_equals_seam_score_matrix(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (1, 0, 2, 2)]
        r1 = seam_score_matrix(imgs, pairs)
        r2 = batch_seam_scores(imgs, pairs)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.score == pytest.approx(b.score, abs=1e-5)

    def test_bgr_images_ok(self):
        imgs = [_bgr(seed=i) for i in range(2)]
        result = batch_seam_scores(imgs, [(0, 1, 1, 3)])
        assert len(result) == 1
        assert 0.0 <= result[0].score <= 1.0

    def test_single_pair_sides_stored(self):
        imgs = [_noisy(seed=0), _noisy(seed=1)]
        result = batch_seam_scores(imgs, [(0, 3, 1, 1)])
        assert result[0].side1 == 3
        assert result[0].side2 == 1

    def test_n_samples_propagated(self):
        imgs = [_noisy(seed=i) for i in range(2)]
        result = batch_seam_scores(imgs, [(0, 1, 1, 3)], n_samples=8)
        assert result[0].params.get("n_samples") == 8

    def test_all_results_in_range(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        pairs = [(i, 1, j, 3) for i in range(4) for j in range(i + 1, 4)]
        for r in batch_seam_scores(imgs, pairs):
            assert 0.0 <= r.score <= 1.0
