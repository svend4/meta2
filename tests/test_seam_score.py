"""Тесты для puzzle_reconstruction/matching/seam_score.py."""
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


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=64, w=64, seed=5):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 150
    img[:, :, 2] = 80
    return img


_COMP_KEYS = {"profile", "color", "texture", "gradient"}


# ─── SeamScoreResult ──────────────────────────────────────────────────────────

class TestSeamScoreResult:
    def _make(self):
        return SeamScoreResult(
            score=0.75,
            component_scores={"profile": 0.8, "color": 0.7,
                               "texture": 0.75, "gradient": 0.6},
            side1=1, side2=3,
        )

    def test_fields(self):
        r = self._make()
        assert r.score == pytest.approx(0.75)
        assert r.side1 == 1
        assert r.side2 == 3

    def test_component_scores_dict(self):
        r = self._make()
        assert isinstance(r.component_scores, dict)
        assert set(r.component_scores.keys()) >= {"profile", "color"}

    def test_method_default(self):
        r = self._make()
        assert r.method == "seam"

    def test_params_default(self):
        r = self._make()
        assert isinstance(r.params, dict)

    def test_repr(self):
        r = self._make()
        s = repr(r)
        assert "SeamScoreResult" in s
        assert "0.75" in s

    def test_score_stored(self):
        r = SeamScoreResult(score=0.42, component_scores={},
                             side1=0, side2=2)
        assert r.score == pytest.approx(0.42)

    def test_sides_stored(self):
        r = SeamScoreResult(score=0.5, component_scores={},
                             side1=2, side2=0)
        assert r.side1 == 2
        assert r.side2 == 0


# ─── compute_seam_score ───────────────────────────────────────────────────────

class TestComputeSeamScore:
    def test_returns_result(self):
        assert isinstance(compute_seam_score(_noisy(), _noisy(seed=9)), SeamScoreResult)

    def test_score_in_range(self):
        r = compute_seam_score(_noisy(), _noisy(seed=9))
        assert 0.0 <= r.score <= 1.0

    def test_method(self):
        assert compute_seam_score(_noisy(), _noisy()).method == "seam"

    def test_all_components_present(self):
        r = compute_seam_score(_noisy(), _noisy(seed=2))
        assert _COMP_KEYS == set(r.component_scores.keys())

    def test_component_scores_in_range(self):
        r = compute_seam_score(_noisy(), _noisy(seed=3))
        for k, v in r.component_scores.items():
            assert 0.0 <= v <= 1.0, f"component {k!r} out of range: {v}"

    def test_gray_input(self):
        r = compute_seam_score(_gray(), _gray())
        assert isinstance(r, SeamScoreResult)

    def test_bgr_input(self):
        r = compute_seam_score(_bgr(), _bgr())
        assert 0.0 <= r.score <= 1.0

    def test_n_samples_stored(self):
        r = compute_seam_score(_noisy(), _noisy(), n_samples=32)
        assert r.params.get("n_samples") == 32

    def test_border_frac_stored(self):
        r = compute_seam_score(_noisy(), _noisy(), border_frac=0.1)
        assert r.params.get("border_frac") == pytest.approx(0.1)

    def test_weights_normalised_stored(self):
        w = {"profile": 1.0, "color": 1.0, "texture": 1.0, "gradient": 1.0}
        r = compute_seam_score(_noisy(), _noisy(), weights=w)
        stored = r.params.get("weights", {})
        total  = sum(stored.values()) if stored else 1.0
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_custom_weights_accepted(self):
        w = {"profile": 0.8, "color": 0.1, "texture": 0.05, "gradient": 0.05}
        r = compute_seam_score(_noisy(), _noisy(), weights=w)
        assert isinstance(r, SeamScoreResult)

    def test_identical_high_score(self):
        img = _noisy()
        r   = compute_seam_score(img, img, side1=1, side2=1)
        assert r.score > 0.5

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        r = compute_seam_score(_noisy(), _noisy(seed=7), side1=side, side2=side)
        assert 0.0 <= r.score <= 1.0

    def test_side1_stored(self):
        r = compute_seam_score(_noisy(), _noisy(), side1=2, side2=0)
        assert r.side1 == 2
        assert r.side2 == 0

    def test_tiny_image(self):
        img = np.full((8, 8), 100, dtype=np.uint8)
        r   = compute_seam_score(img, img)
        assert isinstance(r, SeamScoreResult)


# ─── seam_score_matrix ────────────────────────────────────────────────────────

class TestSeamScoreMatrix:
    def test_returns_list(self):
        imgs  = [_noisy(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (1, 0, 2, 2)]
        r = seam_score_matrix(imgs, pairs)
        assert isinstance(r, list)
        assert len(r) == 2

    def test_each_is_result(self):
        imgs  = [_noisy(), _noisy(seed=1)]
        pairs = [(0, 1, 1, 3)]
        for r in seam_score_matrix(imgs, pairs):
            assert isinstance(r, SeamScoreResult)

    def test_empty_pairs(self):
        assert seam_score_matrix([_noisy()], []) == []

    def test_scores_in_range(self):
        imgs  = [_noisy(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (0, 0, 2, 2), (1, 2, 2, 0)]
        for r in seam_score_matrix(imgs, pairs):
            assert 0.0 <= r.score <= 1.0

    def test_pair_sides_stored(self):
        imgs  = [_noisy(), _noisy(seed=1)]
        pairs = [(0, 2, 1, 0)]
        r = seam_score_matrix(imgs, pairs)
        assert r[0].side1 == 2
        assert r[0].side2 == 0


# ─── normalize_seam_scores ────────────────────────────────────────────────────

class TestNormalizeSeamScores:
    def test_empty_input(self):
        assert normalize_seam_scores([]) == []

    def test_constant_all_ones(self):
        result = normalize_seam_scores([0.5, 0.5, 0.5])
        assert all(v == pytest.approx(1.0) for v in result)

    def test_min_is_zero(self):
        result = normalize_seam_scores([0.2, 0.5, 0.8])
        assert result[0] == pytest.approx(0.0)

    def test_max_is_one(self):
        result = normalize_seam_scores([0.2, 0.5, 0.8])
        assert result[-1] == pytest.approx(1.0)

    def test_mid_value(self):
        result = normalize_seam_scores([0.0, 0.5, 1.0])
        assert result[1] == pytest.approx(0.5)

    def test_length_preserved(self):
        scores = [0.1, 0.3, 0.7, 0.9]
        assert len(normalize_seam_scores(scores)) == 4

    def test_single_item_all_ones(self):
        result = normalize_seam_scores([0.6])
        assert result[0] == pytest.approx(1.0)

    def test_returns_list(self):
        assert isinstance(normalize_seam_scores([0.3, 0.7]), list)


# ─── rank_candidates ──────────────────────────────────────────────────────────

class TestRankCandidates:
    def test_sorted_descending(self):
        inp    = [(0, 0.3), (1, 0.9), (2, 0.6)]
        result = rank_candidates(inp)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_is_highest(self):
        inp    = [(5, 0.2), (3, 0.8), (1, 0.5)]
        result = rank_candidates(inp)
        assert result[0][0] == 3

    def test_empty_returns_empty(self):
        assert rank_candidates([]) == []

    def test_single_item_unchanged(self):
        result = rank_candidates([(7, 0.5)])
        assert result == [(7, 0.5)]

    def test_returns_list(self):
        assert isinstance(rank_candidates([(0, 0.9), (1, 0.1)]), list)

    def test_ids_preserved(self):
        inp    = [(10, 0.4), (20, 0.9), (30, 0.1)]
        result = rank_candidates(inp)
        ids    = [i for i, _ in result]
        assert set(ids) == {10, 20, 30}

    def test_ties_both_present(self):
        inp    = [(0, 0.5), (1, 0.5)]
        result = rank_candidates(inp)
        assert len(result) == 2


# ─── batch_seam_scores ────────────────────────────────────────────────────────

class TestBatchSeamScores:
    def test_returns_list(self):
        imgs  = [_noisy(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (0, 0, 2, 2)]
        r = batch_seam_scores(imgs, pairs)
        assert isinstance(r, list)
        assert len(r) == 2

    def test_each_is_result(self):
        imgs  = [_noisy(), _noisy(seed=1)]
        for r in batch_seam_scores(imgs, [(0, 1, 1, 3)]):
            assert isinstance(r, SeamScoreResult)

    def test_empty_pairs(self):
        assert batch_seam_scores([_noisy()], []) == []

    def test_scores_in_range(self):
        imgs  = [_noisy(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (1, 0, 2, 2)]
        for r in batch_seam_scores(imgs, pairs):
            assert 0.0 <= r.score <= 1.0

    def test_n_samples_forwarded(self):
        imgs  = [_noisy(), _noisy(seed=1)]
        r = batch_seam_scores(imgs, [(0, 1, 1, 3)], n_samples=32)
        assert r[0].params.get("n_samples") == 32

    def test_bgr_input(self):
        imgs = [_bgr(), _bgr()]
        r    = batch_seam_scores(imgs, [(0, 1, 1, 3)])
        assert 0.0 <= r[0].score <= 1.0
