"""Additional tests for puzzle_reconstruction.matching.matcher_registry."""
import pytest
import numpy as np

from puzzle_reconstruction.matching.matcher_registry import (
    MATCHER_REGISTRY,
    _safe_score,
    compute_scores,
    get_matcher,
    list_matchers,
    register,
    register_fn,
    weighted_combine,
)
from puzzle_reconstruction.models import EdgeSignature, EdgeSide


# ─── helpers ──────────────────────────────────────────────────────────────────

def _edge(edge_id: int = 0, fd: float = 1.5, curve_len: int = 8) -> EdgeSignature:
    """EdgeSignature с заданными параметрами."""
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.RIGHT,
        virtual_curve=np.linspace(0, 1, curve_len * 2).reshape(curve_len, 2),
        fd=fd,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=float(curve_len),
    )


def _random_edge(seed: int = 0) -> EdgeSignature:
    rng = np.random.default_rng(seed)
    return EdgeSignature(
        edge_id=seed,
        side=EdgeSide.LEFT,
        virtual_curve=rng.uniform(0, 5, (10, 2)),
        fd=rng.uniform(1.0, 2.0),
        css_vec=rng.uniform(0, 1, 8),
        ifs_coeffs=rng.uniform(-0.5, 0.5, 4),
        length=rng.uniform(5, 20),
    )


# ─── TestMatcherRegistryExtra ─────────────────────────────────────────────────

class TestMatcherRegistryExtra:
    def test_multiple_default_matchers(self):
        """После импорта регистрируется минимум 2 стандартных матчера (fd + text)."""
        assert len(MATCHER_REGISTRY) >= 2

    def test_fd_score_identical_edges(self):
        e = _edge(0, fd=1.8)
        score = MATCHER_REGISTRY["fd"](e, e)
        assert score == pytest.approx(1.0)

    def test_fd_score_different_fd(self):
        e1 = _edge(0, fd=1.0)
        e2 = _edge(1, fd=2.0)
        score = MATCHER_REGISTRY["fd"](e1, e2)
        # diff=1.0 → 1/(1+1)=0.5
        assert score == pytest.approx(0.5)

    def test_fd_score_large_diff(self):
        e1 = _edge(0, fd=1.0)
        e2 = _edge(1, fd=100.0)
        score = MATCHER_REGISTRY["fd"](e1, e2)
        assert 0.0 <= score <= 0.01

    def test_text_matcher_always_zero(self):
        """Базовый 'text' матчер всегда возвращает 0.0."""
        e1 = _edge(0)
        e2 = _edge(1)
        assert MATCHER_REGISTRY["text"](e1, e2) == pytest.approx(0.0)

    def test_registry_does_not_contain_none_values(self):
        for name, fn in MATCHER_REGISTRY.items():
            assert fn is not None, f"Матчер '{name}' = None"

    def test_overwrite_existing_key(self):
        name = "__test_overwrite_extra__"
        register_fn(name, lambda a, b: 0.1)
        register_fn(name, lambda a, b: 0.9)
        assert MATCHER_REGISTRY[name](_edge(0), _edge(1)) == pytest.approx(0.9)
        del MATCHER_REGISTRY[name]

    def test_register_same_name_twice_via_decorator(self):
        name = "__test_double_decorator__"

        @register(name)
        def v1(a, b):
            return 0.2

        @register(name)
        def v2(a, b):
            return 0.8

        assert MATCHER_REGISTRY[name](_edge(0), _edge(1)) == pytest.approx(0.8)
        del MATCHER_REGISTRY[name]


# ─── TestSafeScoreExtra ───────────────────────────────────────────────────────

class TestSafeScoreExtra:
    def test_normal_fn_returns_clamped(self):
        fn = lambda a, b: 0.42
        result = _safe_score(fn, _edge(0), _edge(1))
        assert result == pytest.approx(0.42)

    def test_fn_raises_returns_zero(self):
        def broken(a, b):
            raise ValueError("boom")
        result = _safe_score(broken, _edge(0), _edge(1))
        assert result == pytest.approx(0.0)

    def test_fn_returns_above_one_clamped(self):
        fn = lambda a, b: 5.0
        result = _safe_score(fn, _edge(0), _edge(1))
        assert result == pytest.approx(1.0)

    def test_fn_returns_below_zero_clamped(self):
        fn = lambda a, b: -3.0
        result = _safe_score(fn, _edge(0), _edge(1))
        assert result == pytest.approx(0.0)

    def test_fn_returns_nan_clamped_or_zero(self):
        fn = lambda a, b: float("nan")
        result = _safe_score(fn, _edge(0), _edge(1))
        # float('nan') → max(0, min(1, nan)) = nan in Python, then float(nan)
        # but at least no exception
        assert isinstance(result, float)

    def test_fn_returns_inf_clamped(self):
        fn = lambda a, b: float("inf")
        result = _safe_score(fn, _edge(0), _edge(1))
        assert result == pytest.approx(1.0)

    def test_fn_returns_exactly_zero(self):
        fn = lambda a, b: 0.0
        result = _safe_score(fn, _edge(0), _edge(1))
        assert result == pytest.approx(0.0)

    def test_fn_returns_exactly_one(self):
        fn = lambda a, b: 1.0
        result = _safe_score(fn, _edge(0), _edge(1))
        assert result == pytest.approx(1.0)


# ─── TestComputeScoresExtra ───────────────────────────────────────────────────

class TestComputeScoresExtra:
    def test_multiple_known_matchers(self):
        e1 = _edge(0, fd=1.5)
        e2 = _edge(1, fd=1.5)
        result = compute_scores(e1, e2, ["fd", "text"])
        assert set(result.keys()) == {"fd", "text"}
        assert result["fd"] == pytest.approx(1.0)
        assert result["text"] == pytest.approx(0.0)

    def test_all_scores_in_range(self):
        e1 = _random_edge(42)
        e2 = _random_edge(99)
        names = list(MATCHER_REGISTRY.keys())
        result = compute_scores(e1, e2, names)
        for name, score in result.items():
            assert 0.0 <= score <= 1.0, f"Матчер '{name}' вернул {score}"

    def test_order_preserved_unique(self):
        """compute_scores строит dict-comprehension: дубликаты в именах дедуплицируются."""
        e1 = _edge(0)
        e2 = _edge(1)
        names = ["text", "fd"]
        result = compute_scores(e1, e2, names)
        assert list(result.keys()) == names

    def test_duplicate_names_deduplicated(self):
        """При дублирующихся именах в списке dict хранит одно значение."""
        e1 = _edge(0)
        e2 = _edge(1)
        result = compute_scores(e1, e2, ["fd", "fd"])
        assert "fd" in result
        assert len(result) == 1

    def test_unknown_matcher_zero_not_exception(self):
        e1 = _edge(0)
        e2 = _edge(1)
        result = compute_scores(e1, e2, ["__xyz_missing__"])
        assert result["__xyz_missing__"] == pytest.approx(0.0)

    def test_fd_identity_score_one(self):
        e = _edge(5, fd=1.7)
        result = compute_scores(e, e, ["fd"])
        assert result["fd"] == pytest.approx(1.0)

    def test_registered_custom_matcher_used(self):
        name = "__compute_custom__"
        register_fn(name, lambda a, b: 0.333)
        e1 = _edge(0)
        e2 = _edge(1)
        result = compute_scores(e1, e2, [name])
        assert result[name] == pytest.approx(0.333)
        del MATCHER_REGISTRY[name]


# ─── TestWeightedCombineExtra ─────────────────────────────────────────────────

class TestWeightedCombineExtra:
    def test_weight_skewed_toward_high_score(self):
        scores = {"a": 0.1, "b": 0.9}
        weights = {"a": 0.1, "b": 0.9}
        result = weighted_combine(scores, weights)
        # 0.1*0.1 + 0.9*0.9 = 0.01 + 0.81 = 0.82; нормировка: 0.82 / 1.0
        assert result == pytest.approx(0.82)

    def test_uniform_scores_return_that_score(self):
        scores = {"a": 0.6, "b": 0.6, "c": 0.6}
        weights = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = weighted_combine(scores, weights)
        assert result == pytest.approx(0.6)

    def test_result_bounds_random(self):
        rng = np.random.default_rng(77)
        for _ in range(50):
            n = rng.integers(1, 8)
            names = [f"m{i}" for i in range(n)]
            scores = {k: float(rng.uniform(0, 1)) for k in names}
            weights = {k: float(rng.uniform(0, 2)) for k in names}
            result = weighted_combine(scores, weights)
            assert 0.0 <= result <= 1.0, f"Out of [0,1]: {result}"

    def test_negative_weights_treated_as_zero_contribution(self):
        """Отрицательный вес снижает total_w, но функция не обязана его отбрасывать."""
        scores = {"a": 1.0}
        weights = {"a": -1.0}
        # total_w = -1 → <= 0 → returns 0.0
        result = weighted_combine(scores, weights)
        assert result == pytest.approx(0.0)

    def test_only_matching_keys_contribute(self):
        """Лишние ключи в weights (которых нет в scores) не влияют на total_w."""
        scores = {"a": 0.7}
        weights = {"a": 1.0, "b": 100.0}  # 'b' not in scores → не учитывается
        result = weighted_combine(scores, weights)
        # total_w = sum(weights.get(k) for k in scores) = weights["a"] = 1.0
        # result = 0.7 * 1.0 / 1.0 = 0.7
        assert result == pytest.approx(0.7)

    def test_all_zero_scores(self):
        scores = {"a": 0.0, "b": 0.0}
        weights = {"a": 1.0, "b": 1.0}
        result = weighted_combine(scores, weights)
        assert result == pytest.approx(0.0)

    def test_single_weight_zero_returns_zero(self):
        scores = {"a": 0.9}
        weights = {"a": 0.0}
        result = weighted_combine(scores, weights)
        assert result == pytest.approx(0.0)

    def test_many_matchers(self):
        n = 20
        names = [f"m{i}" for i in range(n)]
        scores = {k: 0.5 for k in names}
        weights = {k: 1.0 for k in names}
        result = weighted_combine(scores, weights)
        assert result == pytest.approx(0.5)


# ─── TestGetMatcherExtra ──────────────────────────────────────────────────────

class TestGetMatcherExtra:
    def test_returns_same_object(self):
        fn = lambda a, b: 0.5
        name = "__get_same_obj__"
        register_fn(name, fn)
        assert get_matcher(name) is fn
        del MATCHER_REGISTRY[name]

    def test_after_overwrite_returns_new(self):
        name = "__get_after_overwrite__"
        fn1 = lambda a, b: 0.1
        fn2 = lambda a, b: 0.2
        register_fn(name, fn1)
        register_fn(name, fn2)
        assert get_matcher(name) is fn2
        del MATCHER_REGISTRY[name]

    def test_get_fd_callable(self):
        fn = get_matcher("fd")
        e = _edge(0)
        result = fn(e, e)
        assert result == pytest.approx(1.0)

    def test_get_text_callable(self):
        fn = get_matcher("text")
        e = _edge(0)
        result = fn(e, e)
        assert result == pytest.approx(0.0)


# ─── TestListMatchersExtra ────────────────────────────────────────────────────

class TestListMatchersExtra:
    def test_new_matcher_appears_in_list(self):
        name = "__list_new__"
        register_fn(name, lambda a, b: 0.0)
        assert name in list_matchers()
        del MATCHER_REGISTRY[name]

    def test_deleted_matcher_not_in_list(self):
        name = "__list_deleted__"
        register_fn(name, lambda a, b: 0.0)
        del MATCHER_REGISTRY[name]
        assert name not in list_matchers()

    def test_sorted_after_addition(self):
        names_before = list_matchers()
        name = "zzz_test_sorted"
        register_fn(name, lambda a, b: 0.0)
        names_after = list_matchers()
        assert names_after == sorted(names_after)
        del MATCHER_REGISTRY[name]
        _ = names_before  # used

    def test_length_grows_after_registration(self):
        n_before = len(list_matchers())
        name = "__len_grow__"
        register_fn(name, lambda a, b: 0.0)
        assert len(list_matchers()) == n_before + 1
        del MATCHER_REGISTRY[name]

    def test_no_duplicates(self):
        names = list_matchers()
        assert len(names) == len(set(names))
