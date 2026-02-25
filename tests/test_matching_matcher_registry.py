"""Тесты для puzzle_reconstruction.matching.matcher_registry."""
import pytest
import numpy as np
from puzzle_reconstruction.matching.matcher_registry import (
    MATCHER_REGISTRY,
    compute_scores,
    get_matcher,
    list_matchers,
    register,
    register_fn,
    weighted_combine,
)
from puzzle_reconstruction.models import EdgeSignature, EdgeSide


def _edge(edge_id: int = 0) -> EdgeSignature:
    """Минимальная EdgeSignature для тестов."""
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.RIGHT,
        virtual_curve=np.zeros((4, 2)),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=10.0,
    )


# ─── TestMATCHER_REGISTRY ──────────────────────────────────────────────────────

class TestMatcherRegistry:
    def test_is_dict(self):
        assert isinstance(MATCHER_REGISTRY, dict)

    def test_not_empty_after_import(self):
        # _register_defaults() запускается при импорте → хотя бы 1 матчер
        assert len(MATCHER_REGISTRY) > 0

    def test_keys_are_strings(self):
        for k in MATCHER_REGISTRY:
            assert isinstance(k, str)

    def test_values_are_callable(self):
        for fn in MATCHER_REGISTRY.values():
            assert callable(fn)

    def test_fd_matcher_registered(self):
        # 'fd' всегда регистрируется без try/except
        assert "fd" in MATCHER_REGISTRY

    def test_text_matcher_registered(self):
        assert "text" in MATCHER_REGISTRY


# ─── TestRegister ─────────────────────────────────────────────────────────────

class TestRegister:
    def test_decorator_adds_to_registry(self):
        name = "__test_register_decorator__"
        @register(name)
        def my_fn(e_i, e_j):
            return 0.5
        assert name in MATCHER_REGISTRY
        assert MATCHER_REGISTRY[name] is my_fn
        # Cleanup
        del MATCHER_REGISTRY[name]

    def test_decorator_returns_function(self):
        name = "__test_register_returns__"
        @register(name)
        def my_fn(e_i, e_j):
            return 0.7
        assert callable(my_fn)
        del MATCHER_REGISTRY[name]


# ─── TestRegisterFn ───────────────────────────────────────────────────────────

class TestRegisterFn:
    def test_registers_function(self):
        name = "__test_register_fn__"
        fn = lambda e_i, e_j: 0.3
        register_fn(name, fn)
        assert name in MATCHER_REGISTRY
        del MATCHER_REGISTRY[name]

    def test_overwrites_existing(self):
        name = "__test_overwrite__"
        fn1 = lambda e_i, e_j: 0.1
        fn2 = lambda e_i, e_j: 0.9
        register_fn(name, fn1)
        register_fn(name, fn2)
        assert MATCHER_REGISTRY[name] is fn2
        del MATCHER_REGISTRY[name]


# ─── TestGetMatcher ───────────────────────────────────────────────────────────

class TestGetMatcher:
    def test_known_name_returns_callable(self):
        fn = get_matcher("fd")
        assert fn is not None
        assert callable(fn)

    def test_unknown_name_returns_none(self):
        assert get_matcher("__no_such_matcher__") is None

    def test_empty_name_returns_none(self):
        assert get_matcher("") is None


# ─── TestListMatchers ─────────────────────────────────────────────────────────

class TestListMatchers:
    def test_returns_list(self):
        result = list_matchers()
        assert isinstance(result, list)

    def test_not_empty(self):
        assert len(list_matchers()) > 0

    def test_sorted(self):
        names = list_matchers()
        assert names == sorted(names)

    def test_fd_in_list(self):
        assert "fd" in list_matchers()

    def test_all_strings(self):
        for n in list_matchers():
            assert isinstance(n, str)


# ─── TestComputeScores ────────────────────────────────────────────────────────

class TestComputeScores:
    def test_returns_dict(self):
        e1 = _edge(0)
        e2 = _edge(10)
        result = compute_scores(e1, e2, ["fd"])
        assert isinstance(result, dict)

    def test_fd_score_in_range(self):
        e1 = _edge(0)
        e2 = _edge(10)
        result = compute_scores(e1, e2, ["fd"])
        assert 0.0 <= result["fd"] <= 1.0

    def test_unknown_matcher_returns_zero(self):
        e1 = _edge(0)
        e2 = _edge(10)
        result = compute_scores(e1, e2, ["__nonexistent__"])
        assert result["__nonexistent__"] == pytest.approx(0.0)

    def test_empty_matchers(self):
        e1 = _edge(0)
        e2 = _edge(10)
        result = compute_scores(e1, e2, [])
        assert result == {}

    def test_text_matcher_returns_zero(self):
        # The default 'text' matcher always returns 0.0
        e1 = _edge(0)
        e2 = _edge(10)
        result = compute_scores(e1, e2, ["text"])
        assert result["text"] == pytest.approx(0.0)

    def test_fd_same_fd_is_high(self):
        # Same fd → diff = 0 → score = 1/(1+0) = 1.0
        e1 = _edge(0)
        e2 = _edge(10)
        result = compute_scores(e1, e2, ["fd"])
        # Both have fd=1.5, diff=0 → score=1.0
        assert result["fd"] == pytest.approx(1.0)


# ─── TestWeightedCombine ──────────────────────────────────────────────────────

class TestWeightedCombine:
    def test_equal_weights(self):
        scores = {"a": 0.6, "b": 0.8}
        weights = {"a": 1.0, "b": 1.0}
        result = weighted_combine(scores, weights)
        assert result == pytest.approx(0.7)

    def test_zero_total_weight_returns_zero(self):
        scores = {"a": 0.5}
        weights = {"a": 0.0}
        result = weighted_combine(scores, weights)
        assert result == pytest.approx(0.0)

    def test_single_score(self):
        scores = {"a": 0.8}
        weights = {"a": 1.0}
        result = weighted_combine(scores, weights)
        assert result == pytest.approx(0.8)

    def test_result_in_range(self):
        scores = {"a": 0.3, "b": 0.7, "c": 0.5}
        weights = {"a": 0.2, "b": 0.5, "c": 0.3}
        result = weighted_combine(scores, weights)
        assert 0.0 <= result <= 1.0

    def test_empty_scores(self):
        result = weighted_combine({}, {"a": 1.0})
        assert result == pytest.approx(0.0)

    def test_missing_weight_treated_as_zero(self):
        scores = {"a": 0.9}
        weights = {"b": 1.0}  # no weight for 'a'
        result = weighted_combine(scores, weights)
        assert result == pytest.approx(0.0)
