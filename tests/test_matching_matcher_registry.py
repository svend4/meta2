"""Тесты для puzzle_reconstruction/matching/matcher_registry.py."""
import pytest
import numpy as np

from puzzle_reconstruction.matching.matcher_registry import (
    MATCHER_REGISTRY,
    register,
    register_fn,
    get_matcher,
    list_matchers,
    compute_scores,
    weighted_combine,
)
from puzzle_reconstruction.models import EdgeSignature, EdgeSide


def _make_edge(eid=0, fid=0):
    curve = np.column_stack([
        np.linspace(0, 10, 20),
        np.sin(np.linspace(0, np.pi, 20))
    ])
    e = EdgeSignature(
        edge_id=eid,
        side=EdgeSide.RIGHT,
        virtual_curve=curve,
        fd=1.2,
        css_vec=np.ones(64) / 8.0,
        ifs_coeffs=np.zeros(12),
        length=10.0,
    )
    return e


class TestRegistry:
    def test_registry_is_dict(self):
        assert isinstance(MATCHER_REGISTRY, dict)

    def test_has_built_in_matchers(self):
        assert "fd" in list_matchers()

    def test_list_matchers_sorted(self):
        matchers = list_matchers()
        assert matchers == sorted(matchers)

    def test_get_matcher_returns_callable(self):
        assert callable(get_matcher("fd"))

    def test_get_missing_matcher_returns_none(self):
        assert get_matcher("__nonexistent_matcher__") is None


class TestRegister:
    def test_register_decorator(self):
        @register("__test_decorator_matcher__")
        def my_fn(e_i, e_j):
            return 0.42
        assert "__test_decorator_matcher__" in MATCHER_REGISTRY
        del MATCHER_REGISTRY["__test_decorator_matcher__"]

    def test_register_fn(self):
        register_fn("__test_fn_matcher__", lambda e_i, e_j: 0.5)
        assert "__test_fn_matcher__" in MATCHER_REGISTRY
        del MATCHER_REGISTRY["__test_fn_matcher__"]

    def test_overwrite_existing(self):
        register_fn("__test_overwrite__", lambda e_i, e_j: 0.1)
        register_fn("__test_overwrite__", lambda e_i, e_j: 0.9)
        e = _make_edge()
        assert MATCHER_REGISTRY["__test_overwrite__"](e, e) == pytest.approx(0.9)
        del MATCHER_REGISTRY["__test_overwrite__"]


class TestComputeScores:
    def test_returns_dict(self):
        e = _make_edge()
        assert isinstance(compute_scores(e, e, ["fd"]), dict)

    def test_fd_matcher_runs(self):
        e = _make_edge()
        scores = compute_scores(e, e, ["fd"])
        assert "fd" in scores
        assert 0.0 <= scores["fd"] <= 1.0

    def test_identical_edges_fd_score_is_one(self):
        e = _make_edge()
        e.fd = 1.5
        assert compute_scores(e, e, ["fd"])["fd"] == pytest.approx(1.0)

    def test_missing_matcher_returns_zero(self):
        e = _make_edge()
        assert compute_scores(e, e, ["__nonexistent__"])["__nonexistent__"] == 0.0

    def test_multiple_matchers(self):
        e = _make_edge()
        matchers = [m for m in list_matchers() if m in ("fd", "text")]
        assert len(compute_scores(e, e, matchers)) == len(matchers)


class TestWeightedCombine:
    def test_single_weight(self):
        assert weighted_combine({"fd": 0.8}, {"fd": 1.0}) == pytest.approx(0.8)

    def test_two_equal_weights(self):
        result = weighted_combine({"fd": 0.6, "text": 0.4}, {"fd": 1.0, "text": 1.0})
        assert result == pytest.approx(0.5)

    def test_zero_weights_returns_zero(self):
        assert weighted_combine({"fd": 0.8}, {"fd": 0.0}) == 0.0

    def test_missing_key_in_weights(self):
        assert weighted_combine({"fd": 0.8}, {"text": 1.0}) == 0.0

    def test_result_in_range(self):
        result = weighted_combine({"fd": 0.7, "text": 0.3}, {"fd": 0.6, "text": 0.4})
        assert 0.0 <= result <= 1.0


class TestBuiltInMatchers:
    def test_fd_same_edge_returns_one(self):
        fn = get_matcher("fd")
        e = _make_edge()
        e.fd = 1.5
        assert fn(e, e) == pytest.approx(1.0)

    def test_fd_different_values_returns_less(self):
        fn = get_matcher("fd")
        e1, e2 = _make_edge(eid=0), _make_edge(eid=1)
        e1.fd, e2.fd = 1.0, 2.0
        score = fn(e1, e2)
        assert 0.0 <= score < 1.0

    def test_text_matcher_returns_zero(self):
        fn = get_matcher("text")
        e = _make_edge()
        assert fn(e, e) == 0.0

    def test_css_matcher_runs_if_available(self):
        if "css" not in MATCHER_REGISTRY:
            pytest.skip("css matcher not registered")
        e = _make_edge()
        score = get_matcher("css")(e, e)
        assert 0.0 <= score <= 1.0
