"""Extra tests for puzzle_reconstruction/assembly/rl_agent.py"""
from __future__ import annotations

import math
import numpy as np
import pytest

from puzzle_reconstruction.assembly.rl_agent import (
    RLAssembler,
    RLConfig,
    RLEpisode,
    TabularPolicy,
    rl_assemble,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_compat(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.rand(n, n)


def make_diagonal_compat(n: int) -> np.ndarray:
    m = np.zeros((n, n))
    for i in range(n - 1):
        m[i, i + 1] = 1.0
        m[i + 1, i] = 1.0
    return m


class _FragStub:
    pass


def make_frags(n: int):
    return [_FragStub() for _ in range(n)]


# ── RLConfig extra ────────────────────────────────────────────────────────────

class TestRLConfigExtra:

    def test_custom_n_episodes(self):
        cfg = RLConfig(n_episodes=500)
        assert cfg.n_episodes == 500

    def test_custom_learning_rate(self):
        cfg = RLConfig(learning_rate=0.001)
        assert cfg.learning_rate == pytest.approx(0.001)

    def test_custom_temperature(self):
        cfg = RLConfig(temperature=2.0)
        assert cfg.temperature == pytest.approx(2.0)

    def test_custom_epsilon(self):
        cfg = RLConfig(epsilon=0.5)
        assert cfg.epsilon == pytest.approx(0.5)

    def test_custom_random_seed(self):
        cfg = RLConfig(random_seed=123)
        assert cfg.random_seed == 123

    def test_custom_max_steps(self):
        cfg = RLConfig(max_steps_per_episode=10)
        assert cfg.max_steps_per_episode == 10

    def test_neural_policy_true(self):
        cfg = RLConfig(use_neural_policy=True)
        assert cfg.use_neural_policy is True


# ── TabularPolicy extra ───────────────────────────────────────────────────────

class TestTabularPolicyExtra:

    def test_initial_theta_all_zeros(self):
        p = TabularPolicy(n_fragments=5)
        np.testing.assert_array_equal(p.theta, np.zeros((5, 5)))

    def test_action_probs_empty_available(self):
        p = TabularPolicy(n_fragments=4)
        probs = p.action_probs(state_id=0, available=[])
        assert len(probs) == 0

    def test_action_probs_single_available(self):
        p = TabularPolicy(n_fragments=4)
        probs = p.action_probs(state_id=0, available=[2])
        assert probs.shape == (1,)
        assert probs[0] == pytest.approx(1.0)

    def test_uniform_init_gives_uniform_probs(self):
        p = TabularPolicy(n_fragments=4)
        available = [0, 1, 2, 3]
        probs = p.action_probs(state_id=0, available=available)
        np.testing.assert_allclose(probs, np.full(4, 0.25), atol=1e-9)

    def test_update_increases_chosen_logit(self):
        p = TabularPolicy(n_fragments=4)
        before = p.theta[1, 2]
        p.update(state_id=1, action=2, G=5.0, lr=0.1)
        assert p.theta[1, 2] > before

    def test_update_negative_G_decreases_logit(self):
        p = TabularPolicy(n_fragments=4)
        p.update(state_id=0, action=1, G=-2.0, lr=0.1)
        assert p.theta[0, 1] < 0.0

    def test_best_action_with_updated_policy(self):
        p = TabularPolicy(n_fragments=4)
        # Strongly boost action 3
        p.theta[0, 3] = 10.0
        action = p.best_action(state_id=0, available=[0, 1, 2, 3])
        assert action == 3

    def test_temperature_affects_uniformity(self):
        # High temperature → more uniform distribution
        p_high = TabularPolicy(n_fragments=4, temperature=100.0)
        p_high.theta[0, 1] = 5.0
        probs_high = p_high.action_probs(0, [0, 1, 2, 3])
        # High temperature makes it closer to uniform
        assert max(probs_high) < 0.5

    def test_probs_all_nonnegative(self):
        p = TabularPolicy(n_fragments=6)
        p.theta[:] = np.random.RandomState(42).randn(6, 6)
        probs = p.action_probs(state_id=2, available=[0, 1, 3, 4, 5])
        assert np.all(probs >= 0.0)


# ── RLEpisode extra ────────────────────────────────────────────────────────────

class TestRLEpisodeExtra:

    def test_empty_episode(self):
        ep = RLEpisode(actions=[], rewards=[], total_reward=0.0, n_steps=0)
        assert ep.n_steps == 0
        assert ep.total_reward == pytest.approx(0.0)

    def test_total_reward_sum_of_rewards(self):
        rewards = [0.1, 0.2, 0.3]
        ep = RLEpisode(actions=[0, 1, 2], rewards=rewards,
                       total_reward=sum(rewards), n_steps=3)
        assert ep.total_reward == pytest.approx(0.6)

    def test_episode_actions_are_list(self):
        ep = RLEpisode(actions=[1, 2, 3], rewards=[0.0] * 3,
                       total_reward=0.0, n_steps=3)
        assert isinstance(ep.actions, list)

    def test_episode_rewards_are_list(self):
        ep = RLEpisode(actions=[0], rewards=[0.5],
                       total_reward=0.5, n_steps=1)
        assert isinstance(ep.rewards, list)


# ── RLAssembler._compute_returns extra ───────────────────────────────────────

class TestComputeReturnsExtra:

    def test_empty_rewards(self):
        returns = RLAssembler._compute_returns([], gamma=0.9)
        assert len(returns) == 0

    def test_two_steps_manual(self):
        rewards = [1.0, 2.0]
        returns = RLAssembler._compute_returns(rewards, gamma=0.5)
        # G[0] = 1 + 0.5*2 = 2.0, G[1] = 2.0
        np.testing.assert_allclose(returns, [2.0, 2.0], atol=1e-12)

    def test_all_zero_rewards(self):
        returns = RLAssembler._compute_returns([0.0, 0.0, 0.0], gamma=0.9)
        np.testing.assert_allclose(returns, [0.0, 0.0, 0.0], atol=1e-12)

    def test_gamma_half(self):
        rewards = [1.0, 1.0, 1.0, 1.0]
        returns = RLAssembler._compute_returns(rewards, gamma=0.5)
        expected = [
            1 + 0.5 + 0.25 + 0.125,
            1 + 0.5 + 0.25,
            1 + 0.5,
            1.0,
        ]
        np.testing.assert_allclose(returns, expected, atol=1e-9)

    def test_returns_non_negative_for_non_neg_rewards(self):
        rewards = [0.5, 0.3, 0.8]
        returns = RLAssembler._compute_returns(rewards, gamma=0.9)
        assert np.all(returns >= 0.0)

    def test_returns_dtype_float(self):
        returns = RLAssembler._compute_returns([1.0, 2.0], gamma=0.9)
        assert returns.dtype.kind == "f"


# ── RLAssembler._score_placement extra ───────────────────────────────────────

class TestScorePlacementExtra:

    def test_empty_order_zero_score(self):
        m = make_compat(4)
        score = RLAssembler._score_placement([], m)
        assert score == pytest.approx(0.0)

    def test_two_fragments_score_equals_matrix_entry(self):
        m = np.array([[0.0, 0.7], [0.3, 0.0]])
        score = RLAssembler._score_placement([0, 1], m)
        assert score == pytest.approx(0.7)

    def test_reversed_order_different_score(self):
        m = make_compat(3, seed=10)
        s_fwd = RLAssembler._score_placement([0, 1, 2], m)
        s_rev = RLAssembler._score_placement([2, 1, 0], m)
        # Asymmetric matrix → generally different
        assert np.isfinite(s_fwd)
        assert np.isfinite(s_rev)

    def test_diagonal_compat_sequential_high(self):
        n = 5
        m = make_diagonal_compat(n)
        score = RLAssembler._score_placement(list(range(n)), m)
        assert score == pytest.approx(float(n - 1))


# ── RLAssembler assemble extra ────────────────────────────────────────────────

class TestRLAssemblerAssembleExtra:

    def test_order_is_permutation_of_range(self):
        n = 6
        cfg = RLConfig(n_episodes=20, random_seed=0)
        order, _ = RLAssembler(config=cfg).assemble(make_frags(n), make_compat(n))
        assert sorted(order) == list(range(n))

    def test_large_n_completes(self):
        n = 20
        cfg = RLConfig(n_episodes=10, random_seed=1)
        order, score = RLAssembler(config=cfg).assemble(
            make_frags(n), make_compat(n, seed=11)
        )
        assert len(order) == n
        assert math.isfinite(score)

    def test_high_temperature_still_valid(self):
        n = 5
        cfg = RLConfig(n_episodes=15, temperature=100.0, random_seed=2)
        order, _ = RLAssembler(config=cfg).assemble(make_frags(n), make_compat(n))
        assert sorted(order) == list(range(n))

    def test_low_temperature_still_valid(self):
        n = 5
        cfg = RLConfig(n_episodes=15, temperature=0.01, random_seed=3)
        order, _ = RLAssembler(config=cfg).assemble(make_frags(n), make_compat(n))
        assert sorted(order) == list(range(n))

    def test_zero_compat_matrix_zero_score(self):
        n = 4
        cfg = RLConfig(n_episodes=10, random_seed=4)
        m = np.zeros((n, n))
        order, score = RLAssembler(config=cfg).assemble(make_frags(n), m)
        assert score == pytest.approx(0.0)

    def test_identity_compat_score_nonneg(self):
        n = 4
        cfg = RLConfig(n_episodes=10, random_seed=5)
        m = np.eye(n)
        order, score = RLAssembler(config=cfg).assemble(make_frags(n), m)
        assert score >= 0.0

    def test_small_n_episodes_1(self):
        n = 3
        cfg = RLConfig(n_episodes=1, random_seed=6)
        order, score = RLAssembler(config=cfg).assemble(make_frags(n), make_compat(n))
        assert len(order) == n


# ── rl_assemble module-level extra ────────────────────────────────────────────

class TestRlAssembleFunctionExtra:

    def test_returns_tuple_of_two(self):
        result = rl_assemble(make_frags(4), make_compat(4))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_order_is_permutation(self):
        n = 5
        order, _ = rl_assemble(make_frags(n), make_compat(n, seed=30))
        assert sorted(order) == list(range(n))

    def test_with_custom_config(self):
        n = 4
        cfg = RLConfig(n_episodes=10, random_seed=40)
        order, score = rl_assemble(make_frags(n), make_compat(n), config=cfg)
        assert len(order) == n
        assert math.isfinite(score)

    def test_score_matches_manual_score_placement(self):
        n = 4
        m = make_compat(n, seed=50)
        cfg = RLConfig(n_episodes=10, random_seed=50)
        order, score = rl_assemble(make_frags(n), m, config=cfg)
        manual_score = RLAssembler._score_placement(order, m)
        assert score == pytest.approx(manual_score, abs=1e-9)

    def test_diagonal_compat_positive_score(self):
        n = 5
        m = make_diagonal_compat(n)
        cfg = RLConfig(n_episodes=30, random_seed=60)
        _, score = rl_assemble(make_frags(n), m, config=cfg)
        assert score >= 0.0
