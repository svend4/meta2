"""
Tests for puzzle_reconstruction.assembly.rl_agent

Covers:
- RLConfig default values and field types
- RLEpisode construction and fields
- TabularPolicy shape, probabilities, updates, best_action
- RLAssembler.assemble with 2 / 4 / 9 fragments
- Uniqueness and validity of placement orders
- train() API and episode structure
- _compute_returns edge cases
- _score_placement correctness
- Determinism with same seed; variation with different seeds
- Module-level rl_assemble()
- Edge-case configs (small n_episodes, epsilon=1.0, epsilon=0.0)
"""
import math
import numpy as np
import pytest

from puzzle_reconstruction.assembly.rl_agent import (
    RLConfig,
    RLEpisode,
    TabularPolicy,
    RLAssembler,
    rl_assemble,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_compat(n: int, seed: int = 0) -> np.ndarray:
    """Return a random non-negative (n, n) compatibility matrix."""
    rng = np.random.RandomState(seed)
    m = rng.rand(n, n)
    return m


def make_diagonal_compat(n: int) -> np.ndarray:
    """Return an identity-like matrix where off-diagonal entries are high
    for adjacent indices and zero otherwise."""
    m = np.zeros((n, n))
    for i in range(n - 1):
        m[i, i + 1] = 1.0
        m[i + 1, i] = 1.0
    return m


# Minimal stub for fragments — only len() is used by RLAssembler
class _FragStub:
    pass


def make_fragments(n: int):
    return [_FragStub() for _ in range(n)]


# ---------------------------------------------------------------------------
# 1-8: RLConfig default values
# ---------------------------------------------------------------------------

class TestRLConfigDefaults:
    def test_n_episodes_default(self):
        cfg = RLConfig()
        assert cfg.n_episodes == 200

    def test_learning_rate_default(self):
        cfg = RLConfig()
        assert cfg.learning_rate == pytest.approx(0.01)

    def test_gamma_default(self):
        cfg = RLConfig()
        assert cfg.gamma == pytest.approx(0.95)

    def test_temperature_default(self):
        cfg = RLConfig()
        assert cfg.temperature == pytest.approx(1.0)

    def test_epsilon_default(self):
        cfg = RLConfig()
        assert cfg.epsilon == pytest.approx(0.1)

    def test_use_neural_policy_default(self):
        cfg = RLConfig()
        assert cfg.use_neural_policy is False

    def test_random_seed_default(self):
        cfg = RLConfig()
        assert cfg.random_seed == 42

    def test_max_steps_per_episode_default(self):
        cfg = RLConfig()
        assert cfg.max_steps_per_episode == 50


# ---------------------------------------------------------------------------
# 9-12: RLConfig field types
# ---------------------------------------------------------------------------

class TestRLConfigTypes:
    def test_n_episodes_is_int(self):
        assert isinstance(RLConfig().n_episodes, int)

    def test_learning_rate_is_float(self):
        assert isinstance(RLConfig().learning_rate, float)

    def test_gamma_is_float(self):
        assert isinstance(RLConfig().gamma, float)

    def test_epsilon_is_float(self):
        assert isinstance(RLConfig().epsilon, float)


# ---------------------------------------------------------------------------
# 13-16: RLEpisode construction
# ---------------------------------------------------------------------------

class TestRLEpisode:
    def test_construction(self):
        ep = RLEpisode(actions=[0, 1, 2], rewards=[0.1, 0.2, 0.3],
                       total_reward=0.6, n_steps=3)
        assert ep.actions == [0, 1, 2]
        assert ep.n_steps == 3

    def test_total_reward_is_float(self):
        ep = RLEpisode(actions=[0], rewards=[1.0], total_reward=1.0, n_steps=1)
        assert isinstance(ep.total_reward, float)

    def test_rewards_stored(self):
        rewards = [0.5, 0.25]
        ep = RLEpisode(actions=[0, 1], rewards=rewards,
                       total_reward=0.75, n_steps=2)
        assert ep.rewards == rewards

    def test_actions_stored(self):
        ep = RLEpisode(actions=[3, 1, 0, 2], rewards=[0.0] * 4,
                       total_reward=0.0, n_steps=4)
        assert ep.actions == [3, 1, 0, 2]


# ---------------------------------------------------------------------------
# 17-22: TabularPolicy
# ---------------------------------------------------------------------------

class TestTabularPolicy:
    def test_theta_shape(self):
        p = TabularPolicy(n_fragments=5)
        assert p.theta.shape == (5, 5)

    def test_action_probs_sums_to_one(self):
        p = TabularPolicy(n_fragments=4)
        probs = p.action_probs(state_id=0, available=[1, 2, 3])
        assert probs.sum() == pytest.approx(1.0, abs=1e-9)

    def test_action_probs_restricted_available(self):
        p = TabularPolicy(n_fragments=6)
        available = [0, 2, 4]
        probs = p.action_probs(state_id=1, available=available)
        assert len(probs) == len(available)
        assert probs.sum() == pytest.approx(1.0, abs=1e-9)

    def test_action_probs_nonnegative(self):
        p = TabularPolicy(n_fragments=5)
        probs = p.action_probs(state_id=0, available=[0, 1, 2, 3, 4])
        assert np.all(probs >= 0.0)

    def test_best_action_valid_index(self):
        p = TabularPolicy(n_fragments=4)
        available = [1, 3]
        action = p.best_action(state_id=0, available=available)
        assert action in available

    def test_update_changes_theta(self):
        p = TabularPolicy(n_fragments=3)
        theta_before = p.theta.copy()
        p.update(state_id=0, action=1, G=1.0, lr=0.1)
        assert not np.array_equal(p.theta, theta_before)


# ---------------------------------------------------------------------------
# 23-29: RLAssembler.assemble
# ---------------------------------------------------------------------------

class TestRLAssemblerAssemble:
    def test_two_fragments_returns_list_of_two(self):
        n = 2
        cfg = RLConfig(n_episodes=10, random_seed=0)
        asm = RLAssembler(config=cfg)
        order, score = asm.assemble(make_fragments(n), make_compat(n))
        assert len(order) == n

    def test_four_fragments_returns_list_of_four(self):
        n = 4
        cfg = RLConfig(n_episodes=20, random_seed=1)
        asm = RLAssembler(config=cfg)
        order, score = asm.assemble(make_fragments(n), make_compat(n))
        assert len(order) == n

    def test_nine_fragments_returns_list_of_nine(self):
        n = 9
        cfg = RLConfig(n_episodes=30, random_seed=2)
        asm = RLAssembler(config=cfg)
        order, score = asm.assemble(make_fragments(n), make_compat(n))
        assert len(order) == n

    def test_all_fragments_appear_exactly_once(self):
        n = 6
        cfg = RLConfig(n_episodes=15, random_seed=3)
        asm = RLAssembler(config=cfg)
        order, _ = asm.assemble(make_fragments(n), make_compat(n))
        assert sorted(order) == list(range(n))

    def test_score_is_float(self):
        n = 4
        cfg = RLConfig(n_episodes=10, random_seed=4)
        asm = RLAssembler(config=cfg)
        _, score = asm.assemble(make_fragments(n), make_compat(n))
        assert isinstance(score, float)

    def test_score_is_finite(self):
        n = 5
        cfg = RLConfig(n_episodes=10, random_seed=5)
        asm = RLAssembler(config=cfg)
        _, score = asm.assemble(make_fragments(n), make_compat(n))
        assert math.isfinite(score)

    def test_score_not_nan(self):
        n = 3
        cfg = RLConfig(n_episodes=5, random_seed=6)
        asm = RLAssembler(config=cfg)
        _, score = asm.assemble(make_fragments(n), make_compat(n))
        assert not math.isnan(score)


# ---------------------------------------------------------------------------
# 30-33: train() API
# ---------------------------------------------------------------------------

class TestRLAssemblerTrain:
    def test_train_returns_list_of_episodes(self):
        n = 4
        cfg = RLConfig(n_episodes=5, random_seed=10)
        asm = RLAssembler(config=cfg)
        episodes = asm.train(make_compat(n), n)
        assert isinstance(episodes, list)
        assert all(isinstance(ep, RLEpisode) for ep in episodes)

    def test_train_episode_count(self):
        n = 3
        cfg = RLConfig(n_episodes=7, random_seed=11)
        asm = RLAssembler(config=cfg)
        episodes = asm.train(make_compat(n), n)
        assert len(episodes) == 7

    def test_train_episodes_n_steps_positive(self):
        n = 4
        cfg = RLConfig(n_episodes=5, random_seed=12)
        asm = RLAssembler(config=cfg)
        episodes = asm.train(make_compat(n), n)
        assert all(ep.n_steps > 0 for ep in episodes)

    def test_train_total_reward_is_float(self):
        n = 3
        cfg = RLConfig(n_episodes=5, random_seed=13)
        asm = RLAssembler(config=cfg)
        episodes = asm.train(make_compat(n), n)
        assert all(isinstance(ep.total_reward, float) for ep in episodes)

    def test_train_episode_actions_length(self):
        n = 5
        cfg = RLConfig(n_episodes=5, random_seed=14)
        asm = RLAssembler(config=cfg)
        episodes = asm.train(make_compat(n), n)
        for ep in episodes:
            assert ep.n_steps == len(ep.actions)


# ---------------------------------------------------------------------------
# 35-38: _compute_returns
# ---------------------------------------------------------------------------

class TestComputeReturns:
    def test_same_length_as_rewards(self):
        rewards = [1.0, 2.0, 3.0]
        returns = RLAssembler._compute_returns(rewards, gamma=0.9)
        assert len(returns) == len(rewards)

    def test_gamma_zero_equals_rewards(self):
        rewards = [1.0, 2.0, 3.0]
        returns = RLAssembler._compute_returns(rewards, gamma=0.0)
        np.testing.assert_allclose(returns, rewards)

    def test_gamma_one_cumulative_sums(self):
        rewards = [1.0, 2.0, 3.0]
        returns = RLAssembler._compute_returns(rewards, gamma=1.0)
        # G[0] = 1+2+3=6, G[1] = 2+3=5, G[2] = 3
        np.testing.assert_allclose(returns, [6.0, 5.0, 3.0])

    def test_single_step(self):
        returns = RLAssembler._compute_returns([5.0], gamma=0.9)
        assert returns[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 39-40: _score_placement
# ---------------------------------------------------------------------------

class TestScorePlacement:
    def test_perfect_diagonal_positive_score(self):
        n = 4
        m = make_diagonal_compat(n)
        # Sequential order 0,1,2,3 traverses all high edges
        score = RLAssembler._score_placement([0, 1, 2, 3], m)
        assert score > 0.0

    def test_single_fragment_score_zero(self):
        m = make_compat(3)
        score = RLAssembler._score_placement([1], m)
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 41-42: Determinism and seed variation
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_deterministic(self):
        n = 5
        m = make_compat(n, seed=99)
        cfg1 = RLConfig(n_episodes=20, random_seed=7)
        cfg2 = RLConfig(n_episodes=20, random_seed=7)
        order1, score1 = RLAssembler(config=cfg1).assemble(make_fragments(n), m)
        order2, score2 = RLAssembler(config=cfg2).assemble(make_fragments(n), m)
        assert order1 == order2
        assert score1 == pytest.approx(score2)

    def test_different_seeds_may_differ(self):
        """With different seeds, results should not be guaranteed identical.
        We run many fragments / episodes to increase the chance of divergence."""
        n = 8
        m = make_compat(n, seed=55)
        cfg_a = RLConfig(n_episodes=50, random_seed=0)
        cfg_b = RLConfig(n_episodes=50, random_seed=999)
        order_a, _ = RLAssembler(config=cfg_a).assemble(make_fragments(n), m)
        order_b, _ = RLAssembler(config=cfg_b).assemble(make_fragments(n), m)
        # Both must still be valid permutations
        assert sorted(order_a) == list(range(n))
        assert sorted(order_b) == list(range(n))


# ---------------------------------------------------------------------------
# 43-46: rl_assemble module-level function
# ---------------------------------------------------------------------------

class TestRlAssembleFunction:
    def test_rl_assemble_works(self):
        n = 4
        order, score = rl_assemble(make_fragments(n), make_compat(n, seed=20))
        assert len(order) == n

    def test_rl_assemble_returns_tuple(self):
        n = 3
        result = rl_assemble(make_fragments(n), make_compat(n, seed=21))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_rl_assemble_order_is_list(self):
        n = 3
        order, score = rl_assemble(make_fragments(n), make_compat(n, seed=22))
        assert isinstance(order, list)

    def test_rl_assemble_score_is_float(self):
        n = 3
        _, score = rl_assemble(make_fragments(n), make_compat(n, seed=23))
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# 47-50: Edge-case configs
# ---------------------------------------------------------------------------

class TestEdgeCaseConfigs:
    def test_small_n_episodes_completes(self):
        n = 4
        cfg = RLConfig(n_episodes=5, random_seed=30)
        order, score = RLAssembler(config=cfg).assemble(
            make_fragments(n), make_compat(n)
        )
        assert len(order) == n

    def test_epsilon_one_random_still_valid(self):
        n = 5
        cfg = RLConfig(n_episodes=10, epsilon=1.0, random_seed=31)
        order, _ = RLAssembler(config=cfg).assemble(
            make_fragments(n), make_compat(n)
        )
        assert sorted(order) == list(range(n))

    def test_epsilon_zero_greedy_still_valid(self):
        n = 5
        cfg = RLConfig(n_episodes=10, epsilon=0.0, random_seed=32)
        order, _ = RLAssembler(config=cfg).assemble(
            make_fragments(n), make_compat(n)
        )
        assert sorted(order) == list(range(n))

    def test_sixteen_fragments(self):
        n = 16
        cfg = RLConfig(n_episodes=10, random_seed=33)
        order, score = RLAssembler(config=cfg).assemble(
            make_fragments(n), make_compat(n, seed=77)
        )
        assert len(order) == n
        assert sorted(order) == list(range(n))
        assert math.isfinite(score)
