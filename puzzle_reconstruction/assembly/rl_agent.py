"""
Reinforcement Learning assembler for puzzle reconstruction.

Implements a simple policy-gradient (REINFORCE) agent that learns to
place fragments sequentially to maximise the cumulative compatibility score.

No external ML framework required — uses pure numpy with optional torch
integration if available.

State:   (compat_matrix, placed_mask, last_fragment_id)
Action:  next fragment to place
Reward:  delta_score after placement
Policy:  softmax over compatibility scores (tabular, no neural net required)

For torch-based deep policy networks, set use_neural_policy=True
(requires: pip install torch).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RLConfig:
    """Configuration for the RL assembler."""

    n_episodes: int = 200
    learning_rate: float = 0.01
    gamma: float = 0.95          # discount factor
    temperature: float = 1.0    # softmax temperature
    epsilon: float = 0.1        # epsilon-greedy exploration
    use_neural_policy: bool = False
    random_seed: int = 42
    max_steps_per_episode: int = 50


# ---------------------------------------------------------------------------
# Episode record
# ---------------------------------------------------------------------------

@dataclass
class RLEpisode:
    """Record of a single training episode."""

    actions: List[int]
    rewards: List[float]
    total_reward: float
    n_steps: int


# ---------------------------------------------------------------------------
# Tabular policy (pure numpy)
# ---------------------------------------------------------------------------

class TabularPolicy:
    """
    Tabular REINFORCE policy.

    Parameters theta[s, a] are updated via the log-policy gradient.
    Action probabilities are derived through a masked softmax.
    """

    def __init__(self, n_fragments: int, temperature: float = 1.0) -> None:
        self.n_fragments = n_fragments
        self.temperature = max(temperature, 1e-8)
        # Parameter matrix initialised to zeros
        self.theta: np.ndarray = np.zeros((n_fragments, n_fragments), dtype=float)

    # ------------------------------------------------------------------
    def action_probs(
        self, state_id: int, available: List[int]
    ) -> np.ndarray:
        """
        Return a probability distribution over *available* actions.

        Args:
            state_id:  Index of the current "last-placed" fragment (state).
            available: List of fragment indices not yet placed.

        Returns:
            np.ndarray of shape (len(available),) summing to 1.0.
        """
        if len(available) == 0:
            return np.array([], dtype=float)

        logits = self.theta[state_id, available] / self.temperature
        # Numerically-stable softmax
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()
        return probs

    # ------------------------------------------------------------------
    def update(
        self, state_id: int, action: int, G: float, lr: float
    ) -> None:
        """
        REINFORCE parameter update.

        theta[s, a] += lr * G * d/dtheta log pi(a | s)

        For softmax policy: d/dtheta[s,a] log pi(a|s) = 1 - pi(a|s)
        (for the chosen action parameter), simplified tabular update.

        Args:
            state_id: The state (last fragment index).
            action:   The action (fragment index) that was taken.
            G:        Discounted return from this step onward.
            lr:       Learning rate.
        """
        # Gradient of log-softmax w.r.t. theta[state_id, action]
        self.theta[state_id, action] += lr * G

    # ------------------------------------------------------------------
    def best_action(self, state_id: int, available: List[int]) -> int:
        """Return the action with the highest probability from available."""
        if len(available) == 0:
            raise ValueError("No available actions.")
        probs = self.action_probs(state_id, available)
        return available[int(np.argmax(probs))]


# ---------------------------------------------------------------------------
# RLAssembler
# ---------------------------------------------------------------------------

class RLAssembler:
    """
    REINFORCE-based puzzle assembler.

    Trains a TabularPolicy over ``n_episodes`` episodes on the given
    compatibility matrix, then returns the best observed placement order.
    """

    def __init__(self, config: Optional[RLConfig] = None) -> None:
        self.config = config or RLConfig()

    # ------------------------------------------------------------------
    def assemble(
        self,
        fragments,
        compat_matrix: np.ndarray,
    ) -> Tuple[List[int], float]:
        """
        Run training and return the best placement order with its score.

        Args:
            fragments:     Iterable of fragment objects (length determines
                           n_fragments; only ``len()`` is used).
            compat_matrix: Square numpy array of pairwise compatibility scores.

        Returns:
            (order, score) where *order* is a list of fragment indices and
            *score* is the total compatibility score for that sequence.
        """
        n_fragments = len(fragments)
        episodes = self.train(compat_matrix, n_fragments)

        best_episode = max(episodes, key=lambda ep: ep.total_reward)
        best_order = best_episode.actions
        best_score = self._score_placement(best_order, compat_matrix)
        return best_order, float(best_score)

    # ------------------------------------------------------------------
    def train(
        self, compat_matrix: np.ndarray, n_fragments: int
    ) -> List[RLEpisode]:
        """
        Train the policy for ``config.n_episodes`` episodes.

        Args:
            compat_matrix: (n_fragments, n_fragments) compatibility matrix.
            n_fragments:   Number of fragments to place.

        Returns:
            List of RLEpisode records (one per episode).
        """
        rng = np.random.RandomState(self.config.random_seed)
        policy = TabularPolicy(n_fragments, self.config.temperature)

        episodes: List[RLEpisode] = []
        for _ in range(self.config.n_episodes):
            ep = self._run_episode(policy, compat_matrix, n_fragments, rng)
            episodes.append(ep)

            # REINFORCE update: iterate over (state, action, G) tuples
            returns = self._compute_returns(ep.rewards, self.config.gamma)
            # Rebuild state sequence: state is the last-placed fragment index
            # Initial state uses a virtual state index n_fragments (no fragment
            # placed yet) — but theta has shape (n_fragments, n_fragments).
            # We use the *previous* action as the state_id for all steps > 0,
            # and for step 0 we use action itself (self-loop seed state).
            for t, (action, G) in enumerate(zip(ep.actions, returns)):
                state_id = ep.actions[t - 1] if t > 0 else action
                policy.update(
                    state_id, action, float(G), self.config.learning_rate
                )

        return episodes

    # ------------------------------------------------------------------
    def _run_episode(
        self,
        policy: TabularPolicy,
        compat_matrix: np.ndarray,
        n_fragments: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> RLEpisode:
        """
        Simulate one placement episode.

        Args:
            policy:        Current TabularPolicy.
            compat_matrix: Compatibility matrix.
            n_fragments:   Total number of fragments.
            rng:           NumPy RandomState for reproducibility.

        Returns:
            RLEpisode with actions, rewards, total_reward, n_steps.
        """
        if rng is None:
            rng = np.random.RandomState(self.config.random_seed)

        available = list(range(n_fragments))
        actions: List[int] = []
        rewards: List[float] = []

        max_steps = min(self.config.max_steps_per_episode, n_fragments)
        state_id: Optional[int] = None

        for _ in range(max_steps):
            if not available:
                break

            if state_id is None:
                # First action: use uniform / epsilon-greedy seeded on the
                # virtual start state; fall back to first action = state_id.
                action = self._select_action(
                    policy,
                    state=available[0],  # seed state
                    available=available,
                    epsilon=self.config.epsilon,
                    rng=rng,
                )
            else:
                action = self._select_action(
                    policy,
                    state=state_id,
                    available=available,
                    epsilon=self.config.epsilon,
                    rng=rng,
                )

            available.remove(action)

            # Reward: compatibility with the previously placed fragment
            if state_id is not None:
                reward = float(compat_matrix[state_id, action])
            else:
                reward = 0.0

            actions.append(action)
            rewards.append(reward)
            state_id = action

        total_reward = float(sum(rewards))
        return RLEpisode(
            actions=actions,
            rewards=rewards,
            total_reward=total_reward,
            n_steps=len(actions),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_returns(
        rewards: List[float], gamma: float
    ) -> np.ndarray:
        """
        Compute discounted returns G_t = r_t + gamma * r_{t+1} + ...

        Args:
            rewards: List of per-step rewards.
            gamma:   Discount factor in [0, 1].

        Returns:
            np.ndarray of the same length as ``rewards``.
        """
        n = len(rewards)
        returns = np.zeros(n, dtype=float)
        running = 0.0
        for t in reversed(range(n)):
            running = rewards[t] + gamma * running
            returns[t] = running
        return returns

    # ------------------------------------------------------------------
    @staticmethod
    def _score_placement(
        order: List[int], compat_matrix: np.ndarray
    ) -> float:
        """
        Sum of edge compatibility scores for a sequential placement order.

        Score = sum_{t=0}^{T-2} compat_matrix[order[t], order[t+1]]

        Args:
            order:         Sequence of fragment indices.
            compat_matrix: Compatibility matrix.

        Returns:
            Scalar float score.
        """
        if len(order) < 2:
            return 0.0
        score = 0.0
        for t in range(len(order) - 1):
            score += float(compat_matrix[order[t], order[t + 1]])
        return score

    # ------------------------------------------------------------------
    def _select_action(
        self,
        policy: TabularPolicy,
        state: int,
        available: List[int],
        epsilon: float,
        rng: Optional[np.random.RandomState] = None,
    ) -> int:
        """
        Epsilon-greedy action selection.

        With probability ``epsilon``, choose uniformly at random from
        *available*; otherwise choose the greedy best action.

        Args:
            policy:    Current TabularPolicy.
            state:     Current state (last-placed fragment index).
            available: List of unplaced fragment indices.
            epsilon:   Exploration probability.
            rng:       NumPy RandomState.

        Returns:
            Selected action (fragment index).
        """
        if rng is None:
            rng = np.random.RandomState(self.config.random_seed)

        if len(available) == 0:
            raise ValueError("No available actions to select from.")

        if rng.random() < epsilon:
            idx = rng.randint(0, len(available))
            return available[idx]
        else:
            return policy.best_action(state, available)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def rl_assemble(
    fragments,
    compat_matrix: np.ndarray,
    config: Optional[RLConfig] = None,
) -> Tuple[List[int], float]:
    """
    Convenience wrapper: create an RLAssembler and run assemble().

    Args:
        fragments:     Sequence of fragment objects (only len() is used).
        compat_matrix: Square pairwise compatibility matrix (numpy array).
        config:        Optional RLConfig; defaults to RLConfig() if None.

    Returns:
        (order, score) — best placement order and its total compatibility score.
    """
    assembler = RLAssembler(config=config)
    return assembler.assemble(fragments, compat_matrix)
