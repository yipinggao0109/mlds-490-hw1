"""Dynamic-programming methods for the shuttle-dispatch problem.

The timing convention used throughout the module is:

1. Observe the number of passengers currently waiting.
2. Choose whether to dispatch a shuttle.
3. Pay a dispatch cost, when applicable, and a holding cost for passengers left behind.
4. Observe the new arrivals and cap the next state at ``max_waiting``.

All algorithms minimize expected discounted cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ShuttleMDP:
    """Finite-state model of the shuttle-dispatch problem."""

    capacity: int = 15
    dispatch_cost: float = 100.0
    holding_cost: float = 2.0
    arrivals: tuple[int, ...] = (1, 2, 3, 4, 5)
    arrival_probabilities: tuple[float, ...] | None = None
    max_waiting: int = 200
    gamma: float = 0.95

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.dispatch_cost < 0 or self.holding_cost < 0:
            raise ValueError("costs must be nonnegative")
        if not self.arrivals or any(a < 0 for a in self.arrivals):
            raise ValueError("arrivals must be a nonempty sequence of nonnegative integers")
        if self.max_waiting < 0:
            raise ValueError("max_waiting must be nonnegative")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must lie in [0, 1)")

        probabilities = self.probabilities
        if len(probabilities) != len(self.arrivals):
            raise ValueError("one probability is required for every arrival count")
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("arrival probabilities must be nonnegative and sum to one")

    @property
    def n_states(self) -> int:
        return self.max_waiting + 1

    @property
    def states(self) -> IntArray:
        return np.arange(self.n_states, dtype=np.int64)

    @property
    def probabilities(self) -> FloatArray:
        if self.arrival_probabilities is None:
            return np.full(len(self.arrivals), 1.0 / len(self.arrivals), dtype=np.float64)
        return np.asarray(self.arrival_probabilities, dtype=np.float64)

    def passengers_after_action(self, state: int, action: int) -> int:
        self._validate_state_action(state, action)
        if action == 0:
            return state
        return max(0, state - self.capacity)

    def immediate_cost(self, state: int, action: int) -> float:
        remaining = self.passengers_after_action(state, action)
        return self.dispatch_cost * action + self.holding_cost * remaining

    def next_state(self, state: int, action: int, arrivals: int) -> int:
        if arrivals < 0:
            raise ValueError("arrivals must be nonnegative")
        remaining = self.passengers_after_action(state, action)
        return min(self.max_waiting, remaining + arrivals)

    def transition_matrices(self) -> tuple[FloatArray, FloatArray]:
        """Return transition matrices and one-step costs for both actions.

        ``transitions[a, s, s_next]`` is the probability of reaching ``s_next`` from
        ``s`` after action ``a``. Action 0 waits and action 1 dispatches.
        """

        transitions = np.zeros((2, self.n_states, self.n_states), dtype=np.float64)
        costs = np.zeros((2, self.n_states), dtype=np.float64)
        for action in (0, 1):
            for state in range(self.n_states):
                costs[action, state] = self.immediate_cost(state, action)
                for arrivals, probability in zip(
                    self.arrivals, self.probabilities, strict=True
                ):
                    next_state = self.next_state(state, action, arrivals)
                    transitions[action, state, next_state] += probability
        return transitions, costs

    def _validate_state_action(self, state: int, action: int) -> None:
        if not 0 <= state <= self.max_waiting:
            raise ValueError(f"state must lie in [0, {self.max_waiting}]")
        if action not in (0, 1):
            raise ValueError("action must be 0 (wait) or 1 (dispatch)")


@dataclass(frozen=True)
class InfiniteHorizonSolution:
    values: FloatArray
    policy: IntArray
    iterations: int
    bellman_residual: float


def bellman_q_values(
    mdp: ShuttleMDP,
    values: FloatArray,
    transitions: FloatArray | None = None,
    costs: FloatArray | None = None,
) -> FloatArray:
    """Compute action values for a cost-minimization Bellman update."""

    if values.shape != (mdp.n_states,):
        raise ValueError(f"values must have shape ({mdp.n_states},)")
    if transitions is None or costs is None:
        transitions, costs = mdp.transition_matrices()
    return costs + mdp.gamma * np.einsum("ask,k->as", transitions, values)


def finite_horizon_enumeration(
    mdp: ShuttleMDP,
    horizon: int = 500,
) -> tuple[FloatArray, IntArray]:
    """Solve the finite-horizon problem by backward state-action enumeration.

    The terminal cost at time ``horizon`` is zero. Returned arrays contain the value
    functions and policies at decision epochs 0 through ``horizon - 1``.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    transitions, costs = mdp.transition_matrices()
    values = np.zeros((horizon + 1, mdp.n_states), dtype=np.float64)
    policy = np.zeros((horizon, mdp.n_states), dtype=np.int64)
    for time in range(horizon - 1, -1, -1):
        q_values = bellman_q_values(mdp, values[time + 1], transitions, costs)
        policy[time] = np.argmin(q_values, axis=0)
        values[time] = np.min(q_values, axis=0)
    return values, policy


def value_iteration(
    mdp: ShuttleMDP,
    tolerance: float = 1e-10,
    max_iterations: int = 100_000,
) -> InfiniteHorizonSolution:
    """Solve the infinite-horizon problem with value iteration."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    transitions, costs = mdp.transition_matrices()
    values = np.zeros(mdp.n_states, dtype=np.float64)
    for iteration in range(1, max_iterations + 1):
        q_values = bellman_q_values(mdp, values, transitions, costs)
        updated = np.min(q_values, axis=0)
        if np.max(np.abs(updated - values)) < tolerance:
            values = updated
            break
        values = updated
    else:
        raise RuntimeError("value iteration did not converge")

    q_values = bellman_q_values(mdp, values, transitions, costs)
    policy = np.argmin(q_values, axis=0).astype(np.int64)
    residual = float(np.max(np.abs(values - np.min(q_values, axis=0))))
    return InfiniteHorizonSolution(values, policy, iteration, residual)


def evaluate_policy_exact(
    mdp: ShuttleMDP,
    policy: IntArray,
    transitions: FloatArray | None = None,
    costs: FloatArray | None = None,
) -> FloatArray:
    """Evaluate a stationary policy by solving its Bellman linear system."""

    if policy.shape != (mdp.n_states,) or not np.all(np.isin(policy, (0, 1))):
        raise ValueError("policy must contain one valid action per state")
    if transitions is None or costs is None:
        transitions, costs = mdp.transition_matrices()
    chosen_transitions = transitions[policy, np.arange(mdp.n_states)]
    chosen_costs = costs[policy, np.arange(mdp.n_states)]
    system = np.eye(mdp.n_states) - mdp.gamma * chosen_transitions
    return np.linalg.solve(system, chosen_costs)


def policy_iteration(
    mdp: ShuttleMDP,
    max_iterations: int = 1_000,
) -> InfiniteHorizonSolution:
    """Solve the infinite-horizon problem with exact policy iteration."""

    transitions, costs = mdp.transition_matrices()
    policy = np.zeros(mdp.n_states, dtype=np.int64)
    for iteration in range(1, max_iterations + 1):
        values = evaluate_policy_exact(mdp, policy, transitions, costs)
        q_values = bellman_q_values(mdp, values, transitions, costs)
        improved = np.argmin(q_values, axis=0).astype(np.int64)
        if np.array_equal(improved, policy):
            residual = float(np.max(np.abs(values - np.min(q_values, axis=0))))
            return InfiniteHorizonSolution(values, policy, iteration, residual)
        policy = improved
    raise RuntimeError("policy iteration did not converge")


def dispatch_threshold(policy: IntArray) -> int | None:
    """Return the first dispatch state when the policy is threshold-like."""

    dispatch_states = np.flatnonzero(policy == 1)
    if dispatch_states.size == 0:
        return None
    return int(dispatch_states[0])

