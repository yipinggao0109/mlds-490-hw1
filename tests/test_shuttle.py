import numpy as np
import pytest

from mlds490_hw1.shuttle import (
    ShuttleMDP,
    dispatch_threshold,
    finite_horizon_enumeration,
    policy_iteration,
    value_iteration,
)


def test_transition_rows_are_probability_distributions() -> None:
    transitions, _ = ShuttleMDP().transition_matrices()
    np.testing.assert_allclose(transitions.sum(axis=2), 1.0)
    assert np.all(transitions >= 0)


def test_capacity_and_station_boundaries() -> None:
    mdp = ShuttleMDP()
    assert mdp.passengers_after_action(0, 1) == 0
    assert mdp.passengers_after_action(10, 1) == 0
    assert mdp.passengers_after_action(20, 1) == 5
    assert mdp.next_state(200, 0, 5) == 200
    assert mdp.next_state(200, 1, 5) == 190


def test_one_period_solution_matches_immediate_costs() -> None:
    mdp = ShuttleMDP()
    values, policy = finite_horizon_enumeration(mdp, horizon=1)
    expected_cost = np.minimum(
        mdp.holding_cost * mdp.states,
        mdp.dispatch_cost + mdp.holding_cost * np.maximum(0, mdp.states - mdp.capacity),
    )
    np.testing.assert_allclose(values[0], expected_cost)
    np.testing.assert_array_equal(policy[0], np.zeros(mdp.n_states, dtype=np.int64))


def test_value_and_policy_iteration_agree() -> None:
    mdp = ShuttleMDP()
    value_solution = value_iteration(mdp)
    policy_solution = policy_iteration(mdp)
    np.testing.assert_allclose(value_solution.values, policy_solution.values, atol=1e-7)
    np.testing.assert_array_equal(value_solution.policy, policy_solution.policy)
    assert value_solution.bellman_residual < 1e-8
    assert policy_solution.bellman_residual < 1e-8


def test_optimal_value_is_monotone_in_waiting_count() -> None:
    values = value_iteration(ShuttleMDP()).values
    assert np.all(np.diff(values) >= -1e-9)


def test_higher_dispatch_cost_weakly_raises_first_dispatch_state() -> None:
    low_cost_policy = value_iteration(ShuttleMDP(dispatch_cost=60)).policy
    high_cost_policy = value_iteration(ShuttleMDP(dispatch_cost=140)).policy
    low_threshold = dispatch_threshold(low_cost_policy)
    high_threshold = dispatch_threshold(high_cost_policy)
    assert low_threshold is not None and high_threshold is not None
    assert high_threshold >= low_threshold


def test_invalid_state_or_action_is_rejected() -> None:
    mdp = ShuttleMDP()
    with pytest.raises(ValueError):
        mdp.immediate_cost(-1, 0)
    with pytest.raises(ValueError):
        mdp.immediate_cost(0, 2)
