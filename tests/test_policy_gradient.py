import numpy as np
import pytest
import torch

from mlds490_hw1.policy_gradient import (
    CartPoleNetwork,
    PongNetwork,
    TrainingConfig,
    discounted_returns,
    moving_average,
    policy_action_to_environment_action,
    preprocess_pong,
)


def test_discounted_returns() -> None:
    actual = discounted_returns([1.0, 2.0, 3.0], gamma=0.5)
    np.testing.assert_allclose(actual, [2.75, 3.5, 3.0])


def test_pong_preprocessing_shape_and_binary_values() -> None:
    frame = np.full((210, 160, 3), 144, dtype=np.uint8)
    frame[35, 0, 0] = 255
    processed = preprocess_pong(frame)
    assert processed.shape == (80, 80)
    assert processed.dtype == np.float32
    assert processed[0, 0] == 1
    assert set(np.unique(processed)).issubset({0.0, 1.0})


def test_pong_preprocessing_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        preprocess_pong(np.zeros((80, 80), dtype=np.uint8))


def test_action_mapping_uses_only_assignment_actions_for_pong() -> None:
    config = TrainingConfig(env="pong", baseline="none", episodes=1)
    assert policy_action_to_environment_action(config, 0) == 2
    assert policy_action_to_environment_action(config, 1) == 3


def test_network_output_shapes() -> None:
    cartpole_logits, cartpole_values = CartPoleNetwork(True)(torch.zeros(3, 4))
    pong_logits, pong_values = PongNetwork(True)(torch.zeros(3, 1, 80, 80))
    assert cartpole_logits.shape == (3, 2)
    assert cartpole_values is not None and cartpole_values.shape == (3,)
    assert pong_logits.shape == (3, 2)
    assert pong_values is not None and pong_values.shape == (3,)


def test_moving_average_keeps_alignment() -> None:
    actual = moving_average(np.arange(1, 6, dtype=float), window=3)
    np.testing.assert_allclose(actual[2:], [2.0, 3.0, 4.0])
    assert np.isnan(actual[:2]).all()

