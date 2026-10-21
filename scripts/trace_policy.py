#!/usr/bin/env python3
"""Record the first steps of a trained policy for algorithm validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mlds490_hw1.policy_gradient import (
    load_run,
    make_environment,
    policy_action_to_environment_action,
    preprocess_pong,
    reset_environment,
    state_for_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1490)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config, model, device = load_run(args.run_dir, args.device)
    environment = make_environment(config)
    observation = reset_environment(environment, config, args.seed)
    previous_frame: np.ndarray | None = None
    rows: list[dict[str, float | int | str]] = []
    cumulative_reward = 0.0

    try:
        for step in range(args.steps):
            state, previous_frame = state_for_policy(config, observation, previous_frame)
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, values = model(state_tensor)
                probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            policy_action = int(np.argmax(probabilities))
            environment_action = policy_action_to_environment_action(config, policy_action)
            next_observation, reward, terminated, truncated, _ = environment.step(
                environment_action
            )
            cumulative_reward += float(reward)
            row: dict[str, float | int | str] = {
                "step": step,
                "policy_action": policy_action,
                "environment_action": environment_action,
                "probability_action_0": float(probabilities[0]),
                "probability_action_1": float(probabilities[1]),
                "reward": float(reward),
                "cumulative_reward": cumulative_reward,
                "value_estimate": float(values.item()) if values is not None else np.nan,
                "terminated": int(terminated),
                "truncated": int(truncated),
            }
            if config.env == "cartpole":
                row.update(
                    {
                        "cart_position": float(state[0]),
                        "cart_velocity": float(state[1]),
                        "pole_angle": float(state[2]),
                        "pole_angular_velocity": float(state[3]),
                    }
                )
            else:
                current = preprocess_pong(np.asarray(observation, dtype=np.uint8))
                row.update(
                    {
                        "foreground_pixels": int(current.sum()),
                        "changed_pixels": int(np.count_nonzero(state)),
                    }
                )
            rows.append(row)
            observation = np.asarray(next_observation)
            if terminated or truncated:
                break
    finally:
        environment.close()

    frame = pd.DataFrame(rows)
    output_path = args.run_dir / f"first_{args.steps}_steps_trace.csv"
    frame.to_csv(output_path, index=False)
    print(frame.to_string(index=False))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()

