#!/usr/bin/env python3
"""Train REINFORCE with or without a learned value baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlds490_hw1.policy_gradient import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=("cartpole", "pong"), required=True)
    parser.add_argument("--baseline", choices=("none", "value"), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=1)
    parser.add_argument("--update-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = TrainingConfig(
        env=args.env,
        baseline=args.baseline,
        episodes=args.episodes,
        seed=args.seed,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        max_grad_norm=args.max_grad_norm,
        checkpoint_every=args.checkpoint_every,
        episodes_per_update=args.episodes_per_update,
        update_batch_size=args.update_batch_size,
        device=args.device,
    )
    run_dir = args.output_dir / f"{args.env}_{args.baseline}_seed{args.seed}"
    train(config, run_dir, resume=args.resume)


if __name__ == "__main__":
    main()
