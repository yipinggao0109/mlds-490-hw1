#!/usr/bin/env python3
"""Evaluate a trained policy and plot the reward distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlds490_hw1.policy_gradient import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate(args.run_dir, args.episodes, args.deterministic, args.device)


if __name__ == "__main__":
    main()

