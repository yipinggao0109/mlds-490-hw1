#!/usr/bin/env python3
"""Create a quantitative comparison of policy-gradient runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlds490_hw1.policy_gradient import moving_average


def summarize_run(run_dir: Path) -> dict[str, float | int | str]:
    history = pd.read_csv(run_dir / "training_history.csv")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rewards = history["reward"].to_numpy(dtype=np.float64)
    smoothed = moving_average(rewards, 100)
    summary: dict[str, float | int | str] = {
        "run": str(run_dir),
        "environment": config["env"],
        "baseline": config["baseline"],
        "learning_rate": float(config["learning_rate"]),
        "training_episodes": len(history),
        "first_100_mean": float(np.mean(rewards[:100])),
        "last_100_mean": float(np.mean(rewards[-100:])),
        "last_500_standard_deviation": float(np.std(rewards[-500:], ddof=0)),
        "best_100_episode_mean": float(np.nanmax(smoothed)),
        "best_window_end_episode": int(np.nanargmax(smoothed) + 1),
    }
    evaluation_path = run_dir / "evaluation_stochastic_summary.json"
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "evaluation_episodes": int(evaluation["episodes"]),
                "evaluation_mean": float(evaluation["mean_reward"]),
                "evaluation_standard_deviation": float(evaluation["standard_deviation"]),
                "evaluation_minimum": float(evaluation["minimum_reward"]),
                "evaluation_maximum": float(evaluation["maximum_reward"]),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("results/comparisons"))
    parser.add_argument("--name", default="policy_gradient_comparison")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [summarize_run(run_dir) for run_dir in args.run_dirs]
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / f"{args.name}.csv", index=False)
    (args.output_dir / f"{args.name}.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for run_dir in args.run_dirs:
        history = pd.read_csv(run_dir / "training_history.csv")
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        smoothed = moving_average(history["reward"].to_numpy(dtype=np.float64), 100)
        label = f"{config['baseline']} baseline, lr={config['learning_rate']:g}"
        ax.plot(history["episode"], smoothed, linewidth=1.8, label=label)
    ax.set(
        xlabel="Training episode",
        ylabel="100-episode moving-average reward",
        title="Policy-gradient training comparison",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{args.name}.png", dpi=200)
    fig.savefig(args.output_dir / f"{args.name}.pdf")
    plt.close(fig)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()

