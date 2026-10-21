#!/usr/bin/env python3
"""Run the shuttle solutions, validation experiments, and required plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlds490_hw1.shuttle import (
    ShuttleMDP,
    dispatch_threshold,
    finite_horizon_enumeration,
    policy_iteration,
    value_iteration,
)


def save_required_plots(
    output_dir: Path,
    mdp: ShuttleMDP,
    finite_values: np.ndarray,
    infinite_values: np.ndarray,
    optimal_policy: np.ndarray,
) -> None:
    states = mdp.states

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(states, finite_values, color="#255C99", linewidth=2)
    ax.set(xlabel="Passengers waiting", ylabel="Optimal expected discounted cost",
           title="Finite-horizon value function at time 0 (T = 500)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "finite_horizon_value_t0.pdf")
    fig.savefig(output_dir / "finite_horizon_value_t0.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(states, infinite_values, color="#D1495B", linewidth=2)
    ax.set(xlabel="Passengers waiting", ylabel="Optimal expected discounted cost",
           title="Infinite-horizon optimal value function")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "infinite_horizon_value.pdf")
    fig.savefig(output_dir / "infinite_horizon_value.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.step(states, optimal_policy, where="post", color="#2A9D8F", linewidth=2)
    ax.set(
        xlabel="Passengers waiting",
        ylabel="Optimal action",
        yticks=[0, 1],
        yticklabels=["Wait", "Dispatch"],
        title="Infinite-horizon optimal dispatch policy",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "optimal_policy.pdf")
    fig.savefig(output_dir / "optimal_policy.png", dpi=200)
    plt.close(fig)


def comparative_statics(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | None]] = []
    for dispatch_cost in (60.0, 80.0, 100.0, 120.0, 140.0):
        mdp = ShuttleMDP(dispatch_cost=dispatch_cost)
        solution = value_iteration(mdp)
        rows.append(
            {
                "experiment": "dispatch_cost",
                "parameter_value": dispatch_cost,
                "first_dispatch_state": dispatch_threshold(solution.policy),
                "bellman_residual": solution.bellman_residual,
            }
        )
    for holding_cost in (1.0, 1.5, 2.0, 2.5, 3.0):
        mdp = ShuttleMDP(holding_cost=holding_cost)
        solution = value_iteration(mdp)
        rows.append(
            {
                "experiment": "holding_cost",
                "parameter_value": holding_cost,
                "first_dispatch_state": dispatch_threshold(solution.policy),
                "bellman_residual": solution.bellman_residual,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "comparative_statics.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/shuttle"))
    parser.add_argument("--horizon", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mdp = ShuttleMDP()
    finite_values, finite_policy = finite_horizon_enumeration(mdp, args.horizon)
    vi_solution = value_iteration(mdp)
    pi_solution = policy_iteration(mdp)

    value_gap = float(np.max(np.abs(vi_solution.values - pi_solution.values)))
    finite_infinite_gap = float(np.max(np.abs(finite_values[0] - vi_solution.values)))
    policy_agreement = float(np.mean(vi_solution.policy == pi_solution.policy))

    save_required_plots(
        args.output_dir,
        mdp,
        finite_values[0],
        vi_solution.values,
        pi_solution.policy,
    )
    statics = comparative_statics(args.output_dir)

    pd.DataFrame(
        {
            "state": mdp.states,
            "finite_value_t0": finite_values[0],
            "finite_policy_t0": finite_policy[0],
            "value_iteration_value": vi_solution.values,
            "value_iteration_policy": vi_solution.policy,
            "policy_iteration_value": pi_solution.values,
            "policy_iteration_policy": pi_solution.policy,
        }
    ).to_csv(args.output_dir / "solutions.csv", index=False)

    summary = {
        "assumptions": {
            "event_order": "decision, dispatch and holding cost, then arrivals",
            "overflow_rule": "clip the next state at 200 passengers",
            "terminal_cost": 0.0,
        },
        "finite_horizon": args.horizon,
        "value_iteration_iterations": vi_solution.iterations,
        "policy_iteration_iterations": pi_solution.iterations,
        "value_iteration_bellman_residual": vi_solution.bellman_residual,
        "policy_iteration_bellman_residual": pi_solution.bellman_residual,
        "vi_pi_max_value_gap": value_gap,
        "finite_infinite_max_value_gap": finite_infinite_gap,
        "vi_pi_policy_agreement": policy_agreement,
        "first_dispatch_state": dispatch_threshold(pi_solution.policy),
        "comparative_statics": statics.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

