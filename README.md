# MLDS 490 Assignment 1

This repository contains reproducible implementations and experiments for the shuttle-dispatch
dynamic programming problem and the policy-gradient experiments on CartPole-v1 and Pong-v5.

## Environment setup

Python 3.11 or 3.12 is recommended. From the repository root, run:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev]'
AutoROM --accept-license
```

The last command installs the Atari ROMs required by Pong-v5. The ROM license must be accepted by
the person running the experiments.

## Quick validation

```bash
pytest
python scripts/run_shuttle.py --output-dir results/shuttle
```

## Policy-gradient experiments

The commands below reproduce the four selected runs reported in the PDF. The learning rates are
recorded here and in each run's `config.json`.

```bash
# Part 1: REINFORCE without a baseline
python scripts/train_policy.py --env cartpole --baseline none --episodes 3000 \
  --learning-rate 0.0005 --output-dir results/lr_0_0005 --device cpu
python scripts/train_policy.py --env pong --baseline none --episodes 5000 \
  --episodes-per-update 10 --learning-rate 0.001 \
  --output-dir results/pong_mlp_batch10_lr_0_001 --device cpu

# Part 2: REINFORCE with a learned state-value baseline
python scripts/train_policy.py --env cartpole --baseline value --episodes 5000 \
  --learning-rate 0.0003 --output-dir results/lr_0_0003 --device cpu
python scripts/train_policy.py --env pong --baseline value --episodes 5000 \
  --episodes-per-update 10 --learning-rate 0.001 \
  --output-dir results/pong_mlp_batch10_lr_0_001 --device cpu
```

Each training command writes a checkpoint, a configuration file, a per-episode CSV file, and a
training-curve figure to a separate directory under `results/`. Evaluation uses the saved model:

```bash
python scripts/evaluate_policy.py \
  --run-dir results/lr_0_0005/cartpole_none_seed490 --episodes 500 --device cpu
python scripts/evaluate_policy.py \
  --run-dir results/lr_0_0003/cartpole_value_seed490 --episodes 500 --device cpu
python scripts/evaluate_policy.py \
  --run-dir results/pong_mlp_batch10_lr_0_001/pong_none_seed490 \
  --episodes 500 --device cpu
python scripts/evaluate_policy.py \
  --run-dir results/pong_mlp_batch10_lr_0_001/pong_value_seed490 \
  --episodes 500 --device cpu
```

## Quantitative validation

```bash
# Uniform random controls
python scripts/evaluate_random_policy.py --env cartpole --episodes 500
python scripts/evaluate_random_policy.py --env pong --episodes 500

# Inspect the first 10 decisions of a saved policy
python scripts/trace_policy.py \
  --run-dir results/pong_mlp_batch10_lr_0_001/pong_value_seed490 --steps 10

# CartPole seed-sensitivity runs used in the report
for seed in 491 492; do
  python scripts/train_policy.py --env cartpole --baseline none --episodes 3000 \
    --seed "$seed" --learning-rate 0.0005 --output-dir results/multiseed --device cpu
  python scripts/train_policy.py --env cartpole --baseline value --episodes 5000 \
    --seed "$seed" --learning-rate 0.0003 --output-dir results/multiseed --device cpu
done

# Compare selected checkpoints
python scripts/compare_runs.py \
  results/pong_mlp_batch10_lr_0_001/pong_none_seed490 \
  results/pong_mlp_batch10_lr_0_001/pong_value_seed490 \
  --output-dir results/comparisons --name pong_final_comparison
```

## Build the report

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
cp report.pdf mlds490_hw1_report.pdf
```

## Reproducibility notes

- Every command records its seed and hyperparameters in `config.json`.
- Evaluation is stochastic by default because the learned object is a stochastic policy.
- The Pong policy selects only actions 2 and 3. A single FIRE action is used only when an Atari
  reset requires it to begin play.
- `mlds_assignment1.pdf` is the original assignment handout and is not part of the implementation.
