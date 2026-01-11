# Phase 1: Foundation & Simple Baseline

Tabular gradient inversion research - baseline experiments for thesis.

## Overview

This implements Phase 1 of the research on optimization-based gradient inversion attacks for tabular federated learning. We test when reconstruction is feasible on the Adult dataset using simple MLP architectures.

## What's Included

- **Dataset**: Adult (binary classification, 45k samples, mixed categorical/numerical features)
- **Architectures**: 1-layer and 2-layer MLPs (64 hidden units)
- **Batch Sizes**: 1, 4, 5, 8, 16
- **Attacker**: Label-known (fixes labels to ground truth)
- **Baseline**: Prior-only reconstruction (no gradients)

## Quick Start

### 1. Install Dependencies

From the LeakPro root:
```bash
pip install -e .[dev]
```

### 2. Run Experiments

```bash
cd examples/gia/tabular/phase1
python run_phase1.py
```

This will:
- Download and preprocess Adult dataset
- Train models for each architecture
- Extract gradients via FedSGD simulation
- Run gradient inversion attacks
- Compare to prior baseline
- Save results to `./results/`

### 3. View Results

Results are saved in:
- `results/all_results.pkl`: Complete results (pickle format)
- `results/{arch}_batch{bs}_trial{n}.json`: Individual experiment results

## Configuration

Edit `config.yaml` to change:
- Architectures to test
- Batch sizes
- Attack hyperparameters (learning rate, iterations, regularization)
- Number of trials per configuration

## Expected Outcomes

**High-risk configurations** (GIA >> Prior):
- Small batch sizes (1, 4)
- Simple architectures

**Low-risk configurations** (GIA ≈ Prior):
- Larger batch sizes (16+)
- More complex architectures (potentially)

## File Structure

```
phase1/
├── config.yaml              # Experiment configuration
├── run_phase1.py            # Main experiment runner
├── tabular_models.py        # 1-layer and 2-layer MLPs
├── README.md               # This file
├── data/                   # Downloaded Adult dataset
└── results/                # Experiment results
```

## Core Components (in leakpro/attacks/gia_attacks/)

- `tabular_data_utils.py`: Dataset loading and preprocessing
- `tabular_gia_attack.py`: Main GIA attack implementation
- `fedsgd_simulator.py`: FedSGD gradient extraction
- `prior_baseline.py`: Prior-only baseline attack
- `tabular_metrics.py`: Evaluation metrics

## Metrics

**Reconstruction Quality**:
- Numerical features: Normalized RMSE
- Categorical features: Accuracy (correct category recovery)
- Aggregate score: Weighted combination

**Risk Classification**:
- **High risk**: GIA improves >20% over baseline
- **Medium risk**: GIA improves 5-20% over baseline
- **Low risk**: GIA improves <5% over baseline

## Next Steps

After Phase 1 validation:
- Phase 2: Systematic architectural sweep (depth, width, activations, normalization)
- Phase 3: FedAvg and model update-based attacks
- Phase 4: Stochastic preprocessing and label-unknown attacks
- Phase 5: Differential privacy mechanisms
