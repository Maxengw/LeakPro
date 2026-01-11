"""Quick validation test for Phase 1 implementation."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

from leakpro.attacks.gia_attacks.tabular_data_utils import (
    download_adult_dataset,
    preprocess_adult_dataset,
    get_data_statistics
)
from leakpro.attacks.gia_attacks.fedsgd_simulator import FedSGDSimulator
from leakpro.attacks.gia_attacks.tabular_gia_attack import TabularGIA
from leakpro.attacks.gia_attacks.prior_baseline import PriorBaseline
from leakpro.attacks.gia_attacks.tabular_metrics import (
    evaluate_reconstruction,
    compare_to_baseline
)
from leakpro.fl_utils.gia_optimizers import MetaSGD
from examples.gia.tabular.phase1.tabular_models import get_model

print("="*60)
print("Phase 1 Quick Validation Test")
print("="*60)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Test 1: Data loading
print("\n[1/6] Loading Adult dataset...")
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)
download_adult_dataset(str(data_dir))
dataset = preprocess_adult_dataset(str(data_dir))
print(f"  Loaded: {len(dataset)} samples, {dataset.num_features} features")

# Test 2: Model creation
print("\n[2/6] Creating models...")
model = get_model("1layer", dataset.num_features, hidden_size=64, num_classes=1)
print(f"  1-layer model: {sum(p.numel() for p in model.parameters())} parameters")

# Test 3: Gradient extraction
print("\n[3/6] Extracting gradients with FedSGD...")
indices = list(range(4))
subset = Subset(dataset, indices)
loader = DataLoader(subset, batch_size=4, shuffle=False)

criterion = nn.BCEWithLogitsLoss()
meta_optimizer = MetaSGD(lr=0.1)

fedsgd_sim = FedSGDSimulator(model, criterion, meta_optimizer)
gradients = fedsgd_sim.extract_client_gradient(loader)
print(f"  Extracted: {len(gradients)} gradient tensors")

# Test 4: GIA attack (very short)
print("\n[4/6] Running GIA attack (50 iterations)...")
config = {
    'attack_lr': 0.1,
    'iterations': 50,
    'reg_weight': 0.01,
    'label_known': True
}

gia_attack = TabularGIA(
    model=model,
    criterion=criterion,
    dataset=dataset,
    client_loader=loader,
    observed_gradients=gradients,
    config=config
)

gia_result = None
for iteration, score, result in gia_attack.run_attack():
    if result is not None:
        gia_result = result

print(f"  GIA Score: {gia_result['reconstruction_metrics']['aggregate']['overall_score']:.4f}")

# Test 5: Prior baseline
print("\n[5/6] Running prior baseline...")
data_stats = get_data_statistics(dataset)
prior_baseline = PriorBaseline(dataset, data_stats)

ground_truth, labels = next(iter(loader))
baseline_reconstruction = prior_baseline.reconstruct_batch(ground_truth, labels)

baseline_metrics = evaluate_reconstruction(
    ground_truth,
    baseline_reconstruction,
    dataset.feature_info,
    return_per_feature=False
)
print(f"  Baseline Score: {baseline_metrics['aggregate']['overall_score']:.4f}")

# Test 6: Comparison
print("\n[6/6] Comparing GIA to baseline...")
comparison = compare_to_baseline(
    gia_result['reconstruction_metrics'],
    baseline_metrics
)
print(f"  Improvement: {comparison['relative_improvement']*100:.2f}%")
print(f"  Risk Level: {comparison['risk_level']}")

print("\n" + "="*60)
print("ALL TESTS PASSED - Phase 1 is ready!")
print("Run 'python run_phase1.py' for full experiments")
print("="*60)
