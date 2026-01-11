"""Quick test script to validate Phase 1 implementation."""

import sys
from pathlib import Path

# Add LeakPro to path
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


def test_data_loading():
    """Test 1: Data loading and preprocessing."""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    download_adult_dataset(str(data_dir))
    dataset = preprocess_adult_dataset(str(data_dir))
    
    assert len(dataset) == 45222, f"Expected 45222 samples, got {len(dataset)}"
    assert dataset.num_features == 104, f"Expected 104 features, got {dataset.num_features}"
    
    print(f"✓ Dataset loaded: {len(dataset)} samples, {dataset.num_features} features")
    return dataset


def test_model_creation(dataset):
    """Test 2: Model creation."""
    print("\n" + "="*60)
    print("TEST 2: Model Creation")
    print("="*60)
    
    model_1layer = get_model("1layer", dataset.num_features, hidden_size=64, num_classes=1)
    model_2layer = get_model("2layer", dataset.num_features, hidden_size=64, num_classes=1)
    
    # Test forward pass
    x = torch.randn(4, dataset.num_features)
    out1 = model_1layer(x)
    out2 = model_2layer(x)
    
    assert out1.shape == (4, 1), f"Expected shape (4, 1), got {out1.shape}"
    assert out2.shape == (4, 1), f"Expected shape (4, 1), got {out2.shape}"
    
    print(f"✓ 1-layer model created: {sum(p.numel() for p in model_1layer.parameters())} parameters")
    print(f"✓ 2-layer model created: {sum(p.numel() for p in model_2layer.parameters())} parameters")
    
    return model_1layer


def test_gradient_extraction(dataset, model):
    """Test 3: FedSGD gradient extraction."""
    print("\n" + "="*60)
    print("TEST 3: Gradient Extraction")
    print("="*60)
    
    # Create small batch
    indices = list(range(4))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=4, shuffle=False)
    
    criterion = nn.BCEWithLogitsLoss()
    meta_optimizer = MetaSGD(lr=0.1)
    
    fedsgd_sim = FedSGDSimulator(model, criterion, meta_optimizer)
    gradients = fedsgd_sim.extract_client_gradient(loader)
    
    assert len(gradients) > 0, "No gradients extracted"
    assert all(g is not None for g in gradients), "Some gradients are None"
    assert all(g.requires_grad for g in gradients), "Gradients don't require grad"
    
    print(f"✓ Extracted {len(gradients)} gradient tensors")
    print(f"✓ Gradients maintain computational graph")
    
    return gradients, loader


def test_gia_attack(dataset, model, gradients, client_loader):
    """Test 4: GIA attack (quick version)."""
    print("\n" + "="*60)
    print("TEST 4: Gradient Inversion Attack (Quick)")
    print("="*60)
    
    criterion = nn.BCEWithLogitsLoss()
    
    config = {
        'attack_lr': 0.1,
        'iterations': 50,  # Very short for quick test
        'reg_weight': 0.01,
        'label_known': True
    }
    
    gia_attack = TabularGIA(
        model=model,
        criterion=criterion,
        dataset=dataset,
        client_loader=client_loader,
        observed_gradients=gradients,
        config=config
    )
    
    # Run attack  
    gia_result = None
    for iteration, score, result in gia_attack.run_attack():
        if result is not None:
            gia_result = result
    
    assert gia_result is not None, "No GIA result returned"
    assert 'reconstruction_metrics' in gia_result, "Missing reconstruction metrics"
    
    print(f"✓ GIA attack completed")
    print(f"✓ Reconstruction score: {gia_result['reconstruction_metrics']['aggregate']['overall_score']:.4f}")
    
    return gia_result


def test_prior_baseline(dataset, client_loader):
    """Test 5: Prior baseline."""
    print("\n" + "="*60)
    print("TEST 5: Prior Baseline")
    print("="*60)
    
    data_stats = get_data_statistics(dataset)
    prior_baseline = PriorBaseline(dataset, data_stats)
    
    ground_truth, labels = next(iter(client_loader))
    baseline_reconstruction = prior_baseline.reconstruct_batch(ground_truth, labels)
    
    assert baseline_reconstruction.shape == ground_truth.shape, "Shape mismatch"
    
    baseline_metrics = evaluate_reconstruction(
        ground_truth,
        baseline_reconstruction,
        dataset.feature_info,
        return_per_feature=False
    )
    
    print(f"✓ Prior baseline completed")
    print(f"✓ Baseline score: {baseline_metrics['aggregate']['overall_score']:.4f}")
    
    return baseline_metrics


def test_comparison(gia_result, baseline_metrics):
    """Test 6: Risk comparison."""
    print("\n" + "="*60)
    print("TEST 6: Risk Classification")
    print("="*60)
    
    comparison = compare_to_baseline(
        gia_result['reconstruction_metrics'],
        baseline_metrics
    )
    
    print(f"✓ GIA Score: {comparison['gia_score']:.4f}")
    print(f"✓ Baseline Score: {comparison['baseline_score']:.4f}")
    print(f"✓ Relative Improvement: {comparison['relative_improvement']*100:.2f}%")
    print(f"✓ Risk Level: {comparison['risk_level']}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Phase 1 Implementation Validation Tests")
    print("="*60)
    
    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Run tests
        dataset = test_data_loading()
        model = test_model_creation(dataset)
        gradients, client_loader = test_gradient_extraction(dataset, model)
        gia_result = test_gia_attack(dataset, model, gradients, client_loader)
        baseline_metrics = test_prior_baseline(dataset, client_loader)
        test_comparison(gia_result, baseline_metrics)
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nPhase 1 implementation is ready!")
        print("Run 'python run_phase1.py' for full experiments.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
