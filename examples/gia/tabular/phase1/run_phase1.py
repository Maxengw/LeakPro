"""Phase 1 Experiment Runner: Foundation for Tabular Gradient Inversion Research.

This script runs the baseline experiments:
- Adult dataset
- 1-layer and 2-layer architectures  
- Batch sizes: 1, 4, 5, 8, 16
- Label-known attacker
- Comparison to prior baseline
"""

import os
import sys
import json
import pickle
import yaml
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Add LeakPro to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

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
    evaluate_utility,
    compare_to_baseline
)
from leakpro.fl_utils.gia_optimizers import MetaSGD
from examples.gia.tabular.phase1.tabular_models import get_model


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    momentum: float
) -> Dict:
    """Train a model and return metrics.
    
    Args:
        model: Model to train
        train_loader: Training data
        test_loader: Test data
        epochs: Number of epochs
        lr: Learning rate
        momentum: SGD momentum
        
    Returns:
        Dictionary with training metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    train_losses = []
    train_accs = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
    
    # Evaluate on test set
    test_metrics = evaluate_utility(model, test_loader, criterion, task_type='binary_classification')
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_metrics': test_metrics
    }


def run_single_experiment(
    architecture: str,
    batch_size: int,
    config: Dict,
    dataset,
    trial_num: int = 0
) -> Dict:
    """Run a single experiment configuration.
    
    Args:
        architecture: "1layer" or "2layer"
        batch_size: Batch size for client
        config: Experiment configuration
        dataset: Preprocessed TabularDataset
        trial_num: Trial number for this configuration
        
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running: {architecture}, batch_size={batch_size}, trial={trial_num}")
    print(f"{'='*60}")
    
    # Set random seed
    seed = config['random_seed'] + trial_num
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataloaders
    train_size = int(config['train_fraction'] * len(dataset))
    test_size = int(config['test_fraction'] * len(dataset))
    
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)
    
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:train_size+test_size]
    
    train_subset = Subset(dataset, train_indices.tolist())
    test_subset = Subset(dataset, test_indices.tolist())
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
    
    # Create model
    model = get_model(
        architecture=architecture,
        input_size=dataset.num_features,
        hidden_size=config['hidden_size'],
        num_classes=config['num_classes']
    )
    
    print("Training target model...")
    training_metrics = train_model(
        model,
        train_loader,
        test_loader,
        epochs=config['training_epochs'],
        lr=config['learning_rate'],
        momentum=config['momentum']
    )
    
    print(f"Model trained. Test accuracy: {training_metrics['test_metrics']['accuracy']:.4f}")
    
    # Create client batch for attack
    client_indices = test_indices[:batch_size]
    client_subset = Subset(dataset, client_indices.tolist())
    client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=False)
    
    # Extract gradients using FedSGD
    print("Extracting client gradients...")
    criterion = nn.BCEWithLogitsLoss()
    meta_optimizer = MetaSGD(lr=config['learning_rate'])
    
    fedsgd_sim = FedSGDSimulator(model, criterion, meta_optimizer)
    observed_gradients = fedsgd_sim.extract_client_gradient(client_loader)
    
    print(f"Extracted {len(observed_gradients)} gradient tensors")
    
    # Run GIA attack
    print("Running gradient inversion attack...")
    gia_attack = TabularGIA(
        model=model,
        criterion=criterion,
        dataset=dataset,
        client_loader=client_loader,
        observed_gradients=observed_gradients,
        config=config['attack_config']
    )
    
    # Run attack and collect final result
    gia_result = None
    for iteration, score, result in gia_attack.run_attack():
        if result is not None:
            gia_result = result
    
    # Run prior baseline
    print("Running prior baseline...")
    data_stats = get_data_statistics(dataset)
    prior_baseline = PriorBaseline(dataset, data_stats)
    
    # Get ground truth for evaluation
    ground_truth_data, ground_truth_labels = next(iter(client_loader))
    
    # Baseline reconstruction
    baseline_reconstruction = prior_baseline.reconstruct_batch(ground_truth_data, ground_truth_labels)
    
    baseline_metrics = evaluate_reconstruction(
        ground_truth_data,
        baseline_reconstruction,
        dataset.feature_info,
        return_per_feature=False
    )
    
    # Compare GIA to baseline
    comparison = compare_to_baseline(
        gia_result['reconstruction_metrics'],
        baseline_metrics
    )
    
    print(f"\nResults:")
    print(f"  GIA Score: {comparison['gia_score']:.4f}")
    print(f"  Baseline Score: {comparison['baseline_score']:.4f}")
    print(f"  Improvement: {comparison['relative_improvement']*100:.2f}%")
    print(f"  Risk Level: {comparison['risk_level']}")
    
    # Compile results
    results = {
        'architecture': architecture,
        'batch_size': batch_size,
        'trial_num': trial_num,
        'seed': seed,
        'training_metrics': training_metrics,
        'gia_metrics': gia_result['reconstruction_metrics'],
        'baseline_metrics': baseline_metrics,
        'comparison': comparison,
        'constraint_violations': gia_result['constraint_violations']
    }
    
    return results


def main():
    """Main experiment runner."""
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Phase 1: Foundation & Simple Baseline")
    print("="*60)
    
    # Prepare data
    data_dir = Path(__file__).parent / config['data_dir']
    data_dir.mkdir(exist_ok=True)
    
    print("\nDownloading and preprocessing Adult dataset...")
    download_adult_dataset(str(data_dir))
    dataset = preprocess_adult_dataset(str(data_dir))
    
    print(f"Dataset ready: {len(dataset)} samples, {dataset.num_features} features")
    
    # Create results directory
    results_dir = Path(__file__).parent / config['results_dir']
    results_dir.mkdir(exist_ok=True)
    
    # Run all experiments
    all_results = []
    
    for architecture in config['architectures']:
        for batch_size in config['batch_sizes']:
            for trial in range(config['num_trials_per_config']):
                try:
                    result = run_single_experiment(
                        architecture=architecture,
                        batch_size=batch_size,
                        config=config,
                        dataset=dataset,
                        trial_num=trial
                    )
                    all_results.append(result)
                    
                    # Save individual result
                    result_file = results_dir / f"{architecture}_batch{batch_size}_trial{trial}.json"
                    with open(result_file, 'w') as f:
                        # Convert to JSON-serializable format
                        json_result = {k: v for k, v in result.items() if not isinstance(v, (torch.Tensor, np.ndarray))}
                        json.dump(json_result, f, indent=2)
                    
                except Exception as e:
                    print(f"Error in experiment {architecture}, batch={batch_size}, trial={trial}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Save all results
    print(f"\n{'='*60}")
    print(f"All experiments complete. Saving summary...")
    
    summary_file = results_dir / "all_results.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"Results saved to {results_dir}")
    print("\nPhase 1 experiments complete!")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    
    for arch in config['architectures']:
        print(f"\n{arch.upper()}:")
        for bs in config['batch_sizes']:
            results_for_config = [r for r in all_results if r['architecture'] == arch and r['batch_size'] == bs]
            if results_for_config:
                avg_gia = np.mean([r['comparison']['gia_score'] for r in results_for_config])
                avg_baseline = np.mean([r['comparison']['baseline_score'] for r in results_for_config])
                avg_improvement = np.mean([r['comparison']['relative_improvement'] for r in results_for_config])
                risk_levels = [r['comparison']['risk_level'] for r in results_for_config]
                most_common_risk = max(set(risk_levels), key=risk_levels.count)
                
                print(f"  Batch={bs:2d}: GIA={avg_gia:.3f}, Baseline={avg_baseline:.3f}, "+
                      f"Improvement={avg_improvement*100:5.1f}%, Risk={most_common_risk}")


if __name__ == "__main__":
    main()
