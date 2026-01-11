"""Evaluation metrics for tabular gradient inversion attacks."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error


def normalized_rmse(true_values: np.ndarray, reconstructed_values: np.ndarray) -> float:
    """Compute normalized RMSE for numerical features.
    
    Args:
        true_values: Ground truth values
        reconstructed_values: Reconstructed values
        
    Returns:
        Normalized RMSE (RMSE / range of true values)
    """
    rmse = np.sqrt(mean_squared_error(true_values, reconstructed_values))
    value_range = true_values.max() - true_values.min() 
    
    if value_range == 0:
        return 0.0  # All values are the same
    
    return rmse / value_range


def categorical_accuracy(true_onehot: np.ndarray, reconstructed_onehot: np.ndarray) -> float:
    """Compute accuracy for one-hot encoded categorical features.
    
    Args:
        true_onehot: Ground truth one-hot vectors (n_samples, n_categories)
        reconstructed_onehot: Reconstructed one-hot vectors (n_samples, n_categories)
        
    Returns:
        Accuracy (fraction of correctly recovered categories)
    """
    true_labels = np.argmax(true_onehot, axis=1)
    reconstructed_labels = np.argmax(reconstructed_onehot, axis=1)
    
    return accuracy_score(true_labels, reconstructed_labels)


def evaluate_reconstruction(
    ground_truth: Tensor,
    reconstructed: Tensor,
    feature_info: Dict,
    return_per_feature: bool = True
) -> Dict:
    """Evaluate reconstruction quality for tabular data.
    
    Args:
        ground_truth: Ground truth data tensor (n_samples, n_features)
        reconstructed: Reconstructed data tensor (n_samples, n_features)
        feature_info: Feature metadata from TabularDataset
        return_per_feature: If True, return metrics for each feature
        
    Returns:
        Dictionary of evaluation metrics
    """
    gt_np = ground_truth.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()
    
    results = {}
    
    # Evaluate numerical features
    numerical_features = feature_info['numerical_features']
    numerical_scores = []
    
    if return_per_feature:
        results['numerical'] = {}
    
    for i, feature_name in enumerate(numerical_features):
        feature_indices = feature_info['dec_to_onehot'][i]
        
        # Get values for this numerical feature
        gt_values = gt_np[:, feature_indices[0]]
        recon_values = recon_np[:, feature_indices[0]]
        
        # Compute RMSE
        nrmse = normalized_rmse(gt_values, recon_values)
        numerical_scores.append(nrmse)
        
        if return_per_feature:
            results['numerical'][feature_name] = {
                'nrmse': float(nrmse),
                'mae': float(mean_absolute_error(gt_values, recon_values))
            }
    
    # Evaluate categorical features
    categorical_features = feature_info['categorical_features']
    categorical_scores = []
    
    if return_per_feature:
        results['categorical'] = {}
    
    for i, feature_name in enumerate(categorical_features):
        # Get the dec_to_onehot index for this categorical feature
        dec_idx = len(numerical_features) + i
        onehot_indices = feature_info['dec_to_onehot'][dec_idx]
        
        # Extract one-hot vectors
        gt_onehot = gt_np[:, onehot_indices]
        recon_onehot = recon_np[:, onehot_indices]
        
        # Compute accuracy
        acc = categorical_accuracy(gt_onehot, recon_onehot)
        categorical_scores.append(acc)
        
        if return_per_feature:
            results['categorical'][feature_name] = {
                'accuracy': float(acc)
            }
    
    # Aggregate scores
    results['aggregate'] = {
        'numerical_mean_nrmse': float(np.mean(numerical_scores)) if numerical_scores else 0.0,
        'categorical_mean_accuracy': float(np.mean(categorical_scores)) if categorical_scores else 0.0,
    }
    
    # Overall reconstruction score (weighted average, higher is better)
    # Convert RMSE (lower is better) to a score (higher is better)
    numerical_score = 1.0 - np.mean(numerical_scores) if numerical_scores else 0.5
    categorical_score = np.mean(categorical_scores) if categorical_scores else 0.5
    
    # Weight equally for now
    overall_score = 0.5 * numerical_score + 0.5 * categorical_score
    results['aggregate']['overall_score'] = float(overall_score)
    
    return results


def count_constraint_violations(
    reconstructed: Tensor,
    feature_info: Dict
) -> Dict:
    """Count constraint violations in reconstructed data.
    
    Args:
        reconstructed: Reconstructed data tensor (n_samples, n_features)
        feature_info: Feature metadata from TabularDataset
        
    Returns:
        Dictionary with violation counts
    """
    recon_np = reconstructed.detach().cpu().numpy()
    n_samples = recon_np.shape[0]
    
    violations = {
        'categorical_invalid_onehot': 0,
        'total_samples': n_samples
    }
    
    # Check categorical features for valid one-hot encoding
    numerical_features = feature_info['numerical_features']
    categorical_features = feature_info['categorical_features']
    
    for i, feature_name in enumerate(categorical_features):
        dec_idx = len(numerical_features) + i
        onehot_indices = feature_info['dec_to_onehot'][dec_idx]
        
        # Extract one-hot vectors
        onehot_vectors = recon_np[:, onehot_indices]
        
        # Check if each vector sums to approximately 1 and has one dominant value
        sums = onehot_vectors.sum(axis=1)
        max_values = onehot_vectors.max(axis=1)
        
        # Count violations (sum not close to 1 or no clear maximum)
        invalid = np.logical_or(np.abs(sums - 1.0) > 0.1, max_values < 0.5)
        violations['categorical_invalid_onehot'] += invalid.sum()
    
    # violations['categorical_invalid_onehot_fraction'] = violations['categorical_invalid_onehot'] / (n_samples * len(categorical_features))
    
    return violations


def evaluate_utility(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    task_type: str = 'binary_classification'
) -> Dict:
    """Evaluate model utility (task performance).
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        task_type: Type of task ('binary_classification', 'multiclass', 'regression')
        
    Returns:
        Dictionary with utility metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_outputs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Compute loss
            if task_type == 'binary_classification':
                labels_for_loss = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels_for_loss)
            else:
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            if task_type == 'binary_classification':
                predictions = (torch.sigmoid(outputs) >= 0.5).long().squeeze()
            elif task_type == 'multiclass':
                predictions = outputs.argmax(dim=1)
            else:  # regression
                predictions = outputs.squeeze()
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    
    avg_loss = total_loss / len(dataloader)
    
    results = {'loss': float(avg_loss)}
    
    if task_type in ['binary_classification', 'multiclass']:
        results['accuracy'] = float(accuracy_score(all_labels, all_predictions))
        results['f1'] = float(f1_score(all_labels, all_predictions, average='binary' if task_type == 'binary_classification' else 'macro'))
        
        # AUC for binary classification
        if task_type == 'binary_classification':
            probabilities = torch.sigmoid(torch.tensor(all_outputs)).numpy()
            results['auc'] = float(roc_auc_score(all_labels, probabilities))
    else:  # regression
        results['rmse'] = float(np.sqrt(mean_squared_error(all_labels, all_predictions)))
        results['mae'] = float(mean_absolute_error(all_labels, all_predictions))
    
    return results


def compare_to_baseline(
    gia_results: Dict,
    baseline_results: Dict
) -> Dict:
    """Compare GIA reconstruction to prior-only baseline.
    
    Args:
        gia_results: Results from GIA reconstruction
        baseline_results: Results from prior-only baseline
        
    Returns:
        Dictionary with comparison metrics and risk classification
    """
    gia_score = gia_results['aggregate']['overall_score']
    baseline_score = baseline_results['aggregate']['overall_score']
    
    improvement = gia_score - baseline_score
    relative_improvement = improvement / baseline_score if baseline_score > 0 else float('inf')
    
    # Risk classification
    # High risk: GIA significantly better than baseline (>20% relative improvement)
    # Medium risk: GIA moderately better (5-20% improvement)
    # Low risk: GIA not significantly better (<5% improvement)
    
    if relative_improvement > 0.20:
        risk_level = 'high'
    elif relative_improvement > 0.05:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    return {
        'gia_score': float(gia_score),
        'baseline_score': float(baseline_score),
        'absolute_improvement': float(improvement),
        'relative_improvement': float(relative_improvement),
        'risk_level': risk_level
    }
