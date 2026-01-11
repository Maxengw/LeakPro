"""Prior-only baseline attack for tabular data (no gradient information)."""

from typing import Dict

import numpy as np
import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.tabular_data_utils import TabularDataset


class PriorBaseline:
    """Baseline attack that reconstructs data using only prior knowledge (no gradients).
    
    This establishes a "low-risk" threshold for gradient inversion attacks.
    If GIA cannot beat this baseline, the setting is considered low-risk.
    """
    
    def __init__(self, dataset: TabularDataset, data_stats: Dict):
        """Initialize prior baseline.
        
        Args:
            dataset: TabularDataset with feature information
            data_stats: Dictionary with data statistics (from get_data_statistics)
        """
        self.dataset = dataset
        self.data_stats = data_stats
        self.feature_info = dataset.feature_info
        
    def reconstruct(
        self,
        batch_size: int,
        labels: Tensor = None,
        random_seed: int = None
    ) -> Tensor:
        """Reconstruct data using only prior distributions.
        
        Args:
            batch_size: Number of samples to reconstruct
            labels: If provided (label-known setting), can use class-conditional priors
            random_seed: Random seed for reproducibility
            
        Returns:
            Reconstructed data tensor of shape (batch_size, num_features)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        reconstructed = np.zeros((batch_size, self.dataset.num_features))
        
        # Sample numerical features from Gaussian
        numerical_features = self.feature_info['numerical_features']
        for i, feature_name in enumerate(numerical_features):
            feature_indices = self.feature_info['dec_to_onehot'][i]
            feature_idx = feature_indices[0]
            
            # Get statistics for this feature
            mean = self.data_stats['x_mean'][feature_idx]
            std = self.data_stats['x_std'][feature_idx]
            
            # Sample from normal distribution
            reconstructed[:, feature_idx] = np.random.normal(mean, std, batch_size)
        
        # Sample categorical features from empirical distributions
        categorical_features = self.feature_info['categorical_features']
        for i, feature_name in enumerate(categorical_features):
            dec_idx = len(numerical_features) + i
            onehot_indices = self.feature_info['dec_to_onehot'][dec_idx]
            
            # Get category frequencies
            frequencies = self.data_stats['categorical_frequencies'][feature_name]
            
            # Sample categories according to frequencies
            sampled_categories = np.random.choice(
                len(frequencies),
                size=batch_size,
                p=frequencies / frequencies.sum()
            )
            
            # Convert to one-hot
            for sample_idx, category_idx in enumerate(sampled_categories):
                reconstructed[sample_idx, onehot_indices[category_idx]] = 1.0
        
        return torch.tensor(reconstructed, dtype=torch.float32)
    
    def reconstruct_batch(
        self,
        ground_truth: Tensor,
        labels: Tensor = None
    ) -> Tensor:
        """Reconstruct a batch matching the shape of ground truth.
        
        Args:
            ground_truth: Ground truth batch (used only for batch size)
            labels: Optional labels for class-conditional sampling
            
        Returns:
            Reconstructed batch
        """
        batch_size = ground_truth.shape[0]
        return self.reconstruct(batch_size, labels)
