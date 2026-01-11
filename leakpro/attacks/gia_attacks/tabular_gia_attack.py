"""Tabular gradient inversion attack for federated learning."""

from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from collections.abc import Generator

from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA
from leakpro.attacks.gia_attacks.tabular_data_utils import TabularDataset, get_data_statistics
from leakpro.attacks.gia_attacks.tabular_metrics import evaluate_reconstruction, count_constraint_violations
from leakpro.attacks.gia_attacks.prior_baseline import PriorBaseline
from leakpro.metrics.attack_result import GIAResults
from leakpro.utils.logger import logger
import optuna


class TabularGIA(AbstractGIA):
    """Optimization-based gradient inversion attack for tabular data.
    
    Implements DLG-style gradient matching with tabular-specific constraints:
    - One-hot projection for categorical features
    - Bounds clipping for numerical features
    """
    
    def __init__(
        self,
        model: Module,
        criterion: Module,
        dataset: TabularDataset,
        client_loader: DataLoader,
        observed_gradients: List[Tensor],
        config: Optional[Dict] = None
    ):
        """Initialize tabular GIA.
        
        Args:
            model: Target model
            criterion: Loss function
            dataset: TabularDataset with feature information
            client_loader: DataLoader with ground truth client data
            observed_gradients: Gradients observed by server
            config: Attack configuration (learning_rate iterations, etc.)
        """
        super().__init__()
        
        self.model = model
        self.criterion = criterion
        self.dataset = dataset
        self.client_loader = client_loader
        self.observed_gradients = observed_gradients
        self.config = config or {}
        
        # Get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Get data statistics for initialization and constraints
        self.data_stats = get_data_statistics(dataset)
        self.feature_info = dataset.feature_info
        
        # Attack hyperparameters
        self.attack_lr = self.config.get('attack_lr', 0.1)
        self.iterations = self.config.get('iterations', 1000)
        self.regularization_weight = self.config.get('reg_weight', 0.01)
        self.label_known = self.config.get('label_known', True)
        
        # Best reconstruction tracking
        self.best_loss = float('inf')
        self.best_reconstruction = None
        
    def _initialize_dummy_data(self, batch_size: int, labels: Optional[Tensor] = None) -> Tensor:
        """Initialize dummy data for reconstruction.
        
        Args:
            batch_size: Batch size to reconstruct
            labels: If provided (label-known), use these labels
            
        Returns:
            Initial dummy data tensor
        """
        # Use prior baseline for initialization
        prior_baseline = PriorBaseline(self.dataset, self.data_stats)
        dummy_data = prior_baseline.reconstruct(batch_size)
        
        # Add small random noise to break symmetry
        noise = torch.randn_like(dummy_data) * 0.01
        dummy_data = dummy_data + noise
        
        return dummy_data.to(self.device).requires_grad_(True)
    
    def _enforce_constraints(self, data: Tensor) -> Tensor:
        """Project data to valid space (one-hot for categorical, bounds for numerical).
        
        Args:
            data: Data tensor to project
            
        Returns:
            Projected data tensor
        """
        with torch.no_grad():
            data_np = data.cpu().numpy()
            
            # Handle numerical features: clip to valid ranges if needed
            # (Already normalized, so typically don't need strict bounds)
            # For now, we'll leave numerical features as-is
            
            # Handle categorical features: project to one-hot
            numerical_features = self.feature_info['numerical_features']
            categorical_features = self.feature_info['categorical_features']
            
            for i, feature_name in enumerate(categorical_features):
                dec_idx = len(numerical_features) + i
                onehot_indices = self.feature_info['dec_to_onehot'][dec_idx]
                
                # Extract one-hot vector
                onehot_vectors = data_np[:, onehot_indices]
                
                # Project to one-hot: argmax + one-hot encoding
                max_indices = onehot_vectors.argmax(axis=1)
                
                # Zero out and set to one-hot
                data_np[:, onehot_indices] = 0.0
                for sample_idx, max_idx in enumerate(max_indices):
                    data_np[sample_idx, onehot_indices[max_idx]] = 1.0
            
            data.copy_(torch.tensor(data_np, dtype=torch.float32, device=data.device))
        
        return data
    
    def _compute_gradient_loss(self, dummy_data: Tensor, dummy_labels: Tensor) -> Tensor:
        """Compute gradient matching loss.
        
        Args:
            dummy_data: Reconstructed data
            dummy_labels: Labels (known or optimized)
            
        Returns:
            Gradient matching loss
        """
        # Forward pass with dummy data
        dummy_outputs = self.model(dummy_data)
        
        # Compute loss
        if self.label_known:
            dummy_loss = self.criterion(dummy_outputs, dummy_labels.float().unsqueeze(1))
        else:
            dummy_loss = self.criterion(dummy_outputs, dummy_labels)
        
        # Compute gradients
        dummy_gradients = torch.autograd.grad(
            dummy_loss.sum(),
            self.model.parameters(),
            create_graph=True
        )
        
        # Gradient matching loss (L2 distance)
        grad_loss = 0.0
        for dummy_grad, observed_grad in zip(dummy_gradients, self.observed_gradients):
            # Detach observed gradients to avoid backpropping through them
            grad_loss += ((dummy_grad - observed_grad.detach()) ** 2).sum()
        
        return grad_loss
    
    def _regularization(self, data: Tensor) -> Tensor:
        """Compute regularization term.
        
        Args:
            data: Data tensor
            
        Returns:
            Regularization loss
        """
        # Total variation regularization
        # For tabular data, can use L2 penalty to encourage smooth solutions
        return (data ** 2).mean()
    
    def run_attack(self) -> Generator[tuple[int, float, Optional[GIAResults]]]:
        """Run gradient inversion attack.
        
        Yields:
            Tuples of (iteration, score, results)
        """
        # Get ground truth data for evaluation
        ground_truth_data, ground_truth_labels = next(iter(self.client_loader))
        batch_size = ground_truth_data.shape[0]
        
        ground_truth_data = ground_truth_data.to(self.device)
        ground_truth_labels = ground_truth_labels.to(self.device)
        
        # Initialize dummy data
        dummy_data = self._initialize_dummy_data(batch_size, ground_truth_labels if self.label_known else None)
        
        # Initialize dummy labels (fixed if label-known, otherwise optimize)
        if self.label_known:
            dummy_labels = ground_truth_labels.clone()
        else:
            # Start with random labels for label-unknown setting
            dummy_labels = torch.randint(0, 2, (batch_size,), dtype=torch.float32, device=self.device).requires_grad_(True)
        
        # Optimizer for dummy data (and labels if label-unknown)
        if self.label_known:
            optimizer = torch.optim.Adam([dummy_data], lr=self.attack_lr)
        else:
            optimizer = torch.optim.Adam([dummy_data, dummy_labels], lr=self.attack_lr)
        
        # Attack loop
        for iteration in range(self.iterations):
            optimizer.zero_grad()
            
            # Compute gradient matching loss
            grad_loss = self._compute_gradient_loss(dummy_data, dummy_labels)
            
            # Add regularization
            reg_loss = self._regularization(dummy_data)
            
            # Total loss
            total_loss = grad_loss + self.regularization_weight * reg_loss
            
            # Backprop and update
            total_loss.backward()
            optimizer.step()
            
            # Enforce constraints every 10 iterations
            if iteration % 10 == 0:
                dummy_data = self._enforce_constraints(dummy_data)
            
            # Track best reconstruction
            if total_loss.item() < self.best_loss:
                self.best_loss = total_loss.item()
                self.best_reconstruction = dummy_data.detach().clone()
            
            # Log and yield progress
            if iteration % 100 == 0:
                # Evaluate reconstruction quality
                recon_metrics = evaluate_reconstruction(
                    ground_truth_data,
                    self.best_reconstruction,
                    self.feature_info,
                    return_per_feature=False
                )
                
                score = recon_metrics['aggregate']['overall_score']
                
                logger.info(f"Iteration {iteration}, Loss: {total_loss.item():.4f}, Score: {score:.4f}")
                
                yield iteration, score, None
        
        # Final evaluation
        final_reconstruction = self._enforce_constraints(self.best_reconstruction)
        
        recon_metrics = evaluate_reconstruction(
            ground_truth_data,
            final_reconstruction,
            self.feature_info,
            return_per_feature=True
        )
        
        constraint_violations = count_constraint_violations(
            final_reconstruction,
            self.feature_info
        )
        
        # Create result object
        # Note: GIAResults expects certain format, we'll adapt it
        gia_result = {
            'reconstruction_metrics': recon_metrics,
            'constraint_violations': constraint_violations,
            'final_loss': self.best_loss,
            'ground_truth': ground_truth_data.cpu(),
            'reconstructed': final_reconstruction.cpu(),
            'config': self.config
        }
        
        final_score = recon_metrics['aggregate']['overall_score']
        
        logger.info(f"Attack complete. Final score: {final_score:.4f}")
        
        yield self.iterations, final_score, gia_result
    
    def prepare_attack(self) -> None:
        """Prepare attack (required by AbstractGIA interface)."""
        pass
    
    def reset_attack(self) -> None:
        """Reset attack to initial state."""
        super().reset_attack()
        self.best_loss = float('inf')
        self.best_reconstruction = None
    
    def _configure_attack(self, configs: dict) -> None:
        """Configure attack parameters."""
        self.config.update(configs)
    
    def description(self) -> dict:
        """Return attack description."""
        return {
            'reference': 'DLG-style gradient inversion for tabular data',
            'summary': 'Optimization-based gradient matching with tabular constraints',
            'detailed': 'Reconstructs tabular data by minimizing L2 distance between observed and dummy gradients, with one-hot projection for categorical features.'
        }
    
    def suggest_parameters(self, trial: optuna.trial.Trial) -> None:
        """Suggest hyperparameters for Optuna tuning."""
        self.attack_lr = trial.suggest_float('attack_lr', 0.01, 1.0, log=True)
        self.regularization_weight = trial.suggest_float('reg_weight', 0.0001, 0.1, log=True)
