"""Simple MLP models for tabular data gradient inversion experiments."""

import torch
import torch.nn as nn


class AdultNet1Layer(nn.Module):
    """Single hidden layer MLP for Adult dataset."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_classes: int = 1):
        """Initialize 1-layer model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_classes: Number of output classes (1 for binary classification with BCEWithLogitsLoss)
        """
        super(AdultNet1Layer, self).__init__()
        self.init_params = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_classes": num_classes,
            "architecture": "1layer"
        }
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class AdultNet2Layer(nn.Module):
    """Two hidden layer MLP for Adult dataset."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_classes: int = 1):
        """Initialize 2-layer model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in each layer
            num_classes: Number of output classes (1 for binary classification with BCEWithLogitsLoss)
        """
        super(AdultNet2Layer, self).__init__()
        self.init_params = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_classes": num_classes,
            "architecture": "2layer"
        }
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


def get_model(architecture: str, input_size: int, hidden_size: int = 64, num_classes: int = 1) -> nn.Module:
    """Factory function to get model by architecture name.
    
    Args:
        architecture: Either "1layer" or "2layer"
        input_size: Number of input features
        hidden_size: Number of hidden units
        num_classes: Number of output classes
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If architecture is not recognized
    """
    if architecture == "1layer":
        return AdultNet1Layer(input_size, hidden_size, num_classes)
    elif architecture == "2layer":
        return AdultNet2Layer(input_size, hidden_size, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose '1layer' or '2layer'.")
