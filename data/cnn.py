"""
models/cnn.py
=============
Standard CNN architecture for CIFAR-10 / MNIST.
"""
import torch
import torch.nn as nn
from typing import Tuple


class SimpleCNN(nn.Module):
    """
    4-layer CNN with batch normalization and dropout.
    Architecture: Conv → BN → ReLU → MaxPool × 2 → FC × 2
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        num_classes: int = 10,
        dropout: float = 0.3
    ):
        """
        Args:
            input_shape: (C, H, W) input tensor shape
            num_classes: Number of output classes
            dropout: Dropout probability for FC layers
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate FC input size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self._forward_conv(dummy)
            fc_input_size = conv_out.view(1, -1).size(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers only."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) input tensor
        
        Returns:
            (B, num_classes) logits
        """
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x