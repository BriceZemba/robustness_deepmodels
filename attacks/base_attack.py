"""
attacks/base_attack.py
======================
Abstract base class for adversarial attacks.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple


class BaseAttack(ABC):
    """
    Abstract interface for adversarial attack methods.
    All attacks must implement the `generate()` method.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        epsilon: float = 8/255
    ):
        """
        Args:
            model: Target model to attack
            device: Computation device (CPU or CUDA)
            epsilon: Maximum perturbation budget (L∞ norm)
        """
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.model.eval()  # Attacks always run in eval mode
    
    @abstractmethod
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate adversarial examples for a batch of images.
        
        Args:
            images: (B, C, H, W) clean images
            labels: (B,) ground-truth labels
        
        Returns:
            (B, C, H, W) adversarial images, clipped to valid range
        """
        raise NotImplementedError
    
    def _clamp(
        self,
        images: torch.Tensor,
        adv_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Clamp adversarial images to:
          1. L∞ ball around clean images (radius epsilon)
          2. Valid pixel range [0, 1]
        
        Args:
            images: Clean images
            adv_images: Perturbed images (before clamping)
        
        Returns:
            Clamped adversarial images
        """
        # L∞ constraint
        perturbation = torch.clamp(
            adv_images - images,
            min=-self.epsilon,
            max=self.epsilon
        )
        adv_images = images + perturbation
        
        # Pixel range constraint
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images