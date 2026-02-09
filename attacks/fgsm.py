import torch
import torch.nn as nn
from .base_attack import BaseAttack


class FGSM(BaseAttack):
    """
    FGSM: Single-step attack using sign of the gradient.
    
    Perturbation: x_adv = x + ε · sign(∇_x L(θ, x, y))
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        epsilon: float = 8/255
    ):
        """
        Args:
            model: Target model
            device: Computation device
            epsilon: Perturbation budget (L∞ norm)
        """
        super().__init__(model, device, epsilon)
        self.criterion = nn.CrossEntropyLoss()
    
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate FGSM adversarial examples.
        
        Args:
            images: (B, C, H, W) clean images in [0, 1]
            labels: (B,) ground-truth labels
        
        Returns:
            (B, C, H, W) adversarial images
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        images.requires_grad = True
        
        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # FGSM perturbation: ε · sign(gradient)
        data_grad = images.grad.data
        perturbation = self.epsilon * data_grad.sign()
        
        # Generate adversarial example
        adv_images = images + perturbation
        
        # Clamp to valid range
        adv_images = self._clamp(images.detach(), adv_images.detach())
        
        return adv_images