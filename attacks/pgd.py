import torch
import torch.nn as nn
from .base_attack import BaseAttack


class PGD(BaseAttack):
    """
    PGD: Multi-step iterative attack with projection.
    
    Initialization: x_0 = x + uniform[-ε, ε]
    Iteration: x_{t+1} = Π_{||·||∞≤ε} (x_t + α · sign(∇_x L(θ, x_t, y)))
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_steps: int = 10,
        random_start: bool = True
    ):
        """
        Args:
            model: Target model
            device: Computation device
            epsilon: Perturbation budget (L∞ norm)
            alpha: Step size per iteration
            num_steps: Number of PGD iterations
            random_start: Whether to start from random perturbation
        """
        super().__init__(model, device, epsilon)
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        self.criterion = nn.CrossEntropyLoss()
    
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate PGD adversarial examples.
        
        Args:
            images: (B, C, H, W) clean images in [0, 1]
            labels: (B,) ground-truth labels
        
        Returns:
            (B, C, H, W) adversarial images
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # Random initialization
        if self.random_start:
            perturbation = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            adv_images = images + perturbation
            adv_images = torch.clamp(adv_images, 0, 1)
        else:
            adv_images = images.clone()
        
        # PGD iterations
        for _ in range(self.num_steps):
            adv_images.requires_grad = True
            
            # Forward + backward
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, labels)
            self.model.zero_grad()
            loss.backward()
            
            # Gradient step
            data_grad = adv_images.grad.data
            adv_images = adv_images.detach() + self.alpha * data_grad.sign()
            
            # Project back to valid region
            adv_images = self._clamp(images, adv_images)
        
        return adv_images