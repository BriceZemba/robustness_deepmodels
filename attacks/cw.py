"""
Carlini-Wagner L2 attack (simplified)
Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks", IEEE S&P 2017
"""
import torch
import torch.nn as nn
from .base_attack import BaseAttack


class CW_L2(BaseAttack):
    """
    Simplified C&W L2 attack via gradient-based optimization.
    
    Objective: minimize ||δ||_2 + c · f(x + δ)
    where f(·) is a margin-based loss encouraging misclassification.
    
    Note: This is a simplified implementation focused on untargeted attacks.
    Production-grade C&W would use binary search over c and tanh reparametrization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        epsilon: float = 8/255,
        c: float = 1.0,
        kappa: float = 0.0,
        num_steps: int = 50,
        lr: float = 0.01
    ):
        """
        Args:
            model: Target model
            device: Computation device
            epsilon: Maximum L∞ perturbation (for final projection)
            c: Weight for adversarial loss
            kappa: Confidence margin (how far to push misclassification)
            num_steps: Number of optimization steps
            lr: Learning rate for perturbation optimization
        """
        super().__init__(model, device, epsilon)
        self.c = c
        self.kappa = kappa
        self.num_steps = num_steps
        self.lr = lr
    
    def _cw_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        C&W loss: maximize(max(Z_other) - Z_target + κ, 0)
        
        Args:
            outputs: (B, K) model logits
            labels: (B,) ground-truth labels
        
        Returns:
            Scalar loss
        """
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)
        
        # One-hot encode labels
        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Logit for true class
        true_logits = (outputs * one_hot).sum(1)
        
        # Max logit among other classes
        other_logits = (outputs * (1 - one_hot) - one_hot * 1e9).max(1)[0]
        
        # Margin loss (untargeted: want other_logits > true_logits + kappa)
        loss = torch.clamp(true_logits - other_logits + self.kappa, min=0)
        
        return loss.mean()
    
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate C&W adversarial examples via gradient descent.
        
        Args:
            images: (B, C, H, W) clean images in [0, 1]
            labels: (B,) ground-truth labels
        
        Returns:
            (B, C, H, W) adversarial images
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # Initialize perturbation
        delta = torch.zeros_like(images, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)
        
        for step in range(self.num_steps):
            adv_images = images + delta
            adv_images = torch.clamp(adv_images, 0, 1)
            
            # Forward pass
            outputs = self.model(adv_images)
            
            # Combined loss: L2 norm + adversarial margin
            l2_loss = delta.pow(2).sum()
            adv_loss = self._cw_loss(outputs, labels)
            total_loss = l2_loss + self.c * adv_loss
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Project delta to L∞ ball
            with torch.no_grad():
                delta.clamp_(-self.epsilon, self.epsilon)
        
        # Final adversarial images
        adv_images = images + delta.detach()
        adv_images = self._clamp(images, adv_images)
        
        return adv_images