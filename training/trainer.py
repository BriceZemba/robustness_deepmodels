"""
Model trainer with support for clean and adversarial training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict
from utils.logger import setup_logger

logger = setup_logger("trainer")


class Trainer:
    """
    Unified trainer for clean and adversarial training.
    Supports optional attack augmentation during training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        adversarial_attack: Optional['BaseAttack'] = None
    ):
        """
        Args:
            model: PyTorch model to train
            device: Computation device
            optimizer: Optimizer instance
            scheduler: Optional learning rate scheduler
            adversarial_attack: Optional attack for adversarial training
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.adversarial_attack = adversarial_attack
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
        
        Returns:
            Dict with 'loss' and 'accuracy'
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Adversarial training mode
            if self.adversarial_attack is not None:
                with torch.no_grad():
                    adv_images = self.adversarial_attack.generate(images, labels)
                # Mix clean and adversarial (50/50)
                combined = torch.cat([images, adv_images], dim=0)
                combined_labels = torch.cat([labels, labels], dim=0)
                outputs = self.model(combined)
                loss = self.criterion(outputs, combined_labels)
            else:
                # Clean training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0) * (2 if self.adversarial_attack else 1)
            correct += predicted.eq(
                torch.cat([labels, labels]) if self.adversarial_attack else labels
            ).sum().item()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = 100.0 * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate on clean data.
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            Dict with 'loss' and 'accuracy'
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = total_loss / len(dataloader.dataset)
        val_acc = 100.0 * correct / total
        
        return {'loss': val_loss, 'accuracy': val_acc}
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        patience: int = 10
    ) -> Dict:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            num_epochs: Maximum epochs
            patience: Early stopping patience
        
        Returns:
            Training history dict
        """
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            logger.info(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.2f}%"
            )
            
            # Early stopping
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                # Save best model (optional)
                torch.save(self.model.state_dict(), 'results/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        return history