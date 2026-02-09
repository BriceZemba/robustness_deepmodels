"""
data/datamodule.py
==================
Data loading and preprocessing for CIFAR-10 and MNIST.
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple


class DataModule:
    """
    Unified data interface for CIFAR-10 and MNIST.
    Handles downloading, normalization, augmentation, and splitting.
    """
    
    def __init__(
        self,
        dataset_name: str = 'cifar10',
        data_dir: str = './data',
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: float = 0.1,
        augment: bool = True
    ):
        """
        Args:
            dataset_name: 'cifar10' or 'mnist'
            data_dir: Root directory for dataset storage
            batch_size: Batch size for DataLoaders
            num_workers: Number of parallel data loading workers
            val_split: Fraction of training set to use for validation
            augment: Whether to apply data augmentation to training set
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.augment = augment
        
        # Dataset-specific config
        if self.dataset_name == 'cifar10':
            self.num_classes = 10
            self.input_shape = (3, 32, 32)
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2470, 0.2435, 0.2616)
            self.dataset_class = datasets.CIFAR10
        elif self.dataset_name == 'mnist':
            self.num_classes = 10
            self.input_shape = (1, 28, 28)
            self.mean = (0.1307,)
            self.std = (0.3081,)
            self.dataset_class = datasets.MNIST
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _get_transforms(self, train: bool = True) -> transforms.Compose:
        """Build transform pipeline for train or test set."""
        transform_list = []
        
        if train and self.augment and self.dataset_name == 'cifar10':
            transform_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        return transforms.Compose(transform_list)
    
    def setup(self) -> None:
        """Download and prepare datasets."""
        # Training set
        train_full = self.dataset_class(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self._get_transforms(train=True)
        )
        
        # Split into train and validation
        n_train = int(len(train_full) * (1 - self.val_split))
        n_val = len(train_full) - n_train
        self.train_dataset, self.val_dataset = random_split(
            train_full, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Test set
        self.test_dataset = self.dataset_class(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self._get_transforms(train=False)
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )