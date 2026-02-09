import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional


def plot_adversarial_examples(
    clean_imgs: torch.Tensor,
    adv_imgs: torch.Tensor,
    perturbations: torch.Tensor,
    labels: List[str],
    save_path: str,
    n_samples: int = 5
) -> None:
    """
    Visualize clean vs adversarial images with perturbations.
    
    Args:
        clean_imgs: (N, C, H, W) clean images
        adv_imgs: (N, C, H, W) adversarial images
        perturbations: (N, C, H, W) perturbation tensors
        labels: List of class labels for each sample
        save_path: Output PNG path
        n_samples: Number of samples to plot
    """
    n = min(n_samples, clean_imgs.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        # Convert to numpy, unnormalize if needed
        clean = clean_imgs[i].cpu().permute(1, 2, 0).numpy()
        adv = adv_imgs[i].cpu().permute(1, 2, 0).numpy()
        pert = perturbations[i].cpu().permute(1, 2, 0).numpy()
        
        # Clip to [0, 1] range
        clean = np.clip(clean, 0, 1)
        adv = np.clip(adv, 0, 1)
        
        # Plot
        axes[i, 0].imshow(clean)
        axes[i, 0].set_title(f"Clean ({labels[i]})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(adv)
        axes[i, 1].set_title(f"Adversarial")
        axes[i, 1].axis('off')
        
        # Perturbation heatmap
        pert_mag = np.linalg.norm(pert, axis=2)
        im = axes[i, 2].imshow(pert_mag, cmap='hot')
        axes[i, 2].set_title(f"Perturbation")
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_robustness_curve(
    epsilons: List[float],
    clean_accs: List[float],
    adv_accs_dict: dict,  # {attack_name: [acc1, acc2, ...]}
    save_path: str,
    title: str = "Robustness vs Perturbation Budget"
) -> None:
    """
    Plot accuracy vs epsilon for multiple attacks.
    
    Args:
        epsilons: List of epsilon values
        clean_accs: Clean accuracy (repeated for each epsilon)
        adv_accs_dict: Dict mapping attack names to accuracy lists
        save_path: Output PNG path
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Clean accuracy baseline
    ax.plot(epsilons, clean_accs, 'k--', linewidth=2, label='Clean', marker='o')
    
    # Adversarial accuracies
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    for idx, (attack_name, accs) in enumerate(adv_accs_dict.items()):
        ax.plot(epsilons, accs, linewidth=2, label=attack_name,
                marker='s', color=colors[idx % len(colors)])
    
    ax.set_xlabel('Perturbation Budget (ε)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_defense_comparison(
    defense_names: List[str],
    clean_accs: List[float],
    robust_accs: List[float],
    save_path: str
) -> None:
    """
    Bar chart comparing clean vs robust accuracy across defenses.
    
    Args:
        defense_names: List of defense method names
        clean_accs: Clean accuracies
        robust_accs: Robust accuracies (under attack)
        save_path: Output PNG path
    """
    x = np.arange(len(defense_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, clean_accs, width, label='Clean Acc.', color='#3498DB', edgecolor='black')
    ax.bar(x + width/2, robust_accs, width, label='Robust Acc.', color='#E74C3C', edgecolor='black')
    
    ax.set_xlabel('Defense Method', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Clean vs Robust Accuracy by Defense', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(defense_names, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()