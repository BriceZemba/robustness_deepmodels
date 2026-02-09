# Adversarial Robustness of Deep Neural Networks

**A modular PyTorch framework for evaluating and defending against adversarial attacks**

---

## Overview

This project provides a **complete, reproducible framework** for studying the adversarial robustness of deep learning models. It implements:

- **3 adversarial attacks**: FGSM, PGD, Carlini-Wagner L2
- **3 defense mechanisms**: Adversarial Training, Gradient Regularization, Randomized Smoothing
- **2 model architectures**: SimpleCNN, ResNet-18 (CIFAR-adapted)
- **Full evaluation pipeline**: Clean accuracy, robust accuracy, robustness curves

The codebase is designed for **doctoral-level research** with emphasis on:
- Clean, modular architecture
- Full reproducibility (fixed seeds, YAML configs)
- Extensibility (easy to add new attacks/defenses)
- Publication-ready outputs (plots, tables, metrics)

---

## Project Structure

```
robustness_dl/
├── configs/               # YAML configuration files
│   ├── base.yaml          # Baseline clean training
│   ├── attack_pgd.yaml    # PGD attack config
│   └── defense_adv_training.yaml  # Adversarial training config
├── data/                  # Data loading and preprocessing
│   └── datamodule.py      # CIFAR-10 / MNIST loaders
├── models/                # Neural network architectures
│   ├── cnn.py             # Standard 4-layer CNN
│   └── resnet.py          # ResNet-18 (CIFAR-adapted)
├── attacks/               # Adversarial attack implementations
│   ├── base_attack.py     # Abstract base class
│   ├── fgsm.py            # Fast Gradient Sign Method
│   ├── pgd.py             # Projected Gradient Descent
│   └── cw.py              # Carlini-Wagner L2
│   ├── adversarial_training.py
│   ├── smoothing.py
│   └── regularization.py
├── training/              # Training and evaluation
│   ├── trainer.py         # Unified trainer (clean + adversarial)
│   └── evaluator.py       # Attack evaluation
├── metrics/               # Evaluation metrics (TODO)
├── utils/                 # Utilities
│   ├── seed.py            # Reproducibility
│   ├── logger.py          # Structured logging
│   └── visualization.py   # Plot generation         # 
├── requirements.txt       
└── README.md              
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/BriceZemba/robustness_deepmodels.git
cd robustness_deepmodels
pip install -r requirements.txt
```

---

## Key Results

| Model | Clean Acc. | FGSM (ε=8/255) | PGD-10 (ε=8/255) | CW-L2 |
|-------|-----------|----------------|------------------|-------|
| Baseline CNN | 96.4% | 68.5% | 63.0% | — |
| Adversarial Training | 95.3% | **84.5%** | **78.2%** | — |

*Results on MNIST proxy dataset (digits). Full CIFAR-10 results in technical report.*


## Attack Implementations

### FGSM (Fast Gradient Sign Method)
```
x_adv = x + ε · sign(∇_x L(θ, x, y))
```
Single-step attack using the sign of the gradient.

### PGD (Projected Gradient Descent)
```
x_{t+1} = Π_{||·||∞≤ε} (x_t + α · sign(∇_x L(θ, x_t, y)))
```
Multi-step iterative attack with random initialization (strongest first-order attack).

### C&W L2
```
minimize ||δ||_2 + c · f(x + δ)
```
Optimization-based attack minimizing L2 perturbation while changing prediction.



## Defense Mechanisms

### Adversarial Training
Train on mix of clean and adversarial examples generated on-the-fly:
```
L = 0.5 · L(θ, x, y) + 0.5 · L(θ, x_adv, y)
```

### Gradient Regularization
Penalize large input gradients to reduce sensitivity:
```
L = L_task + λ · ||∇_x f(x)||^2
```

### Randomized Smoothing
Certifiable defense via Gaussian noise injection at test time.


## Evaluation Metrics

- **Clean Accuracy**: Standard test-set performance
- **Robust Accuracy**: Accuracy under attack at fixed ε
- **Robustness Curve**: Accuracy vs perturbation budget (ε)
- **Attack Success Rate**: Fraction of samples misclassified by attack
- **Perturbation Magnitude**: Mean L2/L∞ norm of successful attacks


## Reproducibility

All experiments are fully reproducible:

1. **Fixed seeds**: All RNGs (Python, NumPy, PyTorch) locked via `utils/seed.py`
2. **YAML configs**: Every hyperparameter logged
3. **Deterministic CUDA**: `torch.backends.cudnn.deterministic = True`
4. **Versioned dependencies**: `requirements.txt` with pinned versions


## References

1. Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
2. Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
3. Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks", IEEE S&P 2017

---


## License

MIT License (see LICENSE file)
