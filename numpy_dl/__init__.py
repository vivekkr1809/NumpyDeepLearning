"""
NumPy Deep Learning Framework

A deep learning framework built from scratch using NumPy, supporting
MLP, CNN, RNN, U-Net, and ResNet architectures with CPU/GPU acceleration.
"""

__version__ = '0.1.0'

# Core components
from numpy_dl.core import (
    Tensor, tensor, zeros, ones, randn, rand,
    Parameter, Module, Sequential, ModuleList, F
)

# Neural network layers
from numpy_dl import nn

# Loss functions
from numpy_dl import loss

# Optimizers
from numpy_dl import optim

# Pre-built models
from numpy_dl import models

# Data utilities
from numpy_dl import data

# Utilities
from numpy_dl import utils

# Experiment tracking
from numpy_dl import tracking

__all__ = [
    # Core
    'Tensor',
    'tensor',
    'zeros',
    'ones',
    'randn',
    'rand',
    'Parameter',
    'Module',
    'Sequential',
    'ModuleList',
    'F',
    # Modules
    'nn',
    'loss',
    'optim',
    'models',
    'data',
    'utils',
    'tracking',
]
