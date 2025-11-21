"""Loss functions."""

from numpy_dl.loss.mse import MSELoss
from numpy_dl.loss.cross_entropy import CrossEntropyLoss, NLLLoss
from numpy_dl.loss.bce import BCELoss, BCEWithLogitsLoss
from numpy_dl.loss.multitask import (
    MultiTaskLoss,
    UncertaintyWeighting,
    GradNorm,
    DynamicWeightAverage,
)
from numpy_dl.loss.vae import VAELoss, KLDivergenceLoss

__all__ = [
    'MSELoss',
    'CrossEntropyLoss',
    'NLLLoss',
    'BCELoss',
    'BCEWithLogitsLoss',
    # Multi-task learning
    'MultiTaskLoss',
    'UncertaintyWeighting',
    'GradNorm',
    'DynamicWeightAverage',
    # Variational autoencoders
    'VAELoss',
    'KLDivergenceLoss',
]
