"""Loss functions."""

from numpy_dl.loss.mse import MSELoss
from numpy_dl.loss.cross_entropy import CrossEntropyLoss, NLLLoss
from numpy_dl.loss.bce import BCELoss, BCEWithLogitsLoss

__all__ = [
    'MSELoss',
    'CrossEntropyLoss',
    'NLLLoss',
    'BCELoss',
    'BCEWithLogitsLoss',
]
