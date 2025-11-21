"""Cross Entropy loss."""

import numpy as np
from numpy_dl.core.module import Module
from numpy_dl.core.tensor import Tensor
from numpy_dl.core import functional as F
from numpy_dl.utils.device import get_array_module


class CrossEntropyLoss(Module):
    """
    Cross Entropy loss.

    Combines LogSoftmax and NLLLoss in one class.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Initialize CrossEntropy loss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute cross entropy loss.

        Args:
            pred: Predicted logits of shape (batch_size, num_classes)
            target: Target class indices of shape (batch_size,)

        Returns:
            Loss value
        """
        xp = get_array_module(pred.data)

        # Apply log-softmax
        log_probs = F.log_softmax(pred, axis=-1)

        # Get log probabilities for target classes
        batch_size = pred.shape[0]
        if isinstance(target, Tensor):
            target_data = target.data.astype(int)
        else:
            target_data = xp.asarray(target, dtype=int)

        # Gather log probs for target classes
        batch_indices = xp.arange(batch_size)
        loss_data = -log_probs.data[batch_indices, target_data]

        loss = Tensor(
            loss_data,
            requires_grad=pred.requires_grad,
            device=pred.device,
            _children=(pred,),
            _op='cross_entropy'
        )

        def _backward():
            if pred.requires_grad:
                # Gradient of cross entropy
                grad = F.softmax(pred, axis=-1).data
                grad[batch_indices, target_data] -= 1

                if self.reduction == 'mean':
                    grad = grad / batch_size

                pred.grad = grad if pred.grad is None else pred.grad + grad

        loss._backward = _backward

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def __repr__(self):
        return f"CrossEntropyLoss(reduction='{self.reduction}')"


class NLLLoss(Module):
    """
    Negative Log Likelihood loss.

    Assumes input is already log-probabilities.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Initialize NLL loss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, log_probs: Tensor, target: Tensor) -> Tensor:
        """
        Compute NLL loss.

        Args:
            log_probs: Log probabilities of shape (batch_size, num_classes)
            target: Target class indices of shape (batch_size,)

        Returns:
            Loss value
        """
        xp = get_array_module(log_probs.data)

        batch_size = log_probs.shape[0]
        if isinstance(target, Tensor):
            target_data = target.data.astype(int)
        else:
            target_data = xp.asarray(target, dtype=int)

        # Gather log probs for target classes
        batch_indices = xp.arange(batch_size)
        loss_data = -log_probs.data[batch_indices, target_data]

        loss = Tensor(
            loss_data,
            requires_grad=log_probs.requires_grad,
            device=log_probs.device,
            _children=(log_probs,),
            _op='nll'
        )

        def _backward():
            if log_probs.requires_grad:
                grad = xp.zeros_like(log_probs.data)
                grad[batch_indices, target_data] = -1

                if self.reduction == 'mean':
                    grad = grad / batch_size

                log_probs.grad = grad if log_probs.grad is None else log_probs.grad + grad

        loss._backward = _backward

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def __repr__(self):
        return f"NLLLoss(reduction='{self.reduction}')"
