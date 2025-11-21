"""Binary Cross Entropy loss."""

import numpy as np
from numpy_dl.core.module import Module
from numpy_dl.core.tensor import Tensor
from numpy_dl.core import functional as F
from numpy_dl.utils.device import get_array_module


class BCELoss(Module):
    """
    Binary Cross Entropy loss.

    Measures the Binary Cross Entropy between the target and the input probabilities.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Initialize BCE loss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute BCE loss.

        Args:
            pred: Predicted probabilities (must be in [0, 1])
            target: Target binary values (0 or 1)

        Returns:
            Loss value
        """
        xp = get_array_module(pred.data)

        # Clip for numerical stability
        eps = 1e-7
        pred_clipped = pred.clip(eps, 1 - eps)

        # BCE = -[target * log(pred) + (1 - target) * log(1 - pred)]
        loss = -(target * pred_clipped.log() + (1 - target) * (1 - pred_clipped).log())

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def __repr__(self):
        return f"BCELoss(reduction='{self.reduction}')"


class BCEWithLogitsLoss(Module):
    """
    Binary Cross Entropy with Logits loss.

    Combines Sigmoid and BCE in one class for numerical stability.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Initialize BCE with logits loss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Compute BCE with logits loss.

        Args:
            logits: Predicted logits (raw outputs)
            target: Target binary values (0 or 1)

        Returns:
            Loss value
        """
        xp = get_array_module(logits.data)

        # Use log-sum-exp trick for numerical stability
        # BCE = max(logits, 0) - logits * target + log(1 + exp(-abs(logits)))
        max_val = xp.maximum(logits.data, 0)
        loss_data = max_val - logits.data * target.data + xp.log(1 + xp.exp(-xp.abs(logits.data)))

        loss = Tensor(
            loss_data,
            requires_grad=logits.requires_grad,
            device=logits.device,
            _children=(logits,),
            _op='bce_with_logits'
        )

        def _backward():
            if logits.requires_grad:
                # Gradient: sigmoid(logits) - target
                sigmoid_val = 1.0 / (1.0 + xp.exp(-logits.data))
                grad = sigmoid_val - target.data

                if self.reduction == 'mean':
                    grad = grad / logits.size
                elif self.reduction == 'sum':
                    pass
                else:
                    pass

                logits.grad = grad if logits.grad is None else logits.grad + grad

        loss._backward = _backward

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def __repr__(self):
        return f"BCEWithLogitsLoss(reduction='{self.reduction}')"
