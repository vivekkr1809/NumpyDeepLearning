"""Mean Squared Error loss."""

from numpy_dl.core.module import Module
from numpy_dl.core.tensor import Tensor


class MSELoss(Module):
    """
    Mean Squared Error loss.

    Computes the mean squared error between predictions and targets.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Initialize MSE loss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute MSE loss.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Loss value
        """
        loss = (pred - target) ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def __repr__(self):
        return f"MSELoss(reduction='{self.reduction}')"
