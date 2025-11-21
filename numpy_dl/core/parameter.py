"""Parameter class for trainable tensors."""

from numpy_dl.core.tensor import Tensor


class Parameter(Tensor):
    """
    A Parameter is a Tensor that is automatically registered as a trainable parameter.

    Parameters are special tensors that always require gradients and are intended
    to be optimized during training.
    """

    def __init__(self, data, device=None):
        """
        Initialize a Parameter.

        Args:
            data: Initial parameter values
            device: Device to place parameter on
        """
        super().__init__(data, requires_grad=True, device=device)

    def __repr__(self):
        return f"Parameter({self.data}, device={self.device})"
