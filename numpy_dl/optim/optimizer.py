"""Base optimizer class."""

from typing import List, Iterator
from numpy_dl.core.parameter import Parameter


class Optimizer:
    """
    Base class for all optimizers.

    All optimizers should subclass this class and implement the step() method.
    """

    def __init__(self, params: Iterator[Parameter], lr: float):
        """
        Initialize optimizer.

        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate
        """
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        """Zero out all parameter gradients."""
        for param in self.params:
            param.zero_grad()

    def step(self):
        """
        Perform a single optimization step.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement step()")

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr})"
