"""Test utilities and helper functions."""

import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numpy_dl.core.tensor import Tensor
from numpy_dl.core.parameter import Parameter


def assert_tensors_close(t1: Tensor, t2: Tensor, rtol: float = 1e-5, atol: float = 1e-8):
    """
    Assert two tensors are approximately equal.

    Args:
        t1: First tensor
        t2: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    data1 = t1.numpy() if isinstance(t1, Tensor) else t1
    data2 = t2.numpy() if isinstance(t2, Tensor) else t2

    np.testing.assert_allclose(data1, data2, rtol=rtol, atol=atol)


def assert_arrays_close(a1, a2, rtol: float = 1e-5, atol: float = 1e-8):
    """Assert two arrays are approximately equal."""
    np.testing.assert_allclose(a1, a2, rtol=rtol, atol=atol)


def numerical_gradient(func, x, eps=1e-5):
    """
    Compute numerical gradient using finite differences.

    Args:
        func: Function that takes x and returns scalar
        x: Input array
        eps: Small perturbation

    Returns:
        Numerical gradient
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        # f(x + h)
        x[idx] = old_value + eps
        fxh_plus = func(x)

        # f(x - h)
        x[idx] = old_value - eps
        fxh_minus = func(x)

        # Restore value
        x[idx] = old_value

        # Compute gradient
        grad[idx] = (fxh_plus - fxh_minus) / (2 * eps)
        it.iternext()

    return grad


def check_gradient(func, x, analytical_grad, eps=1e-5, rtol=1e-3, atol=1e-5):
    """
    Check if analytical gradient matches numerical gradient.

    Args:
        func: Function that takes x and returns scalar
        x: Input array
        analytical_grad: Computed analytical gradient
        eps: Perturbation for numerical gradient
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if gradients match
    """
    numerical_grad = numerical_gradient(func, x, eps)

    try:
        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        print(f"Gradient check failed!")
        print(f"Analytical: {analytical_grad}")
        print(f"Numerical: {numerical_grad}")
        print(f"Difference: {np.abs(analytical_grad - numerical_grad)}")
        return False


def create_simple_dataset(n_samples=100, n_features=10, n_classes=2, seed=42):
    """
    Create a simple synthetic dataset for testing.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        seed: Random seed

    Returns:
        (X, y) tuple
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples).astype(np.int64)
    return X, y


def create_regression_dataset(n_samples=100, n_features=10, noise=0.1, seed=42):
    """
    Create a simple regression dataset for testing.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level
        seed: Random seed

    Returns:
        (X, y) tuple
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    true_weights = np.random.randn(n_features).astype(np.float32)
    y = X @ true_weights + noise * np.random.randn(n_samples).astype(np.float32)
    return X, y


def create_sequence_dataset(n_samples=100, seq_len=10, vocab_size=50, seed=42):
    """
    Create a simple sequence dataset for testing.

    Args:
        n_samples: Number of samples
        seq_len: Sequence length
        vocab_size: Vocabulary size
        seed: Random seed

    Returns:
        Sequence data
    """
    np.random.seed(seed)
    return np.random.randint(0, vocab_size, (n_samples, seq_len))


def count_parameters(module):
    """
    Count trainable parameters in a module.

    Args:
        module: Module instance

    Returns:
        Number of parameters
    """
    return sum(p.data.size for p in module.parameters())


class TemporaryDirectory:
    """Context manager for temporary directory."""

    def __init__(self, prefix='test_'):
        self.prefix = prefix
        self.path = None

    def __enter__(self):
        import tempfile
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix))
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        import shutil
        if self.path and self.path.exists():
            shutil.rmtree(self.path)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
