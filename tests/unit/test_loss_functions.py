"""Unit tests for loss functions."""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from numpy_dl.core.tensor import Tensor
from numpy_dl.loss import (
    MSELoss, CrossEntropyLoss, NLLLoss,
    BCELoss, BCEWithLogitsLoss,
    VAELoss, KLDivergenceLoss
)
from tests.test_utils import assert_arrays_close, assert_tensors_close


class TestMSELoss(unittest.TestCase):
    """Test Mean Squared Error loss."""

    def test_mse_creation(self):
        """Test creating MSE loss."""
        loss_fn = MSELoss()
        self.assertEqual(loss_fn.reduction, 'mean')

    def test_mse_forward(self):
        """Test MSE forward pass."""
        loss_fn = MSELoss()
        pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = Tensor([[1.5, 2.5], [3.5, 4.5]])

        loss = loss_fn(pred, target)

        # MSE = mean((pred - target)^2) = mean(0.25) = 0.25
        expected = 0.25
        self.assertAlmostEqual(float(loss.data), expected, places=5)

    def test_mse_backward(self):
        """Test MSE backward pass."""
        loss_fn = MSELoss()
        pred = Tensor([[1.0, 2.0]], requires_grad=True)
        target = Tensor([[2.0, 3.0]])

        loss = loss_fn(pred, target)
        loss.backward()

        # Gradient: 2 * (pred - target) / n = 2 * [-1, -1] / 2 = [-1, -1]
        expected_grad = np.array([[-1.0, -1.0]])
        assert_arrays_close(pred.grad, expected_grad)

    def test_mse_reduction_sum(self):
        """Test MSE with sum reduction."""
        loss_fn = MSELoss(reduction='sum')
        pred = Tensor([[1.0, 2.0]])
        target = Tensor([[1.5, 2.5]])

        loss = loss_fn(pred, target)

        # Sum of squared errors = 0.25 + 0.25 = 0.5
        expected = 0.5
        self.assertAlmostEqual(float(loss.data), expected, places=5)

    def test_mse_reduction_none(self):
        """Test MSE with no reduction."""
        loss_fn = MSELoss(reduction='none')
        pred = Tensor([[1.0, 2.0]])
        target = Tensor([[1.5, 2.5]])

        loss = loss_fn(pred, target)

        # Element-wise squared errors
        expected = np.array([[0.25, 0.25]])
        assert_arrays_close(loss.data, expected)


class TestCrossEntropyLoss(unittest.TestCase):
    """Test Cross Entropy loss."""

    def test_crossentropy_creation(self):
        """Test creating cross entropy loss."""
        loss_fn = CrossEntropyLoss()
        self.assertEqual(loss_fn.reduction, 'mean')

    def test_crossentropy_forward(self):
        """Test cross entropy forward pass."""
        loss_fn = CrossEntropyLoss()

        # Simple 2-class problem
        logits = Tensor([[2.0, 1.0], [1.0, 3.0]])  # (2, 2)
        targets = Tensor([0, 1])  # Class indices

        loss = loss_fn(logits, targets)

        # Loss should be positive
        self.assertGreater(float(loss.data), 0)

    def test_crossentropy_perfect_prediction(self):
        """Test cross entropy with perfect predictions."""
        loss_fn = CrossEntropyLoss()

        # Perfect prediction (very high logit for correct class)
        logits = Tensor([[100.0, 0.0], [0.0, 100.0]])
        targets = Tensor([0, 1])

        loss = loss_fn(logits, targets)

        # Loss should be very close to 0
        self.assertLess(float(loss.data), 0.1)

    def test_crossentropy_backward(self):
        """Test cross entropy backward pass."""
        loss_fn = CrossEntropyLoss()

        logits = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        targets = Tensor([2])  # Target is class 2

        loss = loss_fn(logits, targets)
        loss.backward()

        # Gradient should exist
        self.assertIsNotNone(logits.grad)
        # Gradient should sum to 0 (property of cross entropy gradient)
        self.assertAlmostEqual(float(np.sum(logits.grad)), 0.0, places=5)

    def test_crossentropy_multiclass(self):
        """Test cross entropy with multiple classes."""
        loss_fn = CrossEntropyLoss()

        # 10 classes
        logits = Tensor(np.random.randn(5, 10))
        targets = Tensor([0, 3, 5, 7, 9])

        loss = loss_fn(logits, targets)

        # Loss should be reasonable (not NaN or Inf)
        self.assertFalse(np.isnan(float(loss.data)))
        self.assertFalse(np.isinf(float(loss.data)))
        self.assertGreater(float(loss.data), 0)


class TestNLLLoss(unittest.TestCase):
    """Test Negative Log Likelihood loss."""

    def test_nll_creation(self):
        """Test creating NLL loss."""
        loss_fn = NLLLoss()
        self.assertEqual(loss_fn.reduction, 'mean')

    def test_nll_forward(self):
        """Test NLL forward pass."""
        loss_fn = NLLLoss()

        # Log probabilities
        log_probs = Tensor([[-0.5, -1.0], [-1.5, -0.3]])
        targets = Tensor([0, 1])

        loss = loss_fn(log_probs, targets)

        # Loss = -(-0.5 + -0.3) / 2 = 0.4
        expected = 0.4
        self.assertAlmostEqual(float(loss.data), expected, places=5)

    def test_nll_backward(self):
        """Test NLL backward pass."""
        loss_fn = NLLLoss()

        log_probs = Tensor([[-0.5, -1.0]], requires_grad=True)
        targets = Tensor([0])

        loss = loss_fn(log_probs, targets)
        loss.backward()

        # Gradient should be [-1, 0] for target class 0
        expected_grad = np.array([[-1.0, 0.0]])
        assert_arrays_close(log_probs.grad, expected_grad)


class TestBCELoss(unittest.TestCase):
    """Test Binary Cross Entropy loss."""

    def test_bce_creation(self):
        """Test creating BCE loss."""
        loss_fn = BCELoss()
        self.assertEqual(loss_fn.reduction, 'mean')

    def test_bce_forward(self):
        """Test BCE forward pass."""
        loss_fn = BCELoss()

        pred = Tensor([[0.9, 0.1], [0.2, 0.8]])
        target = Tensor([[1.0, 0.0], [0.0, 1.0]])

        loss = loss_fn(pred, target)

        # Loss should be low for good predictions
        self.assertLess(float(loss.data), 0.5)
        self.assertGreater(float(loss.data), 0)

    def test_bce_perfect_prediction(self):
        """Test BCE with perfect predictions."""
        loss_fn = BCELoss()

        pred = Tensor([[0.99, 0.01]])
        target = Tensor([[1.0, 0.0]])

        loss = loss_fn(pred, target)

        # Loss should be very small
        self.assertLess(float(loss.data), 0.1)

    def test_bce_backward(self):
        """Test BCE backward pass."""
        loss_fn = BCELoss()

        pred = Tensor([[0.8, 0.2]], requires_grad=True)
        target = Tensor([[1.0, 0.0]])

        loss = loss_fn(pred, target)
        loss.backward()

        # Gradient should exist
        self.assertIsNotNone(pred.grad)

    def test_bce_numerical_stability(self):
        """Test BCE handles edge cases."""
        loss_fn = BCELoss()

        # Near 0 and 1 predictions (should be clipped)
        pred = Tensor([[0.0001, 0.9999]])
        target = Tensor([[0.0, 1.0]])

        loss = loss_fn(pred, target)

        # Should not produce NaN or Inf
        self.assertFalse(np.isnan(float(loss.data)))
        self.assertFalse(np.isinf(float(loss.data)))


class TestBCEWithLogitsLoss(unittest.TestCase):
    """Test Binary Cross Entropy with Logits loss."""

    def test_bce_logits_creation(self):
        """Test creating BCE with logits loss."""
        loss_fn = BCEWithLogitsLoss()
        self.assertEqual(loss_fn.reduction, 'mean')

    def test_bce_logits_forward(self):
        """Test BCE with logits forward pass."""
        loss_fn = BCEWithLogitsLoss()

        logits = Tensor([[2.0, -1.0], [-0.5, 3.0]])
        target = Tensor([[1.0, 0.0], [0.0, 1.0]])

        loss = loss_fn(logits, target)

        # Loss should be positive
        self.assertGreater(float(loss.data), 0)

    def test_bce_logits_vs_bce(self):
        """Test that BCE with logits is more numerically stable."""
        from numpy_dl.nn import Sigmoid

        sigmoid = Sigmoid()
        bce_fn = BCELoss()
        bce_logits_fn = BCEWithLogitsLoss()

        logits = Tensor([[5.0, -5.0]])
        target = Tensor([[1.0, 0.0]])

        # BCE with manual sigmoid
        pred = sigmoid(logits)
        loss_manual = bce_fn(pred, target)

        # BCE with logits
        loss_combined = bce_logits_fn(logits, target)

        # Both should give similar results
        # But BCE with logits is more stable for extreme values
        self.assertLess(abs(float(loss_manual.data) - float(loss_combined.data)), 0.1)

    def test_bce_logits_backward(self):
        """Test BCE with logits backward pass."""
        loss_fn = BCEWithLogitsLoss()

        logits = Tensor([[1.0, -1.0]], requires_grad=True)
        target = Tensor([[1.0, 0.0]])

        loss = loss_fn(logits, target)
        loss.backward()

        # Gradient should exist
        self.assertIsNotNone(logits.grad)


class TestVAELoss(unittest.TestCase):
    """Test VAE loss."""

    def test_vae_loss_creation(self):
        """Test creating VAE loss."""
        loss_fn = VAELoss()
        self.assertIsNotNone(loss_fn)

    def test_vae_loss_forward(self):
        """Test VAE loss forward pass."""
        loss_fn = VAELoss()

        recon_x = Tensor(np.random.randn(2, 10))
        x = Tensor(np.random.randn(2, 10))
        mu = Tensor(np.random.randn(2, 5))
        logvar = Tensor(np.random.randn(2, 5))

        loss = loss_fn(recon_x, x, mu, logvar)

        # Loss should be positive
        self.assertGreater(float(loss.data), 0)

    def test_vae_loss_backward(self):
        """Test VAE loss backward pass."""
        loss_fn = VAELoss()

        recon_x = Tensor(np.random.randn(2, 10), requires_grad=True)
        x = Tensor(np.random.randn(2, 10))
        mu = Tensor(np.random.randn(2, 5), requires_grad=True)
        logvar = Tensor(np.random.randn(2, 5), requires_grad=True)

        loss = loss_fn(recon_x, x, mu, logvar)
        loss.backward()

        # Gradients should exist
        self.assertIsNotNone(recon_x.grad)
        self.assertIsNotNone(mu.grad)
        self.assertIsNotNone(logvar.grad)


class TestKLDivergenceLoss(unittest.TestCase):
    """Test KL Divergence loss."""

    def test_kl_loss_creation(self):
        """Test creating KL divergence loss."""
        loss_fn = KLDivergenceLoss()
        self.assertIsNotNone(loss_fn)

    def test_kl_loss_forward(self):
        """Test KL divergence forward pass."""
        loss_fn = KLDivergenceLoss()

        mu = Tensor([[0.0, 0.0]])
        logvar = Tensor([[0.0, 0.0]])

        loss = loss_fn(mu, logvar)

        # For mu=0, logvar=0 (var=1), KL should be close to 0
        self.assertLess(float(loss.data), 0.1)

    def test_kl_loss_non_zero(self):
        """Test KL divergence with non-zero mean."""
        loss_fn = KLDivergenceLoss()

        mu = Tensor([[1.0, 1.0]])
        logvar = Tensor([[0.0, 0.0]])

        loss = loss_fn(mu, logvar)

        # KL should be > 0 when mu != 0
        self.assertGreater(float(loss.data), 0)

    def test_kl_loss_backward(self):
        """Test KL divergence backward pass."""
        loss_fn = KLDivergenceLoss()

        mu = Tensor([[1.0, 2.0]], requires_grad=True)
        logvar = Tensor([[0.5, -0.5]], requires_grad=True)

        loss = loss_fn(mu, logvar)
        loss.backward()

        # Gradients should exist
        self.assertIsNotNone(mu.grad)
        self.assertIsNotNone(logvar.grad)


class TestLossComparison(unittest.TestCase):
    """Compare different loss functions."""

    def test_all_losses_handle_gradients(self):
        """Test that all losses properly compute gradients."""
        # Test MSE
        mse = MSELoss()
        pred = Tensor([[1.0, 2.0]], requires_grad=True)
        target = Tensor([[1.5, 2.5]])
        loss = mse(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

        # Test CrossEntropy
        ce = CrossEntropyLoss()
        logits = Tensor([[1.0, 2.0]], requires_grad=True)
        targets = Tensor([1])
        loss = ce(logits, targets)
        loss.backward()
        self.assertIsNotNone(logits.grad)

        # Test BCE
        bce = BCELoss()
        pred = Tensor([[0.7]], requires_grad=True)
        target = Tensor([[1.0]])
        loss = bce(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_reduction_modes(self):
        """Test that all losses support different reduction modes."""
        pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = Tensor([[1.5, 2.5], [3.5, 4.5]])

        # Test MSE reductions
        loss_mean = MSELoss(reduction='mean')(pred, target)
        loss_sum = MSELoss(reduction='sum')(pred, target)
        loss_none = MSELoss(reduction='none')(pred, target)

        self.assertEqual(loss_mean.shape, ())  # Scalar
        self.assertEqual(loss_sum.shape, ())   # Scalar
        self.assertEqual(loss_none.shape, (2, 2))  # Same as input

        # Sum should be larger than mean
        self.assertGreater(float(loss_sum.data), float(loss_mean.data))


if __name__ == '__main__':
    unittest.main()
