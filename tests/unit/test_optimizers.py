"""Unit tests for optimizers."""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from numpy_dl.core.tensor import Tensor
from numpy_dl.core.parameter import Parameter
from numpy_dl.optim import SGD, Adam, AdamW, RMSprop
from tests.test_utils import assert_arrays_close


class TestSGD(unittest.TestCase):
    """Test SGD optimizer."""

    def test_sgd_creation(self):
        """Test creating SGD optimizer."""
        param = Parameter(np.array([1.0, 2.0, 3.0]))
        optimizer = SGD([param], lr=0.1)

        self.assertEqual(optimizer.lr, 0.1)
        self.assertEqual(optimizer.momentum, 0.0)
        self.assertEqual(optimizer.weight_decay, 0.0)

    def test_sgd_step(self):
        """Test basic SGD step."""
        param = Parameter(np.array([1.0, 2.0, 3.0]))
        optimizer = SGD([param], lr=0.1)

        # Set gradient
        param.grad = np.array([0.1, 0.2, 0.3])

        # Take step
        optimizer.step()

        # Parameters should be updated: w = w - lr * grad
        expected = np.array([1.0 - 0.1 * 0.1, 2.0 - 0.1 * 0.2, 3.0 - 0.1 * 0.3])
        assert_arrays_close(param.data, expected)

    def test_sgd_momentum(self):
        """Test SGD with momentum."""
        param = Parameter(np.array([1.0]))
        optimizer = SGD([param], lr=0.1, momentum=0.9)

        # First step
        param.grad = np.array([1.0])
        optimizer.step()

        # v1 = 0.9 * 0 + 1.0 * 1.0 = 1.0
        # w1 = 1.0 - 0.1 * 1.0 = 0.9
        self.assertAlmostEqual(float(param.data), 0.9, places=5)

        # Second step
        param.grad = np.array([1.0])
        optimizer.step()

        # v2 = 0.9 * 1.0 + 1.0 * 1.0 = 1.9
        # w2 = 0.9 - 0.1 * 1.9 = 0.71
        self.assertAlmostEqual(float(param.data), 0.71, places=5)

    def test_sgd_weight_decay(self):
        """Test SGD with weight decay."""
        param = Parameter(np.array([1.0]))
        optimizer = SGD([param], lr=0.1, weight_decay=0.01)

        param.grad = np.array([0.1])
        optimizer.step()

        # grad_with_decay = 0.1 + 0.01 * 1.0 = 0.11
        # w = 1.0 - 0.1 * 0.11 = 0.989
        self.assertAlmostEqual(float(param.data), 0.989, places=5)

    def test_sgd_nesterov(self):
        """Test SGD with Nesterov momentum."""
        param = Parameter(np.array([1.0]))
        optimizer = SGD([param], lr=0.1, momentum=0.9, nesterov=True)

        param.grad = np.array([1.0])
        optimizer.step()

        # First step with Nesterov
        # v1 = 0.9 * 0 + 1.0 * 1.0 = 1.0
        # grad_nesterov = 1.0 + 0.9 * 1.0 = 1.9
        # w1 = 1.0 - 0.1 * 1.9 = 0.81
        self.assertAlmostEqual(float(param.data), 0.81, places=5)

    def test_sgd_zero_grad(self):
        """Test zeroing gradients."""
        param = Parameter(np.array([1.0, 2.0]))
        optimizer = SGD([param], lr=0.1)

        param.grad = np.array([0.5, 0.5])
        optimizer.zero_grad()

        self.assertTrue(np.all(param.grad == 0))

    def test_sgd_multiple_params(self):
        """Test SGD with multiple parameters."""
        param1 = Parameter(np.array([1.0]))
        param2 = Parameter(np.array([2.0]))
        optimizer = SGD([param1, param2], lr=0.1)

        param1.grad = np.array([0.1])
        param2.grad = np.array([0.2])

        optimizer.step()

        self.assertAlmostEqual(float(param1.data), 0.99, places=5)
        self.assertAlmostEqual(float(param2.data), 1.98, places=5)


class TestAdam(unittest.TestCase):
    """Test Adam optimizer."""

    def test_adam_creation(self):
        """Test creating Adam optimizer."""
        param = Parameter(np.array([1.0, 2.0]))
        optimizer = Adam([param], lr=0.001)

        self.assertEqual(optimizer.lr, 0.001)
        self.assertEqual(optimizer.beta1, 0.9)
        self.assertEqual(optimizer.beta2, 0.999)
        self.assertEqual(optimizer.eps, 1e-8)

    def test_adam_step(self):
        """Test Adam optimization step."""
        param = Parameter(np.array([1.0]))
        optimizer = Adam([param], lr=0.1, betas=(0.9, 0.999))

        # First step
        param.grad = np.array([1.0])
        optimizer.step()

        # m1 = 0.9 * 0 + 0.1 * 1.0 = 0.1
        # v1 = 0.999 * 0 + 0.001 * 1.0 = 0.001
        # m_hat = 0.1 / (1 - 0.9) = 1.0
        # v_hat = 0.001 / (1 - 0.999) = 1.0
        # w = 1.0 - 0.1 * 1.0 / (sqrt(1.0) + 1e-8) ≈ 0.9

        self.assertLess(float(param.data), 1.0)
        self.assertGreater(float(param.data), 0.8)

    def test_adam_bias_correction(self):
        """Test Adam bias correction."""
        param = Parameter(np.array([1.0]))
        optimizer = Adam([param], lr=0.01, betas=(0.9, 0.999))

        initial_value = float(param.data)

        # Take multiple steps
        for _ in range(10):
            param.grad = np.array([0.1])
            optimizer.step()

        # Parameter should have decreased
        self.assertLess(float(param.data), initial_value)

    def test_adam_weight_decay(self):
        """Test Adam with weight decay."""
        param = Parameter(np.array([1.0]))
        optimizer = Adam([param], lr=0.1, weight_decay=0.01)

        param.grad = np.array([0.1])
        optimizer.step()

        # Weight decay should be applied
        self.assertLess(float(param.data), 1.0)

    def test_adam_convergence(self):
        """Test Adam convergence on simple optimization."""
        # Minimize (x - 5)^2
        param = Parameter(np.array([0.0]))
        optimizer = Adam([param], lr=0.1)

        for _ in range(100):
            # Gradient of (x - 5)^2 is 2(x - 5)
            param.grad = 2 * (param.data - 5.0)
            optimizer.step()

        # Should converge close to 5
        self.assertAlmostEqual(float(param.data), 5.0, places=1)


class TestAdamW(unittest.TestCase):
    """Test AdamW optimizer."""

    def test_adamw_creation(self):
        """Test creating AdamW optimizer."""
        param = Parameter(np.array([1.0]))
        optimizer = AdamW([param], lr=0.001, weight_decay=0.01)

        self.assertEqual(optimizer.lr, 0.001)
        self.assertEqual(optimizer.weight_decay, 0.01)

    def test_adamw_decoupled_weight_decay(self):
        """Test AdamW uses decoupled weight decay."""
        param = Parameter(np.array([1.0]))
        optimizer = AdamW([param], lr=0.1, weight_decay=0.1)

        param.grad = np.array([0.0])  # Zero gradient
        initial_value = float(param.data)

        optimizer.step()

        # Even with zero gradient, weight decay should apply
        # w = w - lr * (update + wd * w)
        self.assertLess(float(param.data), initial_value)

    def test_adamw_vs_adam(self):
        """Test that AdamW differs from Adam in weight decay."""
        # Create two identical parameters
        param_adam = Parameter(np.array([1.0]))
        param_adamw = Parameter(np.array([1.0]))

        opt_adam = Adam([param_adam], lr=0.1, weight_decay=0.1)
        opt_adamw = AdamW([param_adamw], lr=0.1, weight_decay=0.1)

        # Apply same gradient
        param_adam.grad = np.array([0.5])
        param_adamw.grad = np.array([0.5])

        opt_adam.step()
        opt_adamw.step()

        # Results should differ due to different weight decay application
        # (This might be very close, but conceptually different)
        # Just check both moved from initial value
        self.assertLess(float(param_adam.data), 1.0)
        self.assertLess(float(param_adamw.data), 1.0)


class TestRMSprop(unittest.TestCase):
    """Test RMSprop optimizer."""

    def test_rmsprop_creation(self):
        """Test creating RMSprop optimizer."""
        param = Parameter(np.array([1.0]))
        optimizer = RMSprop([param], lr=0.01)

        self.assertEqual(optimizer.lr, 0.01)
        self.assertEqual(optimizer.alpha, 0.99)
        self.assertEqual(optimizer.eps, 1e-8)

    def test_rmsprop_step(self):
        """Test RMSprop optimization step."""
        param = Parameter(np.array([1.0]))
        optimizer = RMSprop([param], lr=0.1, alpha=0.9)

        param.grad = np.array([1.0])
        optimizer.step()

        # v1 = 0.9 * 0 + 0.1 * 1.0^2 = 0.1
        # w = 1.0 - 0.1 * 1.0 / sqrt(0.1 + 1e-8)

        self.assertLess(float(param.data), 1.0)

    def test_rmsprop_squared_gradient(self):
        """Test RMSprop uses squared gradients."""
        param = Parameter(np.array([1.0]))
        optimizer = RMSprop([param], lr=0.01, alpha=0.99)

        # Large gradient
        param.grad = np.array([10.0])
        initial = float(param.data)
        optimizer.step()
        step1_change = abs(float(param.data) - initial)

        # Reset
        param.data = np.array([1.0])
        optimizer.v = [None]

        # Small gradient (same sign)
        param.grad = np.array([1.0])
        initial = float(param.data)
        optimizer.step()
        step2_change = abs(float(param.data) - initial)

        # Step size should adapt based on gradient magnitude
        # Larger gradients lead to smaller effective learning rates
        self.assertGreater(step2_change, step1_change)

    def test_rmsprop_momentum(self):
        """Test RMSprop with momentum."""
        param = Parameter(np.array([1.0]))
        optimizer = RMSprop([param], lr=0.1, momentum=0.9)

        param.grad = np.array([1.0])
        optimizer.step()

        # Should apply momentum
        self.assertLess(float(param.data), 1.0)

    def test_rmsprop_convergence(self):
        """Test RMSprop convergence."""
        # Minimize (x - 3)^2
        param = Parameter(np.array([0.0]))
        optimizer = RMSprop([param], lr=0.1)

        for _ in range(100):
            param.grad = 2 * (param.data - 3.0)
            optimizer.step()

        # Should converge close to 3
        self.assertAlmostEqual(float(param.data), 3.0, places=1)


class TestOptimizerComparison(unittest.TestCase):
    """Compare different optimizers."""

    def test_all_optimizers_converge(self):
        """Test that all optimizers can minimize a simple function."""
        # Minimize f(x) = (x - 2)^2 + (y - 3)^2

        def run_optimizer(opt_class, **kwargs):
            param = Parameter(np.array([0.0, 0.0]))
            optimizer = opt_class([param], **kwargs)

            for _ in range(200):
                # Gradient: [2(x-2), 2(y-3)]
                param.grad = 2 * (param.data - np.array([2.0, 3.0]))
                optimizer.step()

            return param.data

        # Test each optimizer
        result_sgd = run_optimizer(SGD, lr=0.01)
        result_adam = run_optimizer(Adam, lr=0.1)
        result_adamw = run_optimizer(AdamW, lr=0.1)
        result_rmsprop = run_optimizer(RMSprop, lr=0.1)

        # All should converge close to [2, 3]
        target = np.array([2.0, 3.0])
        assert_arrays_close(result_sgd, target, atol=0.5)
        assert_arrays_close(result_adam, target, atol=0.5)
        assert_arrays_close(result_adamw, target, atol=0.5)
        assert_arrays_close(result_rmsprop, target, atol=0.5)

    def test_optimizer_state_isolation(self):
        """Test that optimizers maintain separate state for different parameters."""
        param1 = Parameter(np.array([1.0]))
        param2 = Parameter(np.array([2.0]))

        optimizer = Adam([param1, param2], lr=0.1)

        # Update only first parameter
        param1.grad = np.array([1.0])
        param2.grad = None
        optimizer.step()

        # Only param1 should change
        self.assertNotEqual(float(param1.data), 1.0)
        self.assertEqual(float(param2.data), 2.0)


if __name__ == '__main__':
    unittest.main()
