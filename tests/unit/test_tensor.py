"""Unit tests for Tensor class."""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from numpy_dl.core.tensor import Tensor, zeros, ones, randn, rand
from tests.test_utils import assert_tensors_close, assert_arrays_close, numerical_gradient


class TestTensorCreation(unittest.TestCase):
    """Test tensor creation methods."""

    def test_tensor_from_array(self):
        """Test creating tensor from numpy array."""
        data = np.array([1, 2, 3])
        t = Tensor(data)
        assert_arrays_close(t.data, data)
        self.assertFalse(t.requires_grad)

    def test_tensor_with_grad(self):
        """Test creating tensor with grad tracking."""
        t = Tensor([1, 2, 3], requires_grad=True)
        self.assertTrue(t.requires_grad)
        self.assertIsNone(t.grad)

    def test_zeros(self):
        """Test zeros creation."""
        t = zeros(3, 4)
        self.assertEqual(t.shape, (3, 4))
        assert_arrays_close(t.data, np.zeros((3, 4)))

    def test_ones(self):
        """Test ones creation."""
        t = ones(2, 3)
        self.assertEqual(t.shape, (2, 3))
        assert_arrays_close(t.data, np.ones((2, 3)))

    def test_randn(self):
        """Test random normal creation."""
        np.random.seed(42)
        t = randn(3, 4)
        self.assertEqual(t.shape, (3, 4))
        self.assertTrue(np.abs(t.data.mean()) < 1)  # Roughly centered

    def test_rand(self):
        """Test random uniform creation."""
        t = rand(3, 4)
        self.assertEqual(t.shape, (3, 4))
        self.assertTrue(np.all(t.data >= 0) and np.all(t.data <= 1))


class TestTensorOperations(unittest.TestCase):
    """Test tensor arithmetic operations."""

    def test_addition(self):
        """Test tensor addition."""
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b
        assert_arrays_close(c.data, np.array([5, 7, 9]))

    def test_addition_backward(self):
        """Test addition backward pass."""
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b
        c.backward()
        assert_arrays_close(a.grad, np.ones(3))
        assert_arrays_close(b.grad, np.ones(3))

    def test_multiplication(self):
        """Test tensor multiplication."""
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([2, 3, 4], requires_grad=True)
        c = a * b
        assert_arrays_close(c.data, np.array([2, 6, 12]))

    def test_multiplication_backward(self):
        """Test multiplication backward pass."""
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        c = a * b
        c.backward()
        assert_arrays_close(a.grad, b.data)
        assert_arrays_close(b.grad, a.data)

    def test_power(self):
        """Test power operation."""
        a = Tensor([2, 3, 4], requires_grad=True)
        c = a ** 2
        assert_arrays_close(c.data, np.array([4, 9, 16]))

    def test_power_backward(self):
        """Test power backward pass."""
        a = Tensor([2.0, 3.0], requires_grad=True)
        c = a ** 2
        c.backward()
        assert_arrays_close(a.grad, 2 * a.data)

    def test_matmul(self):
        """Test matrix multiplication."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)
        c = a @ b
        expected = np.array([[19, 22], [43, 50]])
        assert_arrays_close(c.data, expected)

    def test_subtraction(self):
        """Test subtraction."""
        a = Tensor([5, 6, 7])
        b = Tensor([1, 2, 3])
        c = a - b
        assert_arrays_close(c.data, np.array([4, 4, 4]))

    def test_division(self):
        """Test division."""
        a = Tensor([6, 8, 10])
        b = Tensor([2, 4, 5])
        c = a / b
        assert_arrays_close(c.data, np.array([3, 2, 2]))

    def test_negation(self):
        """Test negation."""
        a = Tensor([1, -2, 3])
        b = -a
        assert_arrays_close(b.data, np.array([-1, 2, -3]))


class TestTensorReductions(unittest.TestCase):
    """Test tensor reduction operations."""

    def test_sum(self):
        """Test sum operation."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        s = a.sum()
        self.assertEqual(float(s.data), 10.0)

    def test_sum_backward(self):
        """Test sum backward pass."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        s = a.sum()
        s.backward()
        assert_arrays_close(a.grad, np.ones((2, 2)))

    def test_sum_axis(self):
        """Test sum along axis."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        s = a.sum(axis=0)
        assert_arrays_close(s.data, np.array([4, 6]))

    def test_mean(self):
        """Test mean operation."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        m = a.mean()
        self.assertEqual(float(m.data), 2.5)

    def test_mean_backward(self):
        """Test mean backward pass."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        m = a.mean()
        m.backward()
        assert_arrays_close(a.grad, np.ones((2, 2)) / 4)


class TestTensorReshape(unittest.TestCase):
    """Test tensor reshaping operations."""

    def test_reshape(self):
        """Test reshape operation."""
        a = Tensor([[1, 2], [3, 4]])
        b = a.reshape(4)
        assert_arrays_close(b.data, np.array([1, 2, 3, 4]))

    def test_reshape_backward(self):
        """Test reshape backward pass."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = a.reshape(4)
        b.backward()
        assert_arrays_close(a.grad, np.ones((2, 2)))

    def test_transpose(self):
        """Test transpose operation."""
        a = Tensor([[1, 2], [3, 4]])
        b = a.transpose()
        assert_arrays_close(b.data, np.array([[1, 3], [2, 4]]))

    def test_transpose_property(self):
        """Test transpose property."""
        a = Tensor([[1, 2], [3, 4]])
        b = a.T
        assert_arrays_close(b.data, np.array([[1, 3], [2, 4]]))


class TestTensorElementwise(unittest.TestCase):
    """Test elementwise operations."""

    def test_exp(self):
        """Test exp operation."""
        a = Tensor([0, 1, 2], requires_grad=True)
        b = a.exp()
        assert_arrays_close(b.data, np.exp([0, 1, 2]), rtol=1e-4)

    def test_exp_backward(self):
        """Test exp backward pass."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = a.exp()
        b.backward()
        assert_arrays_close(a.grad, b.data, rtol=1e-4)

    def test_log(self):
        """Test log operation."""
        a = Tensor([1, 2, 3], requires_grad=True)
        b = a.log()
        assert_arrays_close(b.data, np.log([1, 2, 3]), rtol=1e-4)

    def test_sqrt(self):
        """Test sqrt operation."""
        a = Tensor([1, 4, 9])
        b = a.sqrt()
        assert_arrays_close(b.data, np.array([1, 2, 3]))

    def test_clip(self):
        """Test clip operation."""
        a = Tensor([-1, 0, 5, 10])
        b = a.clip(0, 5)
        assert_arrays_close(b.data, np.array([0, 0, 5, 5]))


class TestTensorIndexing(unittest.TestCase):
    """Test tensor indexing."""

    def test_getitem(self):
        """Test indexing."""
        a = Tensor([[1, 2], [3, 4]])
        b = a[0]
        assert_arrays_close(b.data, np.array([1, 2]))

    def test_getitem_backward(self):
        """Test indexing backward pass."""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = a[0]
        b.backward()
        expected_grad = np.array([[1, 1], [0, 0]])
        assert_arrays_close(a.grad, expected_grad)


class TestTensorUtilities(unittest.TestCase):
    """Test tensor utility methods."""

    def test_to_numpy(self):
        """Test conversion to numpy."""
        a = Tensor([1, 2, 3])
        np_array = a.numpy()
        self.assertIsInstance(np_array, np.ndarray)
        assert_arrays_close(np_array, np.array([1, 2, 3]))

    def test_item(self):
        """Test getting scalar value."""
        a = Tensor([5.0])
        self.assertEqual(a.item(), 5.0)

    def test_detach(self):
        """Test detaching from computation graph."""
        a = Tensor([1, 2, 3], requires_grad=True)
        b = a.detach()
        self.assertFalse(b.requires_grad)
        assert_arrays_close(b.data, a.data)

    def test_zero_grad(self):
        """Test zeroing gradient."""
        a = Tensor([1, 2, 3], requires_grad=True)
        a.grad = np.ones(3)
        a.zero_grad()
        self.assertIsNone(a.grad)


class TestTensorBackpropagation(unittest.TestCase):
    """Test backpropagation through complex graphs."""

    def test_simple_chain(self):
        """Test backprop through simple chain."""
        a = Tensor([2.0], requires_grad=True)
        b = a * 3
        c = b + 5
        d = c ** 2
        d.backward()
        # d = (3a + 5)^2
        # dd/da = 2(3a + 5) * 3 = 6(3*2 + 5) = 66
        self.assertAlmostEqual(float(a.grad), 66.0, places=4)

    def test_branching_graph(self):
        """Test backprop through branching graph."""
        a = Tensor([2.0], requires_grad=True)
        b = a * 2
        c = a + 3
        d = b + c
        d.backward()
        # d = 2a + (a + 3) = 3a + 3
        # dd/da = 3
        self.assertAlmostEqual(float(a.grad), 3.0, places=4)

    def test_no_grad_tensor(self):
        """Test that tensors without grad don't accumulate."""
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=False)
        c = a * b
        c.backward()
        assert_arrays_close(a.grad, np.array([3.0]))
        self.assertIsNone(b.grad)


if __name__ == '__main__':
    unittest.main()
