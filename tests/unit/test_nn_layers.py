"""Unit tests for neural network layers."""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from numpy_dl.core.tensor import Tensor
from numpy_dl.nn import (
    Linear, Conv2d, MaxPool2d, AvgPool2d,
    ReLU, Sigmoid, Tanh, Softmax, LogSoftmax,
    Dropout, BatchNorm1d, LayerNorm,
    Embedding, MultiHeadAttention
)
from tests.test_utils import assert_tensors_close, assert_arrays_close, count_parameters


class TestLinear(unittest.TestCase):
    """Test Linear layer."""

    def test_linear_creation(self):
        """Test creating linear layer."""
        layer = Linear(10, 5)
        self.assertEqual(layer.in_features, 10)
        self.assertEqual(layer.out_features, 5)
        self.assertEqual(layer.weight.shape, (5, 10))
        self.assertIsNotNone(layer.bias)

    def test_linear_forward(self):
        """Test linear forward pass."""
        layer = Linear(3, 2, bias=False)
        layer.weight.data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        x = Tensor([[1, 0, 0]], requires_grad=True)
        y = layer(x)

        self.assertEqual(y.shape, (1, 2))
        assert_arrays_close(y.data, np.array([[1, 4]]))

    def test_linear_backward(self):
        """Test linear backward pass."""
        layer = Linear(2, 1, bias=False)
        layer.weight.data = np.array([[1, 2]], dtype=np.float32)

        x = Tensor([[3, 4]], requires_grad=True)
        y = layer(x)
        y.backward()

        # Check input gradient
        assert_arrays_close(x.grad, layer.weight.data)

        # Check weight gradient
        assert_arrays_close(layer.weight.grad, x.data.T)

    def test_linear_no_bias(self):
        """Test linear without bias."""
        layer = Linear(3, 2, bias=False)
        self.assertIsNone(layer.bias)

    def test_linear_parameter_count(self):
        """Test parameter count."""
        layer = Linear(10, 5, bias=True)
        params = count_parameters(layer)
        self.assertEqual(params, 10 * 5 + 5)  # weights + bias


class TestConv2d(unittest.TestCase):
    """Test Conv2d layer."""

    def test_conv2d_creation(self):
        """Test creating conv layer."""
        layer = Conv2d(3, 16, kernel_size=3)
        self.assertEqual(layer.in_channels, 3)
        self.assertEqual(layer.out_channels, 16)
        self.assertEqual(layer.kernel_size, (3, 3))

    def test_conv2d_forward_shape(self):
        """Test conv forward pass output shape."""
        layer = Conv2d(3, 16, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 28, 28))
        y = layer(x)
        self.assertEqual(y.shape, (2, 16, 28, 28))

    def test_conv2d_forward_computation(self):
        """Test conv computation correctness."""
        # Simple 1x1 conv
        layer = Conv2d(1, 1, kernel_size=1, bias=False)
        layer.weight.data = np.array([[[[2.0]]]], dtype=np.float32)

        x = Tensor(np.ones((1, 1, 2, 2), dtype=np.float32))
        y = layer(x)

        assert_arrays_close(y.data, np.ones((1, 1, 2, 2)) * 2)


class TestPooling(unittest.TestCase):
    """Test pooling layers."""

    def test_maxpool2d_shape(self):
        """Test max pooling output shape."""
        layer = MaxPool2d(kernel_size=2)
        x = Tensor(np.random.randn(2, 3, 28, 28))
        y = layer(x)
        self.assertEqual(y.shape, (2, 3, 14, 14))

    def test_maxpool2d_computation(self):
        """Test max pooling computation."""
        layer = MaxPool2d(kernel_size=2)
        x = Tensor(np.array([[[[1, 2], [3, 4]]]], dtype=np.float32))
        y = layer(x)
        self.assertEqual(float(y.data), 4.0)

    def test_avgpool2d_shape(self):
        """Test average pooling output shape."""
        layer = AvgPool2d(kernel_size=2)
        x = Tensor(np.random.randn(2, 3, 28, 28))
        y = layer(x)
        self.assertEqual(y.shape, (2, 3, 14, 14))

    def test_avgpool2d_computation(self):
        """Test average pooling computation."""
        layer = AvgPool2d(kernel_size=2)
        x = Tensor(np.array([[[[1, 2], [3, 4]]]], dtype=np.float32))
        y = layer(x)
        self.assertAlmostEqual(float(y.data), 2.5, places=4)


class TestActivations(unittest.TestCase):
    """Test activation functions."""

    def test_relu(self):
        """Test ReLU activation."""
        relu = ReLU()
        x = Tensor([-1, 0, 1, 2])
        y = relu(x)
        assert_arrays_close(y.data, np.array([0, 0, 1, 2]))

    def test_sigmoid(self):
        """Test Sigmoid activation."""
        sigmoid = Sigmoid()
        x = Tensor([0.0])
        y = sigmoid(x)
        self.assertAlmostEqual(float(y.data), 0.5, places=4)

    def test_tanh(self):
        """Test Tanh activation."""
        tanh = Tanh()
        x = Tensor([0.0])
        y = tanh(x)
        self.assertAlmostEqual(float(y.data), 0.0, places=4)

    def test_softmax(self):
        """Test Softmax activation."""
        softmax = Softmax()
        x = Tensor([[1, 2, 3]])
        y = softmax(x)
        # Check sum to 1
        self.assertAlmostEqual(float(y.data.sum()), 1.0, places=4)
        # Check all positive
        self.assertTrue(np.all(y.data > 0))

    def test_log_softmax(self):
        """Test LogSoftmax activation."""
        log_softmax = LogSoftmax()
        x = Tensor([[1, 2, 3]])
        y = log_softmax(x)
        # Check all negative or zero
        self.assertTrue(np.all(y.data <= 0))


class TestDropout(unittest.TestCase):
    """Test Dropout layer."""

    def test_dropout_training(self):
        """Test dropout in training mode."""
        dropout = Dropout(0.5)
        dropout.train()
        x = Tensor(np.ones((100, 10)))
        y = dropout(x)
        # Some values should be zero
        self.assertTrue(np.any(y.data == 0))
        # Scale should compensate
        self.assertTrue(np.any(y.data > 1))

    def test_dropout_eval(self):
        """Test dropout in eval mode."""
        dropout = Dropout(0.5)
        dropout.eval()
        x = Tensor(np.ones((10, 10)))
        y = dropout(x)
        # No dropout in eval mode
        assert_arrays_close(y.data, x.data)


class TestNormalization(unittest.TestCase):
    """Test normalization layers."""

    def test_batchnorm1d_shape(self):
        """Test BatchNorm1d output shape."""
        bn = BatchNorm1d(10)
        x = Tensor(np.random.randn(32, 10))
        y = bn(x)
        self.assertEqual(y.shape, x.shape)

    def test_batchnorm1d_training(self):
        """Test BatchNorm1d in training mode."""
        bn = BatchNorm1d(3)
        bn.train()
        x = Tensor(np.random.randn(32, 3))
        y = bn(x)

        # Check approximately normalized
        mean = y.data.mean(axis=0)
        std = y.data.std(axis=0)
        assert_arrays_close(mean, np.zeros(3), atol=0.1)
        assert_arrays_close(std, np.ones(3), atol=0.2)

    def test_layernorm_shape(self):
        """Test LayerNorm output shape."""
        ln = LayerNorm(10)
        x = Tensor(np.random.randn(32, 10))
        y = ln(x)
        self.assertEqual(y.shape, x.shape)

    def test_layernorm_computation(self):
        """Test LayerNorm normalization."""
        ln = LayerNorm(4)
        x = Tensor([[1, 2, 3, 4]], dtype=np.float32)
        y = ln(x)

        # Check normalized per sample
        mean = y.data.mean(axis=-1)
        std = y.data.std(axis=-1)
        assert_arrays_close(mean, np.zeros(1), atol=1e-5)


class TestEmbedding(unittest.TestCase):
    """Test Embedding layer."""

    def test_embedding_creation(self):
        """Test creating embedding layer."""
        emb = Embedding(100, 16)
        self.assertEqual(emb.num_embeddings, 100)
        self.assertEqual(emb.embedding_dim, 16)
        self.assertEqual(emb.weight.shape, (100, 16))

    def test_embedding_forward(self):
        """Test embedding forward pass."""
        emb = Embedding(10, 4)
        emb.weight.data = np.arange(40, dtype=np.float32).reshape(10, 4)

        indices = Tensor(np.array([[0, 1, 2]]))
        output = emb(indices)

        self.assertEqual(output.shape, (1, 3, 4))
        assert_arrays_close(output.data[0, 0], np.array([0, 1, 2, 3]))
        assert_arrays_close(output.data[0, 1], np.array([4, 5, 6, 7]))

    def test_embedding_padding(self):
        """Test embedding with padding index."""
        emb = Embedding(10, 4, padding_idx=0)
        # Padding embedding should be zero
        assert_arrays_close(emb.weight.data[0], np.zeros(4))


class TestAttention(unittest.TestCase):
    """Test attention mechanisms."""

    def test_multihead_attention_shape(self):
        """Test multi-head attention output shape."""
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        q = k = v = Tensor(np.random.randn(2, 10, 64))  # (batch, seq, d_model)
        output, attn_weights = mha(q, k, v)

        self.assertEqual(output.shape, (2, 10, 64))

    def test_multihead_attention_heads_divisible(self):
        """Test that d_model must be divisible by num_heads."""
        with self.assertRaises(AssertionError):
            MultiHeadAttention(d_model=65, num_heads=4)

    def test_multihead_attention_parameters(self):
        """Test multi-head attention has correct parameters."""
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        params = count_parameters(mha)
        # Q, K, V, O projections: 4 * (64 * 64) = 16384
        self.assertEqual(params, 64 * 64 * 4)


if __name__ == '__main__':
    unittest.main()
