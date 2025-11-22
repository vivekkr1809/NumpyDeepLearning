"""Unit tests for model architectures."""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from numpy_dl.core.tensor import Tensor
from numpy_dl.models import (
    MLP, SimpleCNN, VGG,
    ResNet, resnet18,
    UNet,
    SimpleRNN, Seq2Seq,
    Autoencoder, ConvAutoencoder, VariationalAutoencoder,
    TransformerEncoderLayer, TransformerEncoder, GPTModel
)
from tests.test_utils import count_parameters


class TestMLP(unittest.TestCase):
    """Test Multi-Layer Perceptron."""

    def test_mlp_creation(self):
        """Test creating MLP."""
        model = MLP(input_size=10, hidden_sizes=[20, 30], output_size=5)
        self.assertEqual(model.input_size, 10)
        self.assertEqual(model.hidden_sizes, [20, 30])
        self.assertEqual(model.output_size, 5)

    def test_mlp_forward_shape(self):
        """Test MLP forward pass output shape."""
        model = MLP(input_size=10, hidden_sizes=[20], output_size=5)
        x = Tensor(np.random.randn(2, 10))
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

    def test_mlp_no_hidden_layers(self):
        """Test MLP with no hidden layers."""
        model = MLP(input_size=10, hidden_sizes=[], output_size=5)
        x = Tensor(np.random.randn(2, 10))
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

    def test_mlp_multiple_hidden_layers(self):
        """Test MLP with multiple hidden layers."""
        model = MLP(input_size=10, hidden_sizes=[20, 30, 40], output_size=5)
        x = Tensor(np.random.randn(2, 10))
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

    def test_mlp_backward(self):
        """Test MLP backward pass."""
        model = MLP(input_size=10, hidden_sizes=[20], output_size=5)
        x = Tensor(np.random.randn(2, 10), requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Input gradient should exist
        self.assertIsNotNone(x.grad)

    def test_mlp_parameters(self):
        """Test MLP parameter count."""
        model = MLP(input_size=10, hidden_sizes=[20], output_size=5, dropout=0.0)
        params = count_parameters(model)
        # 10*20 + 20 (first layer) + 20*5 + 5 (second layer) = 325
        expected = 10 * 20 + 20 + 20 * 5 + 5
        self.assertEqual(params, expected)

    def test_mlp_with_dropout(self):
        """Test MLP with dropout."""
        model = MLP(input_size=10, hidden_sizes=[20], output_size=5, dropout=0.5)
        x = Tensor(np.random.randn(2, 10))

        # Train mode
        model.train()
        y_train = model(x)

        # Eval mode
        model.eval()
        y_eval = model(x)

        # Shapes should match
        self.assertEqual(y_train.shape, y_eval.shape)


class TestSimpleCNN(unittest.TestCase):
    """Test Simple CNN."""

    def test_cnn_creation(self):
        """Test creating SimpleCNN."""
        model = SimpleCNN(input_channels=1, num_classes=10)
        self.assertEqual(model.input_channels, 1)
        self.assertEqual(model.num_classes, 10)

    def test_cnn_forward_shape(self):
        """Test CNN forward pass output shape."""
        model = SimpleCNN(
            input_channels=1,
            num_classes=10,
            conv_channels=[16, 32],
            fc_sizes=[128],
            input_size=(28, 28)
        )
        x = Tensor(np.random.randn(2, 1, 28, 28))
        y = model(x)
        self.assertEqual(y.shape, (2, 10))

    def test_cnn_rgb_input(self):
        """Test CNN with RGB input."""
        model = SimpleCNN(
            input_channels=3,
            num_classes=10,
            conv_channels=[32, 64],
            input_size=(32, 32)
        )
        x = Tensor(np.random.randn(2, 3, 32, 32))
        y = model(x)
        self.assertEqual(y.shape, (2, 10))

    def test_cnn_backward(self):
        """Test CNN backward pass."""
        model = SimpleCNN(input_channels=1, num_classes=10, input_size=(28, 28))
        x = Tensor(np.random.randn(2, 1, 28, 28), requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Input gradient should exist
        self.assertIsNotNone(x.grad)


class TestVGG(unittest.TestCase):
    """Test VGG network."""

    def test_vgg_creation(self):
        """Test creating VGG."""
        model = VGG(
            input_channels=3,
            num_classes=1000,
            architecture=[(64, 2), (128, 2)],
            use_batch_norm=False
        )
        self.assertEqual(model.input_channels, 3)
        self.assertEqual(model.num_classes, 1000)

    def test_vgg_forward_shape(self):
        """Test VGG forward pass output shape."""
        model = VGG(
            input_channels=3,
            num_classes=10,
            architecture=[(64, 1), (128, 1)],
        )
        # Note: VGG expects specific input size for classifier
        x = Tensor(np.random.randn(2, 3, 224, 224))
        y = model(x)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[1], 10)


class TestResNet(unittest.TestCase):
    """Test ResNet."""

    def test_resnet_creation(self):
        """Test creating ResNet."""
        model = resnet18(num_classes=10)
        self.assertIsNotNone(model)

    def test_resnet18_forward_shape(self):
        """Test ResNet18 forward pass output shape."""
        model = resnet18(num_classes=10)
        x = Tensor(np.random.randn(2, 3, 224, 224))
        y = model(x)
        self.assertEqual(y.shape, (2, 10))

    def test_resnet_backward(self):
        """Test ResNet backward pass."""
        model = resnet18(num_classes=10)
        x = Tensor(np.random.randn(1, 3, 224, 224), requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Input gradient should exist
        self.assertIsNotNone(x.grad)


class TestUNet(unittest.TestCase):
    """Test UNet."""

    def test_unet_creation(self):
        """Test creating UNet."""
        model = UNet(in_channels=1, out_channels=1)
        self.assertIsNotNone(model)

    def test_unet_forward_shape(self):
        """Test UNet forward pass output shape."""
        model = UNet(in_channels=1, out_channels=1, features=[16, 32])
        x = Tensor(np.random.randn(1, 1, 64, 64))
        y = model(x)
        # UNet preserves spatial dimensions
        self.assertEqual(y.shape[0], 1)  # batch
        self.assertEqual(y.shape[1], 1)  # out_channels
        # Spatial dimensions might differ slightly due to architecture


class TestRNN(unittest.TestCase):
    """Test RNN models."""

    def test_rnn_creation(self):
        """Test creating SimpleRNN."""
        model = SimpleRNN(input_size=10, hidden_size=20, num_layers=1)
        self.assertIsNotNone(model)

    def test_rnn_forward_shape(self):
        """Test RNN forward pass output shape."""
        model = SimpleRNN(input_size=10, hidden_size=20, num_layers=1)
        x = Tensor(np.random.randn(2, 5, 10))  # (batch, seq_len, input_size)
        output, hidden = model(x)

        self.assertEqual(output.shape, (2, 5, 20))  # (batch, seq_len, hidden_size)

    def test_seq2seq_creation(self):
        """Test creating Seq2Seq."""
        model = Seq2Seq(
            input_vocab_size=100,
            output_vocab_size=100,
            embedding_dim=32,
            hidden_size=64
        )
        self.assertIsNotNone(model)


class TestAutoencoders(unittest.TestCase):
    """Test Autoencoder models."""

    def test_autoencoder_creation(self):
        """Test creating Autoencoder."""
        model = Autoencoder(input_dim=784, latent_dim=32)
        self.assertIsNotNone(model)

    def test_autoencoder_forward_shape(self):
        """Test Autoencoder forward pass output shape."""
        model = Autoencoder(input_dim=784, latent_dim=32, hidden_dims=[256, 128])
        x = Tensor(np.random.randn(2, 784))
        recon = model(x)
        self.assertEqual(recon.shape, (2, 784))

    def test_autoencoder_encoding(self):
        """Test Autoencoder encoding."""
        model = Autoencoder(input_dim=784, latent_dim=32)
        x = Tensor(np.random.randn(2, 784))
        z = model.encode(x)
        self.assertEqual(z.shape, (2, 32))

    def test_conv_autoencoder_creation(self):
        """Test creating ConvAutoencoder."""
        model = ConvAutoencoder(in_channels=1, latent_dim=64)
        self.assertIsNotNone(model)

    def test_conv_autoencoder_forward_shape(self):
        """Test ConvAutoencoder forward pass output shape."""
        model = ConvAutoencoder(in_channels=1, latent_dim=64, channels=[16, 32])
        x = Tensor(np.random.randn(2, 1, 28, 28))
        recon = model(x)
        # Output should match input shape
        self.assertEqual(recon.shape[0], 2)
        self.assertEqual(recon.shape[1], 1)

    def test_vae_creation(self):
        """Test creating VariationalAutoencoder."""
        model = VariationalAutoencoder(input_dim=784, latent_dim=32)
        self.assertIsNotNone(model)

    def test_vae_forward_shape(self):
        """Test VAE forward pass output shape."""
        model = VariationalAutoencoder(input_dim=784, latent_dim=32, hidden_dims=[256])
        x = Tensor(np.random.randn(2, 784))
        recon, mu, logvar = model(x)

        self.assertEqual(recon.shape, (2, 784))
        self.assertEqual(mu.shape, (2, 32))
        self.assertEqual(logvar.shape, (2, 32))

    def test_vae_sampling(self):
        """Test VAE reparameterization trick."""
        model = VariationalAutoencoder(input_dim=784, latent_dim=32)
        x = Tensor(np.random.randn(2, 784))
        recon, mu, logvar = model(x)

        # mu and logvar should be different
        self.assertFalse(np.allclose(mu.data, logvar.data))


class TestTransformer(unittest.TestCase):
    """Test Transformer models."""

    def test_transformer_encoder_layer_creation(self):
        """Test creating TransformerEncoderLayer."""
        layer = TransformerEncoderLayer(d_model=64, num_heads=4)
        self.assertIsNotNone(layer)

    def test_transformer_encoder_layer_forward(self):
        """Test TransformerEncoderLayer forward pass."""
        layer = TransformerEncoderLayer(d_model=64, num_heads=4)
        x = Tensor(np.random.randn(2, 10, 64))  # (batch, seq_len, d_model)
        output = layer(x)
        self.assertEqual(output.shape, (2, 10, 64))

    def test_transformer_encoder_creation(self):
        """Test creating TransformerEncoder."""
        encoder = TransformerEncoder(
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256
        )
        self.assertIsNotNone(encoder)

    def test_transformer_encoder_forward(self):
        """Test TransformerEncoder forward pass."""
        encoder = TransformerEncoder(
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256
        )
        x = Tensor(np.random.randn(2, 10, 64))
        output = encoder(x)
        self.assertEqual(output.shape, (2, 10, 64))

    def test_gpt_model_creation(self):
        """Test creating GPT model."""
        model = GPTModel(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=128
        )
        self.assertIsNotNone(model)

    def test_gpt_forward_shape(self):
        """Test GPT forward pass output shape."""
        model = GPTModel(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=128
        )
        x = Tensor(np.random.randint(0, 100, size=(2, 10)))  # (batch, seq_len)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 10, 100))  # (batch, seq_len, vocab_size)

    def test_gpt_generation(self):
        """Test GPT text generation."""
        model = GPTModel(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=128
        )
        start_tokens = Tensor(np.array([[1, 2, 3]]))  # (1, 3)
        generated = model.generate(start_tokens, max_new_tokens=5)

        # Should generate 5 more tokens
        self.assertEqual(generated.shape, (1, 8))  # 3 + 5

    def test_gpt_generation_temperature(self):
        """Test GPT generation with different temperatures."""
        model = GPTModel(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=128
        )
        start_tokens = Tensor(np.array([[1, 2]]))

        # Low temperature (more deterministic)
        gen_low = model.generate(start_tokens, max_new_tokens=10, temperature=0.1)

        # High temperature (more random)
        gen_high = model.generate(start_tokens, max_new_tokens=10, temperature=2.0)

        # Both should generate correct length
        self.assertEqual(gen_low.shape, (1, 12))
        self.assertEqual(gen_high.shape, (1, 12))


class TestModelComparison(unittest.TestCase):
    """Compare different models."""

    def test_all_models_forward_pass(self):
        """Test that all models can perform forward pass."""
        # MLP
        mlp = MLP(input_size=10, hidden_sizes=[20], output_size=5)
        x_mlp = Tensor(np.random.randn(2, 10))
        y_mlp = mlp(x_mlp)
        self.assertEqual(y_mlp.shape, (2, 5))

        # CNN
        cnn = SimpleCNN(input_channels=1, num_classes=10, input_size=(28, 28))
        x_cnn = Tensor(np.random.randn(2, 1, 28, 28))
        y_cnn = cnn(x_cnn)
        self.assertEqual(y_cnn.shape, (2, 10))

        # Autoencoder
        ae = Autoencoder(input_dim=100, latent_dim=20)
        x_ae = Tensor(np.random.randn(2, 100))
        y_ae = ae(x_ae)
        self.assertEqual(y_ae.shape, (2, 100))

    def test_model_train_eval_mode(self):
        """Test that models can switch between train and eval modes."""
        model = MLP(input_size=10, hidden_sizes=[20], output_size=5, dropout=0.5)

        # Train mode
        model.train()
        self.assertTrue(model.training)

        # Eval mode
        model.eval()
        self.assertFalse(model.training)

    def test_model_parameters(self):
        """Test that all models return parameters."""
        models = [
            MLP(input_size=10, hidden_sizes=[20], output_size=5),
            SimpleCNN(input_channels=1, num_classes=10, input_size=(28, 28)),
            Autoencoder(input_dim=100, latent_dim=20),
        ]

        for model in models:
            params = list(model.parameters())
            self.assertGreater(len(params), 0)


if __name__ == '__main__':
    unittest.main()
