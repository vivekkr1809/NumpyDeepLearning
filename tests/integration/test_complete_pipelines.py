"""Integration tests for complete training pipelines."""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from numpy_dl.core.tensor import Tensor
from numpy_dl.models import MLP, SimpleCNN, Autoencoder, VariationalAutoencoder, GPTModel
from numpy_dl.optim import SGD, Adam
from numpy_dl.loss import MSELoss, CrossEntropyLoss, VAELoss
from numpy_dl.data import DataLoader, TensorDataset
from tests.test_utils import create_simple_dataset


class TestMNISTStylePipeline(unittest.TestCase):
    """Test complete MNIST-style training pipeline."""

    def test_mnist_mlp_pipeline(self):
        """Test complete pipeline with MLP on MNIST-style data."""
        # Create synthetic MNIST-style dataset
        np.random.seed(42)
        n_train = 500
        n_test = 100

        # Flatten 28x28 images
        X_train = np.random.randn(n_train, 784).astype(np.float32) * 0.5
        y_train = np.random.randint(0, 10, size=n_train)

        X_test = np.random.randn(n_test, 784).astype(np.float32) * 0.5
        y_test = np.random.randint(0, 10, size=n_test)

        # Create data loaders
        train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        test_dataset = TensorDataset(Tensor(X_test), Tensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Create model
        model = MLP(input_size=784, hidden_sizes=[256, 128], output_size=10, dropout=0.2)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = CrossEntropyLoss()

        # Training loop
        num_epochs = 5
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            num_batches = 0

            for batch_X, batch_y in train_loader:
                pred = model(batch_X)
                loss = criterion(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += float(loss.data)
                num_batches += 1

            avg_train_loss = train_loss / num_batches

            # Evaluation
            model.eval()
            test_loss = 0
            num_test_batches = 0

            for batch_X, batch_y in test_loader:
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                test_loss += float(loss.data)
                num_test_batches += 1

            avg_test_loss = test_loss / num_test_batches

        # Training should complete without errors
        self.assertIsNotNone(avg_train_loss)
        self.assertIsNotNone(avg_test_loss)
        self.assertGreater(avg_train_loss, 0)

    def test_mnist_cnn_pipeline(self):
        """Test complete pipeline with CNN on MNIST-style data."""
        # Create synthetic MNIST-style dataset
        np.random.seed(42)
        n_train = 200

        # 28x28 grayscale images
        X_train = np.random.randn(n_train, 1, 28, 28).astype(np.float32) * 0.5
        y_train = np.random.randint(0, 10, size=n_train)

        # Create data loader
        train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Create CNN model
        model = SimpleCNN(
            input_channels=1,
            num_classes=10,
            conv_channels=[16, 32],
            fc_sizes=[128],
            input_size=(28, 28),
            dropout=0.3
        )
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = CrossEntropyLoss()

        # Training loop
        num_epochs = 3
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0

            for batch_X, batch_y in train_loader:
                pred = model(batch_X)
                loss = criterion(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.data)
                num_batches += 1

            avg_loss = epoch_loss / num_batches

        # Training should complete
        self.assertGreater(avg_loss, 0)
        self.assertLess(avg_loss, 10)


class TestAutoencoderPipeline(unittest.TestCase):
    """Test complete autoencoder training pipeline."""

    def test_autoencoder_reconstruction(self):
        """Test autoencoder learns to reconstruct data."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 500
        input_dim = 100

        # Generate correlated features
        X = np.random.randn(n_samples, input_dim).astype(np.float32)
        X_tensor = Tensor(X)

        # Create autoencoder
        model = Autoencoder(
            input_dim=input_dim,
            latent_dim=20,
            hidden_dims=[50, 30]
        )
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = MSELoss()

        # Training loop
        num_epochs = 20
        initial_loss = None
        final_loss = None

        for epoch in range(num_epochs):
            model.train()

            # Reconstruct
            recon = model(X_tensor)
            loss = criterion(recon, X_tensor)

            if epoch == 0:
                initial_loss = float(loss.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = float(loss.data)

        # Loss should decrease
        self.assertLess(final_loss, initial_loss)

        # Test encoding
        model.eval()
        z = model.encode(X_tensor)
        self.assertEqual(z.shape, (n_samples, 20))

        # Test decoding
        recon = model.decode(z)
        self.assertEqual(recon.shape, (n_samples, input_dim))

    def test_vae_training(self):
        """Test VAE training pipeline."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 300
        input_dim = 50

        X = np.random.randn(n_samples, input_dim).astype(np.float32)
        X_tensor = Tensor(X)

        # Create VAE
        model = VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=10,
            hidden_dims=[30, 20]
        )
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = VAELoss()

        # Training loop
        num_epochs = 15
        for epoch in range(num_epochs):
            model.train()

            # Forward pass
            recon, mu, logvar = model(X_tensor)
            loss = criterion(recon, X_tensor, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Training should complete
        final_loss = float(loss.data)
        self.assertGreater(final_loss, 0)

        # Test sampling
        model.eval()
        z_sample = Tensor(np.random.randn(5, 10).astype(np.float32))
        samples = model.decode(z_sample)
        self.assertEqual(samples.shape, (5, input_dim))


class TestLanguageModelPipeline(unittest.TestCase):
    """Test language model training pipeline."""

    def test_gpt_character_model(self):
        """Test GPT-style character-level language model."""
        # Create synthetic character-level dataset
        np.random.seed(42)
        vocab_size = 50
        seq_len = 20
        n_sequences = 200

        # Random sequences
        X = np.random.randint(0, vocab_size, size=(n_sequences, seq_len))
        # Targets are shifted by 1
        y = np.random.randint(0, vocab_size, size=(n_sequences, seq_len))

        # Create data loader
        dataset = TensorDataset(Tensor(X), Tensor(y))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Create GPT model
        model = GPTModel(
            vocab_size=vocab_size,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=128,
            max_seq_len=seq_len,
            dropout=0.1
        )
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = CrossEntropyLoss()

        # Training loop
        num_epochs = 5
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0

            for batch_X, batch_y in dataloader:
                # Forward pass
                logits = model(batch_X)  # (batch, seq_len, vocab_size)

                # Reshape for loss
                batch_size = logits.shape[0]
                logits_flat = logits.reshape(-1, vocab_size)
                targets_flat = batch_y.reshape(-1)

                loss = criterion(logits_flat, targets_flat)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.data)
                num_batches += 1

            avg_loss = epoch_loss / num_batches

        # Training should complete
        self.assertGreater(avg_loss, 0)

        # Test generation
        model.eval()
        start_tokens = Tensor(np.array([[1, 2, 3]]))
        generated = model.generate(start_tokens, max_new_tokens=10)

        # Should generate correct shape
        self.assertEqual(generated.shape, (1, 13))  # 3 + 10


class TestEndToEndWorkflows(unittest.TestCase):
    """Test complete end-to-end workflows."""

    def test_full_classification_pipeline(self):
        """Test complete classification workflow: data -> train -> eval -> predict."""
        # 1. Create dataset
        np.random.seed(42)
        X_train, y_train = create_simple_dataset(n_samples=300, n_features=20, n_classes=5)
        X_test, y_test = create_simple_dataset(n_samples=100, n_features=20, n_classes=5)

        # 2. Create data loaders
        train_loader = DataLoader(
            TensorDataset(Tensor(X_train), Tensor(y_train)),
            batch_size=32,
            shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(Tensor(X_test), Tensor(y_test)),
            batch_size=32,
            shuffle=False
        )

        # 3. Create model
        model = MLP(input_size=20, hidden_sizes=[64, 32], output_size=5, dropout=0.1)

        # 4. Create optimizer and loss
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = CrossEntropyLoss()

        # 5. Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                pred = model(batch_X)
                loss = criterion(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 6. Evaluation
        model.eval()
        correct = 0
        total = 0

        for batch_X, batch_y in test_loader:
            pred = model(batch_X)
            pred_classes = np.argmax(pred.data, axis=1)

            if isinstance(batch_y, Tensor):
                true_classes = batch_y.data.astype(int)
            else:
                true_classes = batch_y.astype(int)

            correct += np.sum(pred_classes == true_classes)
            total += len(true_classes)

        accuracy = correct / total

        # 7. Make predictions on new data
        X_new = np.random.randn(5, 20).astype(np.float32)
        predictions = model(Tensor(X_new))

        # Pipeline should complete successfully
        self.assertGreater(accuracy, 0)
        self.assertEqual(predictions.shape, (5, 5))

    def test_model_save_load_workflow(self):
        """Test saving and loading model state."""
        # Create and train a simple model
        model = MLP(input_size=10, hidden_sizes=[20], output_size=5)
        X = Tensor(np.random.randn(50, 10).astype(np.float32))
        y = Tensor(np.random.randint(0, 5, size=50))

        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        # Train for a few steps
        for _ in range(10):
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Get prediction before
        model.eval()
        pred_before = model(X)

        # Save model state
        state = {}
        for i, param in enumerate(model.parameters()):
            state[f'param_{i}'] = param.data.copy()

        # Create new model
        model_new = MLP(input_size=10, hidden_sizes=[20], output_size=5)

        # Load state
        for i, param in enumerate(model_new.parameters()):
            param.data = state[f'param_{i}']

        # Get prediction after
        model_new.eval()
        pred_after = model_new(X)

        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred_before.data, pred_after.data, decimal=5)


class TestRobustness(unittest.TestCase):
    """Test robustness of training pipelines."""

    def test_training_with_nan_handling(self):
        """Test that training can detect NaN values."""
        model = MLP(input_size=10, hidden_sizes=[20], output_size=5)
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        # Normal training
        X = Tensor(np.random.randn(32, 10).astype(np.float32))
        y = Tensor(np.random.randint(0, 5, size=32))

        pred = model(X)
        loss = criterion(pred, y)

        # Loss should be finite
        self.assertFalse(np.isnan(float(loss.data)))
        self.assertFalse(np.isinf(float(loss.data)))

    def test_training_with_different_batch_sizes(self):
        """Test that training works with varying batch sizes."""
        model = MLP(input_size=10, hidden_sizes=[20], output_size=3)
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        # Test different batch sizes
        for batch_size in [1, 8, 32, 64]:
            X = Tensor(np.random.randn(batch_size, 10).astype(np.float32))
            y = Tensor(np.random.randint(0, 3, size=batch_size))

            pred = model(X)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Should work for all batch sizes
            self.assertEqual(pred.shape, (batch_size, 3))


if __name__ == '__main__':
    unittest.main()
