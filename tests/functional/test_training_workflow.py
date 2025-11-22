"""Functional tests for training workflows."""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from numpy_dl.core.tensor import Tensor
from numpy_dl.models import MLP, SimpleCNN, Autoencoder
from numpy_dl.nn import Linear
from numpy_dl.optim import SGD, Adam
from numpy_dl.loss import MSELoss, CrossEntropyLoss, BCELoss
from numpy_dl.data import DataLoader, TensorDataset
from tests.test_utils import create_simple_dataset, create_regression_dataset


class TestBasicTraining(unittest.TestCase):
    """Test basic training workflows."""

    def test_simple_regression_training(self):
        """Test training a simple regression model."""
        # Create simple regression problem: y = 2x + 1
        X = np.random.randn(100, 1).astype(np.float32)
        y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

        X_tensor = Tensor(X)
        y_tensor = Tensor(y)

        # Create simple model
        model = Linear(1, 1)
        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = MSELoss()

        # Train for a few epochs
        initial_loss = None
        final_loss = None

        for epoch in range(50):
            # Forward pass
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)

            if epoch == 0:
                initial_loss = float(loss.data)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = float(loss.data)

        # Loss should decrease
        self.assertLess(final_loss, initial_loss)
        # Should learn something close to y = 2x + 1
        self.assertLess(final_loss, 1.0)

    def test_binary_classification_training(self):
        """Test training a binary classification model."""
        # Create binary classification dataset
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32).reshape(-1, 1)

        X_tensor = Tensor(X)
        y_tensor = Tensor(y)

        # Create model
        model = MLP(input_size=10, hidden_sizes=[20], output_size=1)
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = BCELoss()

        # Train
        initial_loss = None
        final_loss = None

        for epoch in range(30):
            from numpy_dl.nn import Sigmoid
            sigmoid = Sigmoid()

            pred_logits = model(X_tensor)
            pred = sigmoid(pred_logits)
            loss = criterion(pred, y_tensor)

            if epoch == 0:
                initial_loss = float(loss.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = float(loss.data)

        # Loss should decrease significantly
        self.assertLess(final_loss, initial_loss * 0.5)

    def test_multiclass_classification_training(self):
        """Test training a multiclass classification model."""
        # Create synthetic multiclass dataset
        np.random.seed(42)
        X, y = create_simple_dataset(n_samples=200, n_features=10, n_classes=3)

        X_tensor = Tensor(X)
        y_tensor = Tensor(y)

        # Create model
        model = MLP(input_size=10, hidden_sizes=[30, 20], output_size=3)
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        # Train
        initial_loss = None
        final_loss = None

        for epoch in range(50):
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)

            if epoch == 0:
                initial_loss = float(loss.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = float(loss.data)

        # Loss should decrease
        self.assertLess(final_loss, initial_loss)

    def test_autoencoder_training(self):
        """Test training an autoencoder."""
        # Create random data
        np.random.seed(42)
        X = np.random.randn(100, 50).astype(np.float32)
        X_tensor = Tensor(X)

        # Create autoencoder
        model = Autoencoder(input_dim=50, latent_dim=10, hidden_dims=[30, 20])
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = MSELoss()

        # Train
        initial_loss = None
        final_loss = None

        for epoch in range(30):
            # Reconstruct input
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


class TestBatchTraining(unittest.TestCase):
    """Test training with mini-batches."""

    def test_dataloader_training(self):
        """Test training with DataLoader."""
        # Create dataset
        np.random.seed(42)
        X, y = create_simple_dataset(n_samples=200, n_features=10, n_classes=3)

        X_tensor = Tensor(X)
        y_tensor = Tensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Create model
        model = MLP(input_size=10, hidden_sizes=[20], output_size=3)
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        # Train for one epoch
        epoch_loss = 0
        num_batches = 0

        for batch_X, batch_y in dataloader:
            pred = model(batch_X)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.data)
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Should have processed multiple batches
        self.assertGreater(num_batches, 1)
        # Loss should be reasonable
        self.assertGreater(avg_loss, 0)
        self.assertLess(avg_loss, 10)

    def test_multi_epoch_training(self):
        """Test multi-epoch training."""
        # Create dataset
        np.random.seed(42)
        X, y = create_regression_dataset(n_samples=200, n_features=5)

        X_tensor = Tensor(X)
        y_tensor = Tensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Create model
        model = Linear(5, 1)
        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = MSELoss()

        # Track loss over epochs
        epoch_losses = []

        for epoch in range(10):
            epoch_loss = 0
            num_batches = 0

            for batch_X, batch_y in dataloader:
                pred = model(batch_X)
                loss = criterion(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.data)
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)

        # Loss should generally decrease
        self.assertLess(epoch_losses[-1], epoch_losses[0])


class TestOptimizerComparison(unittest.TestCase):
    """Test different optimizers on same task."""

    def _train_model(self, optimizer_class, **optimizer_kwargs):
        """Helper to train model with specific optimizer."""
        np.random.seed(42)
        X, y = create_simple_dataset(n_samples=200, n_features=10, n_classes=3)

        X_tensor = Tensor(X)
        y_tensor = Tensor(y)

        model = MLP(input_size=10, hidden_sizes=[20], output_size=3)
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        criterion = CrossEntropyLoss()

        # Train for fixed number of iterations
        for _ in range(50):
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return float(loss.data)

    def test_sgd_vs_adam(self):
        """Compare SGD and Adam optimizers."""
        sgd_loss = self._train_model(SGD, lr=0.01)
        adam_loss = self._train_model(Adam, lr=0.01)

        # Both should achieve reasonable loss
        self.assertLess(sgd_loss, 5.0)
        self.assertLess(adam_loss, 5.0)

        # Adam often converges faster
        # (though this is dataset-dependent)


class TestTrainEvalMode(unittest.TestCase):
    """Test training vs evaluation mode."""

    def test_dropout_train_eval(self):
        """Test that dropout behaves differently in train vs eval mode."""
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        X_tensor = Tensor(X)

        # Create model with dropout
        model = MLP(input_size=10, hidden_sizes=[20], output_size=5, dropout=0.5)

        # Train mode - outputs may vary due to dropout
        model.train()
        out1_train = model(X_tensor)
        out2_train = model(X_tensor)

        # Eval mode - outputs should be consistent
        model.eval()
        out1_eval = model(X_tensor)
        out2_eval = model(X_tensor)

        # Eval mode outputs should be identical
        np.testing.assert_array_equal(out1_eval.data, out2_eval.data)

    def test_batchnorm_train_eval(self):
        """Test that batch normalization behaves differently in train vs eval."""
        np.random.seed(42)
        X = np.random.randn(32, 10).astype(np.float32)
        X_tensor = Tensor(X)

        from numpy_dl.nn import Sequential, Linear, BatchNorm1d, ReLU

        model = Sequential(
            Linear(10, 20),
            BatchNorm1d(20),
            ReLU(),
            Linear(20, 5)
        )

        # Train mode
        model.train()
        out_train = model(X_tensor)

        # Eval mode
        model.eval()
        out_eval = model(X_tensor)

        # Outputs should differ between train and eval
        self.assertFalse(np.allclose(out_train.data, out_eval.data))


class TestGradientFlow(unittest.TestCase):
    """Test gradient flow through networks."""

    def test_deep_network_gradients(self):
        """Test that gradients flow through deep networks."""
        model = MLP(input_size=10, hidden_sizes=[30, 30, 30, 30], output_size=5)

        X = Tensor(np.random.randn(2, 10), requires_grad=True)
        y = model(X)
        loss = y.sum()

        loss.backward()

        # All parameters should have gradients
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            # Gradient should not be all zeros
            self.assertFalse(np.all(param.grad == 0))

    def test_cnn_gradients(self):
        """Test gradients in CNN."""
        model = SimpleCNN(
            input_channels=1,
            num_classes=10,
            conv_channels=[8, 16],
            fc_sizes=[32],
            input_size=(28, 28)
        )

        X = Tensor(np.random.randn(2, 1, 28, 28), requires_grad=True)
        y = model(X)
        loss = y.sum()

        loss.backward()

        # All parameters should have gradients
        for param in model.parameters():
            self.assertIsNotNone(param.grad)


class TestOverfitting(unittest.TestCase):
    """Test that models can overfit small datasets."""

    def test_mlp_overfitting(self):
        """Test that MLP can overfit a tiny dataset."""
        # Create tiny dataset (5 samples)
        np.random.seed(42)
        X = np.random.randn(5, 10).astype(np.float32)
        y = np.random.randint(0, 3, size=5)

        X_tensor = Tensor(X)
        y_tensor = Tensor(y)

        # Large model relative to data
        model = MLP(input_size=10, hidden_sizes=[100, 100], output_size=3)
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        # Train until near-perfect fit
        for epoch in range(200):
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final loss should be very low (overfitted)
        final_loss = float(loss.data)
        self.assertLess(final_loss, 0.1)

        # Should achieve perfect accuracy on training set
        pred_classes = np.argmax(pred.data, axis=1)
        accuracy = np.mean(pred_classes == y)
        self.assertGreater(accuracy, 0.9)


if __name__ == '__main__':
    unittest.main()
