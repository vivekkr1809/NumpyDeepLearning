"""
Text Classification Example

Train an RNN model for text classification (e.g., sentiment analysis).
"""

import numpy as np
import numpy_dl as ndl
from numpy_dl.models import SimpleRNN
from numpy_dl.optim import Adam
from numpy_dl.loss import CrossEntropyLoss
from numpy_dl.data import TensorDataset, DataLoader
from numpy_dl.utils import accuracy, MetricTracker
from numpy_dl.tracking import ExperimentTracker


def generate_synthetic_text_data(n_samples=500, seq_len=20, vocab_size=1000):
    """Generate synthetic text data for classification."""
    print("Generating synthetic text data...")

    # Random sequences
    X = np.random.randint(0, vocab_size, (n_samples, seq_len)).astype(np.float32)

    # Random binary labels
    y = np.random.randint(0, 2, n_samples)

    return X, y


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    metric_tracker = MetricTracker()

    for batch_idx, (data, target) in enumerate(dataloader):
        # data shape: (batch_size, seq_len)
        # Reshape to (seq_len, batch_size, input_size)
        data = data[:, :, np.newaxis]  # Add feature dimension
        data = np.transpose(data, (1, 0, 2))  # (seq_len, batch, features)

        data = ndl.tensor(data, requires_grad=False, device=device)
        target = ndl.tensor(target, device=device)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        acc = accuracy(output.numpy(), target.numpy())
        metric_tracker.update(loss=loss.item(), accuracy=acc)

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    return metric_tracker.compute()


def main():
    """Main training function."""
    # Hyperparameters
    seq_len = 20
    vocab_size = 1000
    embedding_dim = 50
    hidden_size = 128
    num_layers = 2
    num_classes = 2
    batch_size = 32
    epochs = 5
    learning_rate = 0.001
    device = 'cpu'

    # Initialize tracker
    tracker = ExperimentTracker('rnn_text_classification')
    tracker.log_hyperparameters(
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        model='SimpleRNN-LSTM'
    )

    # Generate data
    X_train, y_train = generate_synthetic_text_data(n_samples=500, seq_len=seq_len)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create model (using word indices as simple features)
    model = SimpleRNN(
        input_size=1,  # Each word index is a single feature
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.3,
        cell_type='lstm'
    )
    model.to(device)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print(f"Model: {model}")
    print(f"Training samples: {len(train_dataset)}")

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.4f}")

        tracker.log_metrics(
            epoch=epoch,
            train_loss=train_metrics['loss'],
            train_accuracy=train_metrics['accuracy']
        )

    tracker.finish()
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
