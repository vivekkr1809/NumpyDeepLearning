"""
Audio Classification Example

Train a CNN model for audio classification using spectrograms.
"""

import numpy as np
import numpy_dl as ndl
from numpy_dl.models import SimpleCNN
from numpy_dl.optim import Adam
from numpy_dl.loss import CrossEntropyLoss
from numpy_dl.data import TensorDataset, DataLoader
from numpy_dl.utils import accuracy, MetricTracker
from numpy_dl.tracking import ExperimentTracker


def generate_synthetic_audio_data(n_samples=300, n_classes=10):
    """
    Generate synthetic audio spectrogram data.

    In practice, would load actual audio files and convert to spectrograms.
    """
    print("Generating synthetic audio spectrogram data...")

    # Simulate mel-spectrograms: (batch, channels, freq_bins, time_frames)
    # Typical dimensions: 1 channel, 128 frequency bins, 128 time frames
    X = np.random.randn(n_samples, 1, 128, 128).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)

    return X, y


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    metric_tracker = MetricTracker()

    for batch_idx, (data, target) in enumerate(dataloader):
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

        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    return metric_tracker.compute()


def main():
    """Main training function."""
    # Hyperparameters
    n_classes = 10
    batch_size = 16
    epochs = 3
    learning_rate = 0.001
    device = 'cpu'

    # Initialize tracker
    tracker = ExperimentTracker('audio_cnn_classification')
    tracker.log_hyperparameters(
        n_classes=n_classes,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        model='SimpleCNN',
        input_type='mel_spectrogram'
    )

    # Generate data
    X_train, y_train = generate_synthetic_audio_data(n_samples=300, n_classes=n_classes)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create model - CNN works well for spectrograms
    model = SimpleCNN(
        input_channels=1,
        num_classes=n_classes,
        conv_channels=[32, 64, 128],
        fc_sizes=[256],
        input_size=(128, 128),
        dropout=0.5
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
    print("\nNote: This example uses synthetic spectrogram data.")
    print("For real audio classification:")
    print("1. Load audio files (e.g., using librosa)")
    print("2. Convert to spectrograms/mel-spectrograms")
    print("3. Use the same CNN architecture")


if __name__ == '__main__':
    main()
