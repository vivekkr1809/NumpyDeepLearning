"""
Image Segmentation Example

Train a U-Net model for image segmentation.
"""

import numpy as np
import numpy_dl as ndl
from numpy_dl.models import UNet
from numpy_dl.optim import Adam
from numpy_dl.loss import BCEWithLogitsLoss
from numpy_dl.data import TensorDataset, DataLoader
from numpy_dl.tracking import ExperimentTracker


def generate_synthetic_segmentation_data(n_samples=100):
    """Generate synthetic segmentation data."""
    print("Generating synthetic segmentation data...")

    # Create images with simple shapes
    images = np.random.randn(n_samples, 3, 128, 128).astype(np.float32)
    masks = np.random.randint(0, 2, (n_samples, 1, 128, 128)).astype(np.float32)

    return images, masks


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

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

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / n_batches


def main():
    """Main training function."""
    # Hyperparameters
    batch_size = 4
    epochs = 3
    learning_rate = 0.001
    device = 'cpu'

    # Initialize tracker
    tracker = ExperimentTracker('unet_segmentation')
    tracker.log_hyperparameters(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        model='UNet'
    )

    # Generate data
    X, y = generate_synthetic_segmentation_data(n_samples=50)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = UNet(input_channels=3, num_classes=1, features=[32, 64, 128])
    model.to(device)

    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print(f"Model: {model}")
    print(f"Training samples: {len(dataset)}")

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")

        tracker.log_metrics(epoch=epoch, train_loss=train_loss)

    tracker.finish()
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
