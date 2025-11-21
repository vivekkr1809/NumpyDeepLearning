"""Visualization utilities for results and training progress."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple


sns.set_style('whitegrid')


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot training history (loss and metrics).

    Args:
        history: Dictionary of metric names to lists of values
        save_path: Path to save figure
        figsize: Figure size
    """
    metrics = list(history.keys())
    num_metrics = len(metrics)

    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    if num_metrics == 1:
        axes = [axes]

    for idx, (metric_name, values) in enumerate(history.items()):
        axes[idx].plot(values, marker='o', linestyle='-', linewidth=2, markersize=4)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric_name.capitalize())
        axes[idx].set_title(f'{metric_name.capitalize()} over Epochs')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = False
):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    plt.show()


def plot_loss_landscape(
    losses: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot loss landscape (2D heatmap).

    Args:
        losses: 2D array of loss values
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    sns.heatmap(losses, cmap='viridis', cbar_kws={'label': 'Loss'})

    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title('Loss Landscape')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss landscape to {save_path}")

    plt.show()


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot predictions vs targets (for regression).

    Args:
        predictions: Predicted values
        targets: True values
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    plt.scatter(targets, predictions, alpha=0.5, s=20)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()],
             'r--', lw=2, label='Perfect Prediction')

    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved predictions plot to {save_path}")

    plt.show()


def plot_images_grid(
    images: np.ndarray,
    labels: Optional[np.ndarray] = None,
    predictions: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    num_images: int = 16,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 12)
):
    """
    Plot grid of images with labels and predictions.

    Args:
        images: Array of images (N, H, W) or (N, H, W, C)
        labels: True labels
        predictions: Predicted labels
        class_names: List of class names
        num_images: Number of images to display
        save_path: Path to save figure
        figsize: Figure size
    """
    num_images = min(num_images, len(images))
    grid_size = int(np.ceil(np.sqrt(num_images)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()

    for idx in range(num_images):
        img = images[idx]

        # Handle different image formats
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:  # Grayscale
            img = img.squeeze()

        axes[idx].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[idx].axis('off')

        # Add title with label and prediction
        title = ""
        if labels is not None:
            label = class_names[labels[idx]] if class_names else labels[idx]
            title += f"True: {label}"
        if predictions is not None:
            pred = class_names[predictions[idx]] if class_names else predictions[idx]
            if title:
                title += f"\n"
            title += f"Pred: {pred}"
            # Color title based on correctness
            if labels is not None and labels[idx] != predictions[idx]:
                axes[idx].set_title(title, color='red', fontsize=8)
            else:
                axes[idx].set_title(title, fontsize=8)
        elif title:
            axes[idx].set_title(title, fontsize=8)

    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved images grid to {save_path}")

    plt.show()


def plot_feature_maps(
    feature_maps: np.ndarray,
    num_maps: int = 16,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 12)
):
    """
    Plot feature maps from a convolutional layer.

    Args:
        feature_maps: Feature maps of shape (C, H, W)
        num_maps: Number of feature maps to display
        save_path: Path to save figure
        figsize: Figure size
    """
    num_maps = min(num_maps, feature_maps.shape[0])
    grid_size = int(np.ceil(np.sqrt(num_maps)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()

    for idx in range(num_maps):
        axes[idx].imshow(feature_maps[idx], cmap='viridis')
        axes[idx].axis('off')
        axes[idx].set_title(f'Map {idx}', fontsize=8)

    for idx in range(num_maps, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Feature Maps', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature maps to {save_path}")

    plt.show()


def plot_gradient_flow(
    named_parameters: List[Tuple[str, np.ndarray]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot gradient flow through the network.

    Args:
        named_parameters: List of (name, gradient) tuples
        save_path: Path to save figure
        figsize: Figure size
    """
    avg_grads = []
    max_grads = []
    layers = []

    for name, grad in named_parameters:
        if grad is not None:
            layers.append(name)
            avg_grads.append(np.abs(grad).mean())
            max_grads.append(np.abs(grad).max())

    plt.figure(figsize=figsize)

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, label='max-gradient')
    plt.bar(np.arange(len(avg_grads)), avg_grads, alpha=0.5, label='mean-gradient')

    plt.hlines(0, 0, len(avg_grads) + 1, linewidth=2, color='k')
    plt.xticks(range(0, len(avg_grads), 1), layers, rotation='vertical')
    plt.xlim(left=0, right=len(avg_grads))
    plt.ylim(bottom=-0.001, top=max(max_grads) * 1.1)
    plt.xlabel('Layers')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Flow')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved gradient flow plot to {save_path}")

    plt.show()
