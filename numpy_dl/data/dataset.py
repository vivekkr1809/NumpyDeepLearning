"""Dataset base classes."""

from typing import Tuple, Optional, Callable
import numpy as np


class Dataset:
    """
    Base class for all datasets.

    All datasets should subclass this and implement __len__ and __getitem__.
    """

    def __len__(self) -> int:
        """Return the size of the dataset."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple:
        """Get a sample from the dataset."""
        raise NotImplementedError


class TensorDataset(Dataset):
    """
    Dataset wrapping tensors/arrays.

    Each sample is retrieved by indexing tensors along the first dimension.
    """

    def __init__(self, *arrays):
        """
        Initialize TensorDataset.

        Args:
            *arrays: Arrays to wrap (must have same length)
        """
        assert all(arrays[0].shape[0] == arr.shape[0] for arr in arrays), \
            "All arrays must have the same length"
        self.arrays = arrays

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.arrays[0].shape[0]

    def __getitem__(self, index: int) -> Tuple:
        """Get a sample from the dataset."""
        return tuple(arr[index] for arr in self.arrays)


class TransformDataset(Dataset):
    """Dataset with transformations applied."""

    def __init__(self, dataset: Dataset, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize TransformDataset.

        Args:
            dataset: Base dataset
            transform: Transform to apply to inputs
            target_transform: Transform to apply to targets
        """
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple:
        """Get a sample with transforms applied."""
        sample = self.dataset[index]

        if len(sample) == 2:
            x, y = sample
            if self.transform:
                x = self.transform(x)
            if self.target_transform:
                y = self.target_transform(y)
            return x, y
        else:
            if self.transform:
                sample = tuple(self.transform(s) for s in sample)
            return sample


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset: The whole dataset
        indices: Indices in the whole dataset to include
    """

    def __init__(self, dataset: Dataset, indices: np.ndarray):
        """
        Initialize Subset.

        Args:
            dataset: Base dataset
            indices: Indices to include in subset
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        """Return the size of the subset."""
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple:
        """Get a sample from the subset."""
        return self.dataset[self.indices[index]]


def train_test_split(dataset: Dataset, test_size: float = 0.2,
                    shuffle: bool = True, seed: Optional[int] = None) -> Tuple[Subset, Subset]:
    """
    Split dataset into train and test subsets.

    Args:
        dataset: Dataset to split
        test_size: Proportion of dataset to include in test split
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(dataset)
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(n * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def train_val_test_split(dataset: Dataset, val_size: float = 0.1, test_size: float = 0.1,
                         shuffle: bool = True, seed: Optional[int] = None) -> Tuple[Subset, Subset, Subset]:
    """
    Split dataset into train, validation, and test subsets.

    Args:
        dataset: Dataset to split
        val_size: Proportion of dataset to include in validation split
        test_size: Proportion of dataset to include in test split
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(dataset)
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    test_split = int(n * (1 - test_size))
    val_split = int(test_split * (1 - val_size / (1 - test_size)))

    train_indices = indices[:val_split]
    val_indices = indices[val_split:test_split]
    test_indices = indices[test_split:]

    return (Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices))
