"""DataLoader for batching and iterating over datasets."""

import numpy as np
from typing import Optional, Iterator
from numpy_dl.data.dataset import Dataset


class DataLoader:
    """
    Data loader for iterating over datasets in batches.

    Combines a dataset and a sampler, and provides an iterable over the dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize DataLoader.

        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data at every epoch
            drop_last: Whether to drop the last incomplete batch
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self.n_samples = len(dataset)
        self.n_batches = self.n_samples // batch_size
        if not drop_last and self.n_samples % batch_size != 0:
            self.n_batches += 1

    def __len__(self) -> int:
        """Return the number of batches."""
        return self.n_batches

    def __iter__(self) -> Iterator:
        """Iterate over batches."""
        if self.seed is not None:
            np.random.seed(self.seed)

        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]

            # Skip incomplete batch if drop_last is True
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # Gather batch samples
            batch = [self.dataset[idx] for idx in batch_indices]

            # Stack samples into batches
            if isinstance(batch[0], tuple):
                # Multiple outputs (e.g., x, y)
                num_outputs = len(batch[0])
                batched = []
                for i in range(num_outputs):
                    items = [item[i] for item in batch]
                    # Stack into array
                    if isinstance(items[0], np.ndarray):
                        batched.append(np.stack(items))
                    else:
                        batched.append(np.array(items))
                yield tuple(batched)
            else:
                # Single output
                if isinstance(batch[0], np.ndarray):
                    yield np.stack(batch)
                else:
                    yield np.array(batch)


def collate_fn(batch):
    """
    Default collate function.

    Stacks samples into batches.

    Args:
        batch: List of samples

    Returns:
        Batched data
    """
    if isinstance(batch[0], tuple):
        return tuple(np.stack([item[i] for item in batch]) for i in range(len(batch[0])))
    return np.stack(batch)
