from pathlib import Path
from typing import Iterator
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import math

from preprocess import train_preprocess_sample
from competition_api import N_CLASSES, load_dataset


class SeriesDataset(Dataset):
    def __init__(self, npz_path: str | Path, is_train: bool = True):
        # Load dataset arrays
        ds = load_dataset(npz_path)
        self.series = ds.series  # Shape: (N, 4, 256)
        self.labels = ds.labels  # Shape: (N,)
        self.meta = ds.meta
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract sample array as an independent float32 copy
        raw_x = np.array(self.series[idx])
        y = int(self.labels[idx])
        meta = {k: v[idx].item() for k, v in self.meta.items()}

        # Handle NaNs
        x = train_preprocess_sample(raw_x, meta)

        # Apply data augmentation during training only
        if self.is_train:
            x = self.augment(x)

        tensor_x = torch.from_numpy(x)
        tensor_y = torch.tensor(y, dtype=torch.long)
        return tensor_x, tensor_y

    def augment(self, x: np.ndarray) -> np.ndarray:
        # Placeholder for future time-series augmentations
        return x


class BatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: np.ndarray | torch.Tensor,
        batch_size: int,
        num_classes: int = N_CLASSES,
    ):
        super().__init__()
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Group dataset indices by their target class
        self.class_indices: list[np.ndarray] = [
            np.where(self.labels == c)[0] for c in range(self.num_classes)
        ]

        smallest_class_size = min(len(idx) for idx in self.class_indices)

        # The total samples per epoch must be divisible by both the number of classes and batch size
        # That is, it must be divisible by the least common divisor
        lcm = math.lcm(self.num_classes, self.batch_size)
        self.total_samples = ((smallest_class_size * self.num_classes) // lcm) * lcm
        self.n_batches = self.total_samples // self.batch_size
        self.class_samples = self.total_samples // self.num_classes

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        epoch_indices: list[int] = []

        # Collect n_min samples from each class
        for indices in self.class_indices:
            selected = np.random.choice(indices, size=self.class_samples, replace=False)
            epoch_indices.extend(selected.tolist())

        # Shuffle the balanced pool
        np.random.shuffle(epoch_indices)

        # Truncate to exact multiple of batch_size
        epoch_indices = epoch_indices[: self.total_samples]

        # Yield batches of size batch_size
        for start_idx in range(0, self.total_samples, self.batch_size):
            yield epoch_indices[start_idx : start_idx + self.batch_size]
