from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from competition_api import load_dataset

data_path = Path("data/train_100k.npz")
ds = load_dataset(data_path)

indices = np.arange(len(ds))
labels = np.asarray(ds.labels)

# Stratified split: 90k train / 10k val
train_idx, val_idx = train_test_split(
    indices, test_size=10000, random_state=67, stratify=labels
)

# Sort indices to keep memmap access linear and fast
train_idx = np.sort(train_idx)
val_idx = np.sort(val_idx)

# Save validation split in harness-compatible format
np.savez(
    "data/val_split.npz",
    series=ds.series[val_idx],
    labels=labels[val_idx],
    sample_index=ds.sample_index[val_idx],
    **{f"meta__{k}": v[val_idx] for k, v in ds.meta.items()},
)

# Save training split
np.savez(
    "data/train_split.npz",
    series=ds.series[train_idx],
    labels=labels[train_idx],
    sample_index=ds.sample_index[train_idx],
    **{f"meta__{k}": v[train_idx] for k, v in ds.meta.items()},
)

print(
    f"Dataset split complete: {len(train_idx)} train samples, {len(val_idx)} val samples."
)
