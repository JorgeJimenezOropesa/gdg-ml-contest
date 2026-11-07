import numpy as np


def inference_preprocess_sample(series: np.ndarray, meta: dict):
    sample_rate = meta["sample_rate"]

    # Handle nans
    series = fill_nan(series)

    # Adjust series so frequency matches
    # If sample rate is 1, return both halfs
    # for better accuracy on inference
    if sample_rate == 1:
        series_list = [series[:, 0:128], series[:, 128:256]]
    else:  # sample rate == 2
        series = series.reshape(series.shape[0], -1, 2).mean(axis=-1)
        series_list = [series]

    # Normalize
    series_list = [normalize(s) for s in series_list]

    return series_list


def train_preprocess_sample(series: np.ndarray, meta: dict):
    sample_rate = meta["sample_rate"]

    # Handle nans
    series = fill_nan(series)

    # Adjust series so frequency matches
    if sample_rate == 1:
        offset = np.random.randint(0, 128)
        series = series[:, offset : offset + 128]
    else:  # sample rate == 2
        series = series.reshape(series.shape[0], -1, 2).mean(axis=-1)

    # Normalize channels
    series = normalize(series)

    return series


def fill_nan(series: np.ndarray) -> np.ndarray:
    """Interpolate NaNs linearly across the temporal axis for each sensor channel."""
    series = np.array(series, dtype=np.float32, copy=True)
    n_channels, length = series.shape
    time_indices = np.arange(length)

    for ch in range(n_channels):
        channel_data = series[ch]
        nan_mask = np.isnan(channel_data)

        if not nan_mask.any():
            continue

        # If an entire sensor is dead (all NaNs), fallback to zero
        if nan_mask.all():
            channel_data[:] = 0.0
            continue

        # np.interp linearly interpolates internal gaps and repeats border values
        valid_indices = time_indices[~nan_mask]
        valid_values = channel_data[~nan_mask]
        channel_data[nan_mask] = np.interp(
            time_indices[nan_mask],
            valid_indices,
            valid_values,
        )

    return series


def normalize(series: np.ndarray) -> np.ndarray:
    """Apply per-sample, per-channel Z-score normalization in float32."""
    mean = series.mean(axis=-1, keepdims=True)
    std = series.std(axis=-1, keepdims=True)
    # Prevent division by zero on constant channels or dead sensors filled with 0.0
    safe_std = np.where(std > np.float32(1e-6), std, np.float32(1.0))
    return (series - mean) / safe_std
