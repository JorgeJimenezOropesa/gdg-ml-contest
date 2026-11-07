import numpy as np


def preprocess_sample(series: np.ndarray, meta: dict):
    sample_rate = meta["sample_rate"]
    series = fill_nan(series)
    series = match_sample_rate(series, sample_rate)
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


def match_sample_rate(series: np.ndarray, sample_rate: float) -> np.ndarray:
    if sample_rate == 1:
        offset = np.random.randint(0, 128)
        return series[:, offset : offset + 128]
    else:  # sample rate == 2
        return series.reshape(series.shape[0], -1, 2).mean(axis=-1)
