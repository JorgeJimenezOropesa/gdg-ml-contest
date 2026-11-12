import numpy as np


def inference_preprocess_sample(series: np.ndarray, meta: dict):
    sample_rate = meta["sample_rate"]

    # Handle nans
    series = fill_nan(despike(series))

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
    series = fill_nan(despike(series))

    # Adjust series so frequency matches
    if sample_rate == 1:
        offset = np.random.randint(0, 128)
        series = series[:, offset : offset + 128]
    else:  # sample rate == 2
        series = series.reshape(series.shape[0], -1, 2).mean(axis=-1)

    # Normalize channels
    series = normalize(series)

    return series


_COORDS = {}


def fill_nan(series: np.ndarray) -> np.ndarray:
    """Interpolate NaNs linearly across the temporal axis, all channels in one np.interp call."""
    series = np.array(series, dtype=np.float32, copy=True)
    nan = np.isnan(series)

    if not nan.any():
        return series
    n_channels, length = series.shape

    dead = nan.all(axis=1)
    
    if dead.any():  # all NaNs
        series[dead] = 0.0
        nan[dead] = False

    co = _COORDS.get(series.shape)

    if co is None:  
        i = np.arange(n_channels * length)
        co = _COORDS[series.shape] = (i // length) * 1e7 + (i % length)

    flat, nanf = series.reshape(-1), nan.reshape(-1)
    flat[nanf] = np.interp(co[nanf], co[~nanf], flat[~nanf])
    return series

def _median(x: np.ndarray) -> np.ndarray:
    """Faster implementation of np.median to gain speed on large arrays"""
    h = x.shape[-1] // 2
    if x.shape[-1] % 2:
        return np.partition(x, h, axis=-1)[..., h:h + 1]
    p = np.partition(x, (h - 1, h), axis=-1)
    return 0.5 * (p[..., h - 1:h] + p[..., h:h + 1])


def despike(series: np.ndarray, t: float = 8.0) -> np.ndarray:
    """Marks spikes as NaN."""
    x = fill_nan(series)
    med = _median(x)
    mad = 1.4826 * _median(np.abs(x - med)) + 1e-6
    spikes = np.abs(x - med) > t * mad
    if not spikes.any():
        return x  # there are no spikes left
    out = np.array(series, dtype=np.float32, copy=True)
    out[spikes] = np.nan
    return out

def channel_extras(series: np.ndarray, sample_rate: float) -> np.ndarray:
    """5 extra inputs: robust log-scale of each channel minus their mean (clipped to +-3) + SR flag."""
    x = fill_nan(series)
    med = _median(x)
    ls = np.log(1.4826 * _median(np.abs(x - med))[:, 0] + 1e-6)
    ls = np.clip(ls - ls.mean(), -3.0, 3.0)
    return np.append(ls, 1.0 if sample_rate >= 1.5 else 0.0).astype(np.float32)


def normalize(series: np.ndarray) -> np.ndarray:
    """Apply per-sample, per-channel Z-score normalization in float32."""
    mean = series.mean(axis=-1, keepdims=True)
    std = series.std(axis=-1, keepdims=True)
    # Prevent division by zero on constant channels or dead sensors filled with 0.0
    safe_std = np.where(std > np.float32(1e-6), std, np.float32(1.0))
    return (series - mean) / safe_std
