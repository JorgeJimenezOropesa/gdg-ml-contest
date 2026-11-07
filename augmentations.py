import numpy as np

from config import AugmentConfig

def _chance(rng: np.random.Generator, prob: float) -> bool:
    # Returns True with probability prob
    return rng.random() < prob


def add_poly_trend(x: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    # Adds a polynomial trend to the data
    n_ch, length = x.shape
    shape = (n_ch, 1) if cfg.poly_per_channel else (1, 1)
    t = np.linspace(-1, 1, length)
    c1 = rng.uniform(-cfg.poly_linear_max, cfg.poly_linear_max, size=shape)
    c2 = rng.uniform(-cfg.poly_quad_max, cfg.poly_quad_max, size=shape)
    return x + c1 * t + c2 * t**2


def add_gaussian_noise(x: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    # Controls the std deviation for each channel, ignoring NaNs. If it's 0 or NaN, we use 1
    ch_std = np.nanstd(x, axis=1, keepdims=True)
    ch_std = np.where(np.isfinite(ch_std) & (ch_std > 1e-6), ch_std, 1)
    rel = rng.uniform(*cfg.noise_rel_range, size=(x.shape[0], 1))
    return x + rng.normal(size=x.shape) * rel * ch_std


def add_nan_intervals(x: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    # Adds NaN intervals to the data
    out = x.copy()
    n_ch, length = out.shape

    for c in range(n_ch):

        original = out[c].copy()
        for _ in range(rng.poisson(cfg.nan_intervals_mean)):
            gap = int(min(rng.geometric(1 / cfg.nan_len_mean), cfg.nan_len_max, length - 1))
            start = int(rng.integers(0, length - gap + 1))
            out[c, start : start + gap] = np.nan

        # We avoid leaving the entire channel as NaN
        if np.isnan(out[c]).all():
            out[c] = original

    return out


def augment(x: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    # Applies augmentations to the data
    if not cfg.enabled:
        return x

    if _chance(rng, cfg.poly_prob):
        x = add_poly_trend(x, cfg, rng)

    if _chance(rng, cfg.noise_prob):
        x = add_gaussian_noise(x, cfg, rng)

    if _chance(rng, cfg.nan_prob):
        x = add_nan_intervals(x, cfg, rng)

    return x.astype(np.float32)