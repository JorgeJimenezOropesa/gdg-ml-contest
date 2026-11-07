from dataclasses import dataclass

@dataclass(frozen=True)
class AugmentConfig:

    # Allows deactivation of augmentations. False if we don't want to augmentate the data
    enabled: bool = True

    # Soft polynomial tendency
    poly_prob: float = 0.3
    poly_linear_max: float = 1.0
    poly_quad_max: float = 0.5
    # False if we want the same tendency for all channels
    poly_per_channel: bool = False

    # Gaussian noise (independent from the channel)
    noise_prob: float = 0.9
    noise_rel_range: tuple[float, float] = (0.15, 0.5)

    # NaN intervals
    nan_prob: float = 0.5
    nan_intervals_mean: float = 0.6 # mean number of NaN per channel
    nan_len_mean: float = 3.0
    nan_len_max: int = 40


AUGMENT = AugmentConfig()