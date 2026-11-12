from dataclasses import dataclass

@dataclass(frozen=True)
class AugmentConfig:

    # Allows deactivation of augmentations. False if we don't want to augmentate the data
    enabled: bool = True

    poly_prob: float = 0.819816158610183
    poly_linear_max: float = 0.8016791720261657
    poly_quad_max: float = 0.14420613635145557
    poly_per_channel: bool = False

    noise_prob: float = 0.8590477538171948
    noise_rel_range: tuple[float, float] = (0.1024621596530517, 0.721041448729279)

    nan_prob: float = 0.29404034812512714
    nan_intervals_mean: float = 1.7955251393881067
    nan_len_mean: float = 18.054766797037736
    nan_len_max: int = 42

AUGMENT = AugmentConfig()