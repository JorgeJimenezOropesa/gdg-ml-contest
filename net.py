import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class StageConfig:
    res_kernel_size: int  # odd
    down_kernel_size: int  # odd
    num_blocks: int
    out_channels: int


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="zeros",
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="zeros",
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += res
        return out


class ConvNet(nn.Module):
    def __init__(
        self,
        stem_channels: int,
        stem_kernel_size: int,
        stages: list[StageConfig],
        in_channels: int = 4,
        num_classes: int = 8,
        dense_hidden_dim: int | None = 48,
        n_extra: int = 5,
    ):
        super().__init__()

        # Initial projection of channels to (256, 4) to (128, 32)
        self.stem = nn.Conv1d(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=stem_kernel_size,
            padding=stem_kernel_size // 2,
            padding_mode="zeros",
            stride=1,
            bias=True,
        )

        # The stages of the net
        self.stages = nn.ModuleList()
        current_channels = stem_channels

        for idx, stage_cfg in enumerate(stages):
            layers: list[nn.Module] = []

            # Chain residual blocks
            for _ in range(stage_cfg.num_blocks):
                layers.append(
                    ResBlock1D(
                        channels=current_channels,
                        kernel_size=stage_cfg.res_kernel_size,
                    )
                )

            # Downsample temporal dimension and adjust channels
            if idx < len(stages) - 1:
                # All stages except for the last one should downsample
                layers.append(
                    nn.Conv1d(
                        in_channels=current_channels,
                        out_channels=stage_cfg.out_channels,
                        kernel_size=stage_cfg.down_kernel_size,
                        padding=stage_cfg.down_kernel_size // 2,
                        stride=2,
                        bias=False,
                    )
                )
                current_channels = stage_cfg.out_channels
            else:
                # Last stage should not downsample, but activate
                layers.append(nn.BatchNorm1d(current_channels))
                layers.append(nn.ReLU())

            self.stages.append(nn.Sequential(*layers))

        # Classification head: [mean over time, max over time, extra inputs] -> MLP
        in_features = 2 * current_channels + n_extra
        classifier_layers: list[nn.Module] = []
        if dense_hidden_dim is not None:
            classifier_layers.extend(
                [
                    nn.Linear(in_features, dense_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            in_features = dense_hidden_dim
        classifier_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        # Initial dimension widening
        x = self.stem(x)

        # Stages
        for stage in self.stages:
            x = stage(x)

        # Pool over time (mean and max) and append the extra inputs
        feat = torch.cat([x.mean(dim=-1), x.amax(dim=-1), extra], dim=1)

        # Classify
        return self.classifier(feat)
