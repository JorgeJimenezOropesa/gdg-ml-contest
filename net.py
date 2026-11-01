import torch
import torch.nn as nn


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
    def __init__(self, in_channels: int = 4, num_classes: int = 8):
        super().__init__()

        # Stem: maps raw 4 sensor channels to 32 feature channels (L=256)
        self.stem = nn.Conv1d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=7,
            padding="same",
            padding_mode="zeros",
            bias=True,
        )

        # Stage 1: resolution 256, 32 channels
        self.stage1 = nn.Sequential(
            ResBlock1D(channels=32, kernel_size=7),
            ResBlock1D(channels=32, kernel_size=7),
        )
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)  # 256 -> 128

        # Stage 2: resolution 128, 32 channels
        self.stage2 = nn.Sequential(
            ResBlock1D(channels=32, kernel_size=5),
            ResBlock1D(channels=32, kernel_size=5),
        )
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)  # 128 -> 64

        # Stage 3: resolution 64, 32 channels
        self.stage3 = nn.Sequential(
            ResBlock1D(channels=32, kernel_size=3),
            ResBlock1D(channels=32, kernel_size=3),
        )
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)  # 64 -> 32

        # Stage 4: resolution 32, 32 channels
        self.stage4 = nn.Sequential(
            ResBlock1D(channels=32, kernel_size=3),
            ResBlock1D(channels=32, kernel_size=3),
        )

        # Global Average Pooling: collapses temporal length (B, 32, 32) -> (B, 32, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Linear classification head directly from pooled features (32 -> 8)
        self.classifier = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)  # (Batch, 32, 256)

        # Stage 1
        x = self.stage1(x)  # (Batch, 32, 256)
        x = self.pool1(x)  # (Batch, 32, 128)

        # Stage 2
        x = self.stage2(x)  # (Batch, 32, 128)
        x = self.pool2(x)  # (Batch, 32, 64)

        # Stage 3
        x = self.stage3(x)  # (Batch, 32, 64)
        x = self.pool3(x)  # (Batch, 32, 32)

        # Stage 4
        x = self.stage4(x)  # (Batch, 32, 32)

        # Global average pooling and flatten
        x = self.gap(x)  # (Batch, 32, 1)
        x = torch.flatten(x, start_dim=1)  # (Batch, 32)

        # Logits
        logits = self.classifier(x)  # (Batch, 8)
        return logits
