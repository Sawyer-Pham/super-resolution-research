import torch
from torch import nn
from torch.nn import functional as F


class SRCNN(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 5, padding=2),
        )

    def forward(self, x):
        x = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
        return self.layers(x)


class ESPCN(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 3 * scale**2, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class EDSR(nn.Module):
    def __init__(self, scale=2, channels=64, blocks=16):
        super().__init__()
        if scale not in (2, 3, 4):
            raise ValueError("Scale must be 2, 3, or 4")

        self.register_buffer(
            "mean",
            torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1),
        )
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(blocks)],
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        upsampling = []
        factors = [2, 2] if scale == 4 else [scale]
        for factor in factors:
            upsampling.extend([
                nn.Conv2d(
                    channels, channels * factor**2, 3, padding=1
                ),
                nn.PixelShuffle(factor),
            ])

        self.tail = nn.Sequential(
            *upsampling,
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x):
        features = self.head(x - self.mean)
        features = features + self.body(features)
        return self.tail(features) + self.mean