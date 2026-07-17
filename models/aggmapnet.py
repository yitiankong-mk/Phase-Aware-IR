"""AggMapNet architecture used for mapped IR inputs."""

import torch
from torch import nn


class _InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, units: int) -> None:
        super().__init__()
        self.branch5 = nn.Conv2d(in_channels, units, kernel_size=5, padding=2)
        self.branch3 = nn.Conv2d(in_channels, units, kernel_size=3, padding=1)
        self.branch1 = nn.Conv2d(in_channels, units, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(torch.cat([self.branch5(x), self.branch3(x), self.branch1(x)], dim=1))


class AggMapNetPaperLike(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_labels: int,
        conv1_kernel_size: int = 13,
        dense_layers: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        batch_norm: bool = True,
        n_inception: int = 2,
    ) -> None:
        super().__init__()
        stem: list[nn.Module] = [
            nn.Conv2d(in_channels, 48, kernel_size=conv1_kernel_size, padding=conv1_kernel_size // 2)
        ]
        if batch_norm:
            stem.append(nn.BatchNorm2d(48, eps=1e-5, momentum=0.2))
        stem.append(nn.ReLU(inplace=True))
        self.stem = nn.Sequential(*stem)

        blocks: list[nn.Module] = []
        channels = 48
        for index in range(n_inception):
            units = 32 * (2**index)
            blocks.extend(
                [nn.MaxPool2d(kernel_size=3, stride=2, padding=1), _InceptionBlock(channels, units)]
            )
            channels = units * 3
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        classifier: list[nn.Module] = []
        in_features = channels
        for units in dense_layers:
            classifier.extend([nn.Linear(in_features, units), nn.ReLU(inplace=True)])
            if dropout:
                classifier.append(nn.Dropout(dropout))
            in_features = units
        classifier.append(nn.Linear(in_features, n_labels))
        self.fc = nn.Sequential(*classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.features(self.stem(x))
        return self.fc(self.pool(x).flatten(1))
