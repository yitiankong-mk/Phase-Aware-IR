"""ResNet-50-1D model with a 128-channel stem."""

import torch
from torch import nn


class _Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = nn.Conv1d(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet50_1DCNN(nn.Module):
    def __init__(
        self,
        input_length: int,
        num_classes: int,
        in_channels: int = 1,
        base_channels: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.inplanes = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base_channels, blocks=3, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks=4, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks=6, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, blocks=3, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 8 * _Bottleneck1D.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, mid_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [_Bottleneck1D(self.inplanes, mid_channels, stride=stride)]
        self.inplanes = mid_channels * _Bottleneck1D.expansion
        layers.extend(
            _Bottleneck1D(self.inplanes, mid_channels, stride=1) for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.dropout(self.pool(x).flatten(1)))
