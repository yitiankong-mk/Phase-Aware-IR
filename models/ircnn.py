"""IRCNN architecture adapted to 16-label IR classification."""

import torch
from torch import nn


class IRCNNPaper(nn.Module):
    def __init__(
        self,
        input_length: int = 600,
        num_classes: int = 16,
        dropout: float = 0.48599073736368,
        init_std: float = 0.05,
    ) -> None:
        super().__init__()
        if input_length != 600:
            raise ValueError("IRCNNPaper expects input_length=600.")
        self.features = nn.Sequential(
            nn.Conv1d(1, 31, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(31),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(31, 62, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(62),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(62 * 150, 4927),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4927, 2785),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2785, 1574),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1574, num_classes),
        )
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
