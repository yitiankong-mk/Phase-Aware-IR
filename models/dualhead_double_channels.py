"""Dual-head CNN variant with doubled convolutional channels."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class DualHeadDoubleChannelsCNN(nn.Module):
    def __init__(
        self,
        input_length: int,
        num_classes: int,
        embed_dim: int = 128,
        mlp_drop1: float = 0.25,
        mlp_drop2: float = 0.25,
        mlp_drop3: float = 0.10,
    ) -> None:
        super().__init__()
        self.medium_conv = nn.Sequential(
            nn.Conv1d(1, 64, 71, padding=35), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 61, padding=30), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, embed_dim, 25, padding=12), nn.BatchNorm1d(embed_dim), nn.ReLU(),
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 64, 25, padding=12), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 17, padding=8), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, embed_dim, 11, padding=5), nn.BatchNorm1d(embed_dim), nn.ReLU(),
        )
        self.q_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.k_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.v_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.scale = math.sqrt(embed_dim)
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(embed_dim * 2, embed_dim, 1), nn.BatchNorm1d(embed_dim), nn.ReLU()
        )
        self.refine_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm1d(embed_dim), nn.GELU()
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * (input_length // 4), 1024),
            nn.GELU(), nn.Dropout(mlp_drop1),
            nn.Linear(1024, 512),
            nn.GELU(), nn.Dropout(mlp_drop2),
            nn.Linear(512, 256),
            nn.GELU(), nn.Dropout(mlp_drop3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        medium = self.medium_conv(x)
        local = self.local_conv(x)
        length = min(medium.shape[-1], local.shape[-1])
        medium = F.adaptive_avg_pool1d(medium, length)
        local = F.adaptive_avg_pool1d(local, length)
        query = self.q_proj(medium).permute(0, 2, 1)
        key = self.k_proj(local)
        value = self.v_proj(local).permute(0, 2, 1)
        attention = torch.softmax(torch.bmm(query, key) / self.scale, dim=-1)
        attended = self.dropout(torch.bmm(attention, value).permute(0, 2, 1))
        fused = self.refine_conv(self.fuse_conv(torch.cat([medium, attended], dim=1)))
        return self.fc(fused.view(batch_size, -1))
