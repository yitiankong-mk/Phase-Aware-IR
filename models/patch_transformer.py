"""Patch-transformer model for one-dimensional IR spectra."""

import math

import torch
from torch import nn


class IRPatchTransformerCNN(nn.Module):
    def __init__(
        self,
        input_length: int,
        num_classes: int,
        d_model: int = 128,
        patch_size: int = 16,
        stride: int = 8,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        head_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.patch_embed = nn.Conv1d(
            1,
            d_model,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        token_count = math.floor(
            (input_length + 2 * (patch_size // 2) - patch_size) / stride
        ) + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, d_model))
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, num_classes),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.kaiming_normal_(self.patch_embed.weight, nonlinearity="linear")
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).transpose(1, 2)
        x = self.embed_norm(x + self.pos_embed[:, : x.shape[1], :])
        x = self.encoder(self.embed_drop(x))
        pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        return self.fc(pooled)
