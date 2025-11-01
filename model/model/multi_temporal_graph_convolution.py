from typing import List
import torch
import torch.nn as nn


class MultiTemporalGC(nn.Module):
    """
    Multi-Temporal Graph Convolutional Block (MT-GC)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.3,
    ):
        super().__init__()
        self._branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2),
                # LayerNorm expects (N, L, C) -> so we’ll permute before and after
                nn.ReLU(),
            )
            for k in kernel_sizes
        ])
        self._layer_norms = nn.ModuleList([
            nn.LayerNorm(out_channels)
            for _ in kernel_sizes
        ])
        self._dropout = nn.Dropout(dropout)
        self._proj = nn.Conv1d(
            out_channels * len(kernel_sizes),
            out_channels,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, features, T]
        outs = []
        for branch, ln in zip(self._branches, self._layer_norms):
            out = branch(x)  # [B, C, T]
            out = out.permute(0, 2, 1)  # [B, T, C]
            out = ln(out)               # normalize over C
            out = out.permute(0, 2, 1)  # back to [B, C, T]
            outs.append(out)

        out = torch.cat(outs, dim=1)
        out = self._proj(out)
        return self._dropout(out)
