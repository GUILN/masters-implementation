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
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
            for k in kernel_sizes
        ])
        self._dropout = nn.Dropout(dropout)
        self._proj = nn.Conv1d(
            out_channels * len(kernel_sizes),
            out_channels,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, features, T]
        outs = [branch(x) for branch in self._branches]
        out = torch.cat(outs, dim=1)  # Concatenate along feature dimension
        out = self._proj(out)
        return self._dropout(out)
