import torch
import torch.nn as nn

class MultiHeadTemporalPooling(nn.Module):
    """
    Multi-head temporal attention pooling.
    Input:  [B, C, T]
    Output: [B, C]
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape

        # reshape to [B, T, C]
        x_t = x.permute(0, 2, 1)

        # query is learned global token
        query = self.query.expand(B, -1, -1)  # [B, 1, C]

        pooled, _ = self.attn(query, x_t, x_t)

        return pooled.squeeze(1)  # [B, C]
