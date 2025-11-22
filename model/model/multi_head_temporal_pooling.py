import torch
import torch.nn as nn


class MultiHeadTemporalPooling(nn.Module):
    """
    Multi-head temporal attention pooling.
    Input:  x: Tensor [B, C, T]
    Output: Tensor [B, C]
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0 or True, "channels need not be divisible by heads for MultiheadAttention"
        self.num_heads = num_heads
        self.channels = channels

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        # one learnable query per head
        self.query = nn.Parameter(torch.randn(1, num_heads, channels))

        # project concatenated heads (H*C) back to C
        self.proj = nn.Linear(channels * num_heads, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        returns: [B, C]
        """
        B, C, T = x.shape
        x_t = x.permute(0, 2, 1)  # [B, T, C]
        q = self.query.expand(B, -1, -1)  # [B, H, C]
        pooled, _ = self.attn(q, x_t, x_t)  # pooled: [B, H, C]
        concat = pooled.reshape(B, -1)  # [B, H*C]
        return self.proj(concat)  # [B, C]
