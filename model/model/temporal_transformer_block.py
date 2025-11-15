import torch
import torch.nn as nn


class TemporalTransformerBlock(nn.Module):
    """
    Lightweight Transformer block for temporal modeling.
    Input:  [B, C, T]
    Output: [B, C, T]
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.pos_embedding = nn.Parameter(torch.randn(1, channels, 500))  # supports up to 500 frames

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape [B, C, T]
        Returns:
            shape [B, C, T]
        """
        B, C, T = x.shape

        # Add positional embeddings
        pos = self.pos_embedding[:, :, :T]
        x = x + pos

        # Transformer expects [B, T, C]
        x_t = x.permute(0, 2, 1)

        # Multi-head attention
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x_t = self.norm1(x_t + attn_out)

        # Feed-forward network
        ff_out = self.ff(x_t)
        x_t = self.norm2(x_t + ff_out)

        # Back to [B, C, T]
        return x_t.permute(0, 2, 1)
