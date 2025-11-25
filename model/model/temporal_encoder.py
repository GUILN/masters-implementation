import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(channels, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, channels)  # residual projection

    def forward(self, x):
        # x: [B, C, T] -> [B, T, C]
        x_t = x.permute(0, 2, 1)
        out, _ = self.gru(x_t)          # [B, T, hidden]
        out = self.proj(out)            # [B, T, C]
        return out.permute(0, 2, 1)     # [B, C, T]
