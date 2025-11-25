import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.Tanh(),
            nn.Linear(in_channels // 2, 1)
        )

    def forward(self, x):
        # x: [batch, channels, time]
        x = x.permute(0, 2, 1)           # -> [batch, time, channels]
        attn_weights = self.attn(x)      # -> [batch, time, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        x = torch.sum(x * attn_weights, dim=1)  # weighted sum over time
        return x                          # [batch, channels]
