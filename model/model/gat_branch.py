from typing import Literal
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

PoolingType = Literal["mean", "max"]


class GATBranch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        pooling: PoolingType = "mean"
    ):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim)
        self.gat2 = GATConv(hidden_dim, out_dim)
        self._pooling = pooling

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        if self._pooling == "mean":
            return global_mean_pool(
                x,
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            )
        else:
            return global_max_pool(
                x,
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            )
