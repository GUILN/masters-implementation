import random
import torch
from torch_geometric.data import Data
from typing import List, Optional

def random_temporal_crop(
    graphs: List[Data],
    target_len: int,
) -> List[Data]:
    """
    Randomly crop a sequence of graphs to a fixed temporal length.
    If the sequence is shorter, it is padded with the last frame.
    """
    total_len = len(graphs)
    if total_len <= target_len:
        # Pad with last frame if needed
        return graphs + [graphs[-1]] * (target_len - total_len)
    
    start = random.randint(0, total_len - target_len)
    return graphs[start:start + target_len]


def add_feature_noise(
    graph: Data,
    noise_std: float = 0.01,
) -> Data:
    """
    Adds Gaussian noise to the node features of a graph.
    """
    noisy_x = graph.x + noise_std * torch.randn_like(graph.x)
    return Data(x=noisy_x, edge_index=graph.edge_index)
