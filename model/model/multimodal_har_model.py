from typing import List
import torch
import torch.nn as nn
from torch_geometric.data import Data

from model.gat_branch import GATBranch


class MultiModalHARModel(nn.Module):
    def __init__(
        self,
        obj_in: int,
        joint_in: int,
        gat_hidden: int,
        gat_out: int,
        temporal_hidden: int,
        num_classes: int
    ):
        super().__init__()
        self.obj_gat = GATBranch(obj_in, gat_hidden, gat_out)
        self.joint_gat = GATBranch(
            joint_in,
            gat_hidden,
            gat_out,
            pooling="max"
        )

        self.temporal_model = nn.LSTM(
            input_size=gat_out * 2,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(temporal_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        graphs_objects: List[Data],
        graphs_joints: List[Data]
    ):
        # Each is a list of T Data objects
        frame_features = []
        for G_obj, G_joint in zip(graphs_objects, graphs_joints):
            v_obj = self.obj_gat(G_obj)
            v_joint = self.joint_gat(G_joint)
            frame_features.append(torch.cat([v_obj, v_joint], dim=-1))

        x = torch.stack(frame_features, dim=1)  # [batch=1, T, features]
        out, _ = self.temporal_model(x)
        out = out[:, -1, :]  # last hidden state
        return self.classifier(out)
