from datetime import datetime
import os
from typing import Any, List, Optional
import torch
import torch.nn as nn
from torch_geometric.data import Data
from settings.global_settings import GlobalSettings

from model.gat_branch import GATBranch

logger = GlobalSettings.get_logger()


class MultiModalHARModel(nn.Module):
    def __init__(
        self,
        obj_in: int,
        joint_in: int,
        gat_hidden: int,
        gat_out: int,
        temporal_hidden: int,
        num_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self._model_config = {
            "obj_in": obj_in,
            "joint_in": joint_in,
            "gat_hidden": gat_hidden,
            "gat_out": gat_out,
            "temporal_hidden": temporal_hidden,
            "num_classes": num_classes,
            "dropout": dropout,
        }
        self.obj_gat = GATBranch(
            obj_in,
            gat_hidden,
            gat_out,
            dropout=dropout,
        )
        self.joint_gat = GATBranch(
            joint_in,
            gat_hidden,
            gat_out,
            pooling="max",
            dropout=dropout,
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
            nn.Dropout(dropout),
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

    def save(self, training_history: Optional[Any]) -> None:
        model_settings = GlobalSettings.get_config().model_settings
        os.makedirs(model_settings.model_save_dir, exist_ok=True)

        save_path = os.path.join(
            model_settings.model_save_dir,
            f"har_model_{model_settings.model_version}_{model_settings.dataset_prefix}_{datetime.now()}.pht"
        )
        logger.info(f"Saving model to {save_path}...")
        torch.save({
            "model_state_dict": self.state_dict(),
            "training_history": training_history,
            "model_config": self._model_config,
        }, save_path)
        logger.info("Model saved successfully.")
