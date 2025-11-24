from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import torch
import torch.nn as nn
from torch_geometric.data import Data
from model.multi_head_temporal_pooling import MultiHeadTemporalPooling
from model.multi_temporal_graph_convolution import MultiTemporalGC
from model.temporal_attention_pooling import TemporalAttentionPooling
from model.temporal_encoder import TemporalEncoder
from model.temporal_transformer_block import TemporalTransformerBlock
from settings.global_settings import GlobalSettings

from model.gat_branch import GATBranch

logger = GlobalSettings.get_logger()

PoolingType = Literal["min", "max", "attn_pool"]


class MultiModalHARModel(nn.Module):
    def __init__(
        self,
        obj_in: int,
        joint_in: int,
        gat_hidden: int,
        gat_out: int,
        temporal_hidden: int,
        num_classes: int,
        dropout: float = 0.1,
        temporal_pooling: PoolingType = "max",
        use_layer_norm: bool = False,
        attention_pooling_heads: int = 4,
        temporal_transformer_heads: int = 4,
        use_object_branch: bool = True,
        device: str = "cpu",
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
            "temporal_pooling": temporal_pooling,
            "use_layer_norm": use_layer_norm,
            "attention_pooling_heads": attention_pooling_heads,
            "temporal_transformer_heads": temporal_transformer_heads,
            "use_object_branch": use_object_branch,
            "device": device,
        }
        logger.info(f"Model configuration: {self._model_config}")
        self._temporal_pooling = temporal_pooling
        self._use_layer_norm = use_layer_norm
        self._use_object_branch = use_object_branch
        self._device = device
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

        self.temporal_model = MultiTemporalGC(
            in_channels=gat_out * 2,
            out_channels=temporal_hidden,
            kernel_sizes=[3, 5, 7],
            dropout=dropout,
        )

        self.pre_norm = nn.LayerNorm(temporal_hidden)

        self.temporal_transformer = TemporalTransformerBlock(
            channels=temporal_hidden,
            num_heads=temporal_transformer_heads,
            dropout=dropout
        )

        self.temporal_encoder = TemporalEncoder(
            channels=temporal_hidden,
            hidden=temporal_hidden,
        )

        self.attn_pool = MultiHeadTemporalPooling(
            temporal_hidden,
            num_heads=attention_pooling_heads,
        )

        self.classifier = nn.Sequential(
            nn.Linear(temporal_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def _forward_x(
        self,
        graphs_objects: List[Data],
        graphs_joints: List[Data],
    ) -> torch.Tensor:
        # Each is a list of T Data objects
        frame_features = []
        for G_obj, G_joint in zip(graphs_objects, graphs_joints):
            v_joint = self.joint_gat(G_joint)
            if self._use_object_branch:
                v_obj = self.obj_gat(G_obj)
                frame_features.append(torch.cat([v_obj, v_joint], dim=-1))
            else:
                frame_features.append(v_joint)

        x = torch.stack(frame_features, dim=2)  # [batch=1, features, T]
        x = self.temporal_model(x)
        if self._use_layer_norm:
            x = x.permute(0, 2, 1)  # [B, T, C]
            x = self.pre_norm(x)
            x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.temporal_transformer(x)
        x = self.temporal_encoder(x)
        if self._temporal_pooling == "max":
            x, _ = torch.max(x, dim=-1)  # Global temporal pooling
        elif self._temporal_pooling == "min":
            x, _ = torch.min(x, dim=-1)  # Global temporal pooling
        elif self._temporal_pooling == "attn_pool":
            x = self.attn_pool(x)  # Temporal attention pooling
        else:
            raise ValueError(f"Unknown temporal pooling type: {self._temporal_pooling}")
        return x

    def forward(
        self,
        graphs_objects: List[Data],
        graphs_joints: List[Data]
    ):
        x = self._forward_x(graphs_objects, graphs_joints)
        return self.classifier(x)

    def save(self, training_history: Optional[Any]) -> None:
        model_settings = GlobalSettings.get_config().model_settings
        os.makedirs(model_settings.model_save_dir, exist_ok=True)

        save_path = os.path.join(
            model_settings.model_save_dir,
            f"har_model_{model_settings.model_version}_{model_settings.dataset_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pht"
        )
        logger.info(f"Saving model to {save_path}...")
        torch.save({
            "model_state_dict": self.state_dict(),
            "training_history": training_history,
            "model_config": self._model_config,
        }, save_path)
        logger.info("Model saved successfully.")

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        device: str = "cpu"
    ) -> Tuple["MultiModalHARModel", Dict[any, any]]:
        """
        Standalone function to load HAR model.

        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load on
        Returns:
            tuple: (model, checkpoint_data)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading model from {checkpoint_path}...")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract configuration
        model_config = checkpoint.get('model_config')
        if not model_config:
            raise ValueError("Model configuration not found in checkpoint")

        logger.info(f"Model config: {model_config}")

        # Create and load model
        model = MultiModalHARModel(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()  # Set to evaluation mode

        logger.info("✅ Model loaded and ready for inference")

        return model, checkpoint

    @torch.no_grad()
    def forward_features(
        self,
        graphs_objects: List[Data],
        graphs_joints: List[Data]
    ) -> torch.Tensor:
        """
        Forward pass until the temporal pooling layer (before classification).
        Returns a feature embedding for visualization (e.g., t-SNE).
        """
        self.eval()
        return self._forward_x(graphs_objects, graphs_joints)
