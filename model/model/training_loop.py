
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Optional, cast
import torch
from tqdm import tqdm
from dataset.video_dataset import VideoData, VideoDataset
from model.multimodal_har_model import MultiModalHARModel
from torch.utils.data import DataLoader
from settings.global_settings import GlobalSettings
import torch.nn as nn
from model.early_stopping import EarlyStopping, ESMode


logger = GlobalSettings.get_logger()

EvaluationFunction = Callable[[nn.Module, VideoDataset], float]


@dataclass
class EarlyStoppingParams:
    """Parameters for early stopping during training."""
    patience: int = 5
    min_delta: float = 1e-4
    mode: ESMode = 'max'  # 'max' for accuracy, 'min' for loss
    evaluation_function: Optional[EvaluationFunction] = None
    evaluation_dataset: Optional[VideoDataset] = None


def train(
    model: MultiModalHARModel,
    video_dataset: VideoDataset,
    epochs: int = 20,
    batch_size: int = 1,
    lr=1e-3,
    device: str = "cpu",
    weight_decay: Optional[float] = None,
    early_stopping: Optional[EarlyStoppingParams] = None,
):
    logger.info("Starting training loop...")
    loader = DataLoader(
        video_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: x
    )
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )
    if weight_decay is not None:
        logger.info(f"Using weight decay: {weight_decay}")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    es = None
    if early_stopping is not None:
        logger.info("Using early stopping")
        es = EarlyStopping(
            patience=early_stopping.patience,
            mode=early_stopping.mode,
            delta=early_stopping.min_delta,
        )
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()

            if batch_size > 1:
                raise NotImplementedError(
                    "Batch size > 1 not implemented yet."
                )
            sample = cast(VideoData, batch[0])
            graphs_objects = [g.to(device) for g in sample.graphs_objects]
            graphs_joints = [g.to(device) for g in sample.graphs_joints]
            label = sample.label.unsqueeze(0).to(device)

            out = model(graphs_objects, graphs_joints)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if es is not None:
            logger.info("Evaluating for early stopping...")
            metric = early_stopping.evaluation_function(
                model, early_stopping.evaluation_dataset
            )
            stop = es.step(
                model=model,
                metric=metric
            )
            if stop:
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}."
                )
                es.load_best(model)
                logger.info("Loaded best model state from early stopping.")
                return
        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}"
        )
