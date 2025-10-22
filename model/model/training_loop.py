
from typing import cast
import torch
from tqdm import tqdm
from dataset.video_dataset import VideoData, VideoDataset
from model.multimodal_har_model import MultiModalHARModel
from torch.utils.data import DataLoader
from settings.global_settings import GlobalSettings


logger = GlobalSettings.get_logger()


def train(
    model: MultiModalHARModel,
    video_dataset: VideoDataset,
    epochs: int = 20,
    batch_size: int = 1,
    lr=1e-3,
    device: str = "cpu",
    weight_decay: float = 1e-4,
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
        weight_decay=weight_decay
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
        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}"
        )
