from typing import NamedTuple
import numpy as np
from tqdm import tqdm

from dataset.video_dataset import VideoDataset
from model.multimodal_har_model import MultiModalHARModel
from validation.result_types import EmbeddingsResult



def extract_embeddings(
    model: MultiModalHARModel,
    dataset: VideoDataset,
    device="cpu"
) -> EmbeddingsResult:
    model.eval()

    embeddings = []
    labels = []

    for sample in tqdm(dataset, desc="Extracting embeddings"):
        graphs_objects = sample.graphs_objects
        graphs_joints = sample.graphs_joints
        label = sample.label

        # Move to device
        graphs_objects = [g.to(device) for g in graphs_objects]
        graphs_joints = [g.to(device) for g in graphs_joints]

        emb = model.forward_features(graphs_objects, graphs_joints)
        embeddings.append(emb.cpu().detach().numpy())
        labels.append(label)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    return embeddings, labels
