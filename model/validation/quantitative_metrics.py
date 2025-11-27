from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from typing import List
from tqdm import tqdm

from dataset.video_dataset import VideoDataset
from validation.result_types import QuantitativeMetrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: VideoDataset,
    device: torch.device
) -> QuantitativeMetrics:
    """
    Evaluates a trained model on a VideoDataset.
    Computes accuracy & macro/micro AUC.
    """
    model.eval()
    y_true_list: List[int] = []
    y_pred_list: List[torch.Tensor] = []

    for video_data in tqdm(dataset, desc="Evaluating"):
        graphs_objects = [g.to(device) for g in video_data.graphs_objects]
        graphs_joints = [g.to(device) for g in video_data.graphs_joints]
        label = video_data.label.to(device)

        # Forward pass
        logits: torch.Tensor = model(graphs_objects, graphs_joints)
        y_pred_list.append(logits.cpu())
        y_true_list.append(label.cpu().item())

    # Prepare tensors
    y_true = torch.tensor(y_true_list)               # (N,)
    y_pred_logits = torch.cat(y_pred_list, dim=0)     # (N, C)
    y_pred_probs = torch.softmax(y_pred_logits, dim=1)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred_probs)
    return metrics


def compute_metrics(
    y_true: torch.Tensor,
    y_pred_probs: torch.Tensor
) -> QuantitativeMetrics:
    """
    Compute accuracy and AUC metrics.
    """
    y_true_np = y_true.numpy()
    y_pred_np = y_pred_probs.numpy()

    # Accuracy
    preds = y_pred_np.argmax(axis=1)
    accuracy = (preds == y_true_np).mean()

    # AUC Metrics
    try:
        macro_auc = roc_auc_score(
            y_true_np, y_pred_np,
            multi_class='ovr',
            average='macro'
        )
        micro_auc = roc_auc_score(
            y_true_np, y_pred_np,
            multi_class='ovr',
            average='micro'
        )
    except ValueError as e:
        print(
            "[WARNING] AUC could not be computed. Possibly only one class present."
        )
        raise e

    return QuantitativeMetrics(
        accuracy=accuracy,
        macro_auc=macro_auc,
        micro_auc=micro_auc
    )
