

from numpy import ndarray
from sklearn.metrics import confusion_matrix
import torch
from dataset.video_dataset import VideoDataset
from model.multimodal_har_model import MultiModalHARModel


def calculate_confusion_matrix(
    har_model: MultiModalHARModel,
    video_dataset: VideoDataset,
    device: str = "cpu",
) -> ndarray:
    def pred_distribution():
        har_model.eval()
        preds = []
        truths = []
        with torch.no_grad():
            for s in video_dataset:
                graphs_objects = [g.to(device) for g in s.graphs_objects]
                graphs_joints = [g.to(device) for g in s.graphs_joints]
                out = har_model(graphs_objects, graphs_joints)  # [1, C]
                preds.append(int(torch.argmax(out, dim=-1)))
                truths.append(int(s.label))
        return preds, truths
    preds, truths = pred_distribution()
    return confusion_matrix(truths, preds)