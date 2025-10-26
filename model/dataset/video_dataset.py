from torch.utils.data import Dataset
from torch_geometric.data import Data

from dataset.video_loader import VideoDataLoader
from typing import Callable, Dict, List, Literal, Optional
from dataclasses import dataclass
from src.models.video_frame import VideoFrame
import torch
import torch.nn.functional as F
import torch_geometric.utils as utils
import itertools

# Type alias
Transform = Callable[[Data], Data]

NormalizationType = Literal[
    "per_sample",
    "dataset_wide",
    "no_normalization",
    "across_frames",
]


@dataclass(frozen=True)
class VideoData:
    graphs_objects: List[Data]
    graphs_joints: List[Data]
    label: torch.Tensor


@dataclass(frozen=True)
class NormalizationStats:
    mean: torch.Tensor
    std: torch.Tensor


def build_object_graph(frame: VideoFrame) -> Data:
    X: List[List[float]] = []
    for obj in frame.frame_objects:
        X_obj = [int(obj.object_class)]
        # TODO: maybe normalize bbox coordinates by frame size
        X_obj.extend(obj.bbox)
        X.append(X_obj)
    obj_edges = torch.tensor(
        [
            [i, j] for i, j in itertools.combinations(
                range(len(frame.frame_objects)), 2
            )
        ],
        dtype=torch.long
    ).t().contiguous()
    obj_edges = utils.to_undirected(obj_edges)
    return Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=obj_edges,
    )


def build_skeleton_graph(frame: VideoFrame) -> Data:
    X: List[List[float]] = []
    # Assuming only one skeleton per frame for simplicity
    for joint in frame.frame_skeletons[0].joints:
        # TODO: maybe normalize x, y coordinates by frame size
        X_joint = [joint.joint_id, joint.x, joint.y]
        X.append(X_joint)
    obj_edges = torch.tensor(
        [
            [i, j] for i, j in itertools.combinations(
                range(len(frame.frame_skeletons[0].joints)), 2
            )
        ],
        dtype=torch.long
    ).t().contiguous()
    obj_edges = utils.to_undirected(obj_edges)
    return Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=obj_edges,
    )


class VideoDataset(Dataset):
    def __init__(
        self,
        video_data_loader: VideoDataLoader,
        normalization_type: NormalizationType = "no_normalization",
        transform: Optional[Transform] = None,
        T: Optional[int] = None,
        normalization_stats: Optional[NormalizationStats] = None,
    ):
        self._video_data_loader = video_data_loader
        self._transform = transform
        self._T = T
        self._user_all_frames = False if T is not None else True
        self._item_cache: Dict[int, VideoData] = {}
        self._labels_map: Dict[str, int] = {}
        self._labels_counter = 0
        if normalization_type == "dataset_wide" and not normalization_stats:
            raise ValueError(
                "Normalization stats must be provided "
                "for dataset_wide normalization"
            )
        self._normalization_stats = normalization_stats
        self._normalization_type = normalization_type

    @property
    def labels_map(self) -> Dict[str, int]:
        return self._labels_map

    def get_label_name_from_label_value(
        self,
        label_value: torch.Tensor
    ) -> Optional[str]:
        int_value = int(label_value.item())
        for name, value in self._labels_map.items():
            if value == int_value:
                return name
        return None

    def __len__(self):
        return len(self._video_data_loader.load_videos())

    def __getitem__(self, idx: int) -> VideoData:
        if idx in self._item_cache:
            return self._item_cache[idx]
        video = self._video_data_loader.load_videos()[idx]
        frames = video.frames
        graphs_objects: List[Data] = []
        graphs_joints: List[Data] = []
        if not self._user_all_frames:
            frames = frames[:self._T]

        for frame in frames:
            graphs_objects.append(build_object_graph(frame))
            graphs_joints.append(build_skeleton_graph(frame))
        if self._normalization_type == "across_frames":
            graphs_objects = self._normalize_graphs(graphs_objects)
            graphs_joints = self._normalize_graphs(graphs_joints)

        if video.category not in self._labels_map:
            self._labels_map[video.category] = self._labels_counter
            self._labels_counter += 1
        label = torch.tensor(
            self._labels_map[video.category], dtype=torch.long
        )

        self._item_cache[idx] = VideoData(
            graphs_objects=graphs_objects,
            graphs_joints=graphs_joints,
            label=label,
        )
        return self._item_cache[idx]

    def _normalize_graphs(
        self,
        graphs: List[Data]
    ) -> List[Data]:
        """Normalize features of a list of graphs across frames."""
        # concatenate all node features
        all_x = torch.cat([g.x for g in graphs], dim=0)
        mean = all_x.mean(dim=0, keepdim=True)
        std = all_x.std(dim=0, keepdim=True) + 1e-8
        for g in graphs:
            g.x = (g.x - mean) / std
        return graphs

    def _normalize_features(
        self,
        x: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Normalize feature tensor according to mode."""
        if x is None:
            return None

        if self._normalization_type == "no_normalization":
            return x
        elif self._normalization_type == "per_sample":
            # L2 normalization across feature dimension
            return F.normalize(x, p=2, dim=-1)
        elif self._normalization_type == "dataset_wide":
            mean = self._normalization_stats.mean
            std = self._normalization_stats.std
            return (x - mean) / (std + 1e-6)

        return x
