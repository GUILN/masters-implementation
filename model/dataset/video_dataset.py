from torch.utils.data import Dataset
from torch_geometric.data import Data

from dataset.video_loader import VideoDataLoader
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass
from src.models.video_frame import VideoFrame
import torch
import torch_geometric.utils as utils
import itertools
from copy import deepcopy

# Type alias
Transform = Callable[[Data], Data]


@dataclass(frozen=True)
class VideoData:
    graphs_objects: List[Data]
    graphs_joints: List[Data]
    label: torch.Tensor


def build_object_graph(frame: VideoFrame) -> Data:
    X: List[List[float]] = []
    for obj in frame.frame_objects:
        X_obj = [int(obj.object_class)]
        # TODO: maybe normalize bbox coordinates by frame size
        X_obj.extend(obj.bbox)
        X.append(deepcopy(X_obj))
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
        X.append(deepcopy(X_joint))
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
        transform: Optional[Transform] = None,
        T: Optional[int] = None
    ):
        self._video_data_loader = video_data_loader
        self._transform = transform
        self._T = T
        self._user_all_frames = False if T is not None else True
        self._item_cache: Dict[int, VideoData] = {}
        self._labels_map: Dict[str, int] = {}
        self._labels_counter = 0

    @property
    def labels_map(self) -> Dict[str, int]:
        return self._labels_map

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
