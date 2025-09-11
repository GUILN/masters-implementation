
from typing import List

from src.models.frame_object import FrameObject
from src.models.skeleton import Skeleton


class VideoFrame:
    def __init__(
        self,
        frame_id: str,
        frame_sequence: int,
        time_stamp: float,
    ):
        self.frame_id = frame_id
        self.frame_sequence = frame_sequence
        self.time_stamp = time_stamp
        self._frame_objects = []
        self._frame_skeletons = []

    def to_dict(self):
        return {
            "frame_id": self.frame_id,
            "frame_sequence": self.frame_sequence,
            "time_stamp": self.time_stamp,
            "frame_objects": [obj.to_dict() for obj in self._frame_objects],
            "frame_skeletons": [skeleton.to_dict() for skeleton in self._frame_skeletons]
        }

    @classmethod
    def from_dict(cls, data: dict):
        frame = cls(
            frame_id=data["frame_id"],
            frame_sequence=data["frame_sequence"],
            time_stamp=data["time_stamp"]
        )
        for obj_data in data.get("frame_objects", []):
            frame.add_frame_object(FrameObject.from_dict(obj_data))
        for skeleton_data in data.get("frame_skeletons", []):
            frame.add_frame_skeleton(Skeleton.from_dict(skeleton_data))
        return frame

    @property
    def frame_objects(self) -> List[FrameObject]:
        return self._frame_objects

    @property
    def frame_skeletons(self) -> List[Skeleton]:
        return self._frame_skeletons

    def add_frame_object(self, frame_object: FrameObject):
        self._frame_objects.append(frame_object)

    def add_frame_skeleton(self, frame_skeleton: Skeleton):
        self._frame_skeletons.append(frame_skeleton)
