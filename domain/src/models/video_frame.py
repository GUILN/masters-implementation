
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
        self._frame_objects: List[FrameObject] = []
        self._frame_skeletons: List[Skeleton] = []

    def to_dict(self):
        return {
            "frame_id": self.frame_id,
            "frame_sequence": self.frame_sequence,
            "time_stamp": self.time_stamp,
            "frame_objects": [obj.to_dict() for obj in self._frame_objects],
            "frame_skeletons": [skeleton.to_dict() for skeleton in self._frame_skeletons]
        }

    @property
    def frame_objects(self) -> List[FrameObject]:
        return self._frame_objects

    def add_frame_object(self, frame_object: FrameObject):
        self._frame_objects.append(frame_object)
