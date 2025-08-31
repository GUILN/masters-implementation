
from typing import List

from src.models.frame_object import FrameObject


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

    @property
    def frame_objects(self) -> List[FrameObject]:
        return self._frame_objects

    def add_frame_object(self, frame_object: FrameObject):
        self._frame_objects.append(frame_object)
