

from typing import List

from src.models.video_frame import VideoFrame


class Video:
    def __init__(
        self,
        video_id: str,
        category: str,
    ):
        self.video_id = video_id
        self.category = category
        self._frames = []

    def add_frame(self, frame: VideoFrame):
        self._frames.append(frame)

    @property
    def frames(self) -> List[VideoFrame]:
        return self._frames
