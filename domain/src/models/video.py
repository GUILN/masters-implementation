

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

    def to_dict(self):
        return {
            "video_id": self.video_id,
            "category": self.category,
            "frames": [frame.to_dict() for frame in self.frames]
        }

    @property
    def frames(self) -> List[VideoFrame]:
        self._frames = sorted(self._frames, key=lambda f: f.frame_sequence)
        return self._frames
