

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

    @classmethod
    def from_dict(cls, data: dict):
        video = cls(
            video_id=data["video_id"],
            category=data["category"]
        )
        for frame_data in data.get("frames", []):
            video.add_frame(VideoFrame.from_dict(frame_data))
        return video

    @classmethod
    def from_json(cls, json_str: str):
        import json
        data = json.loads(json_str)
        return cls.from_dict(data)

    @property
    def frames(self) -> List[VideoFrame]:
        self._frames = sorted(self._frames, key=lambda f: f.frame_sequence)
        return self._frames
