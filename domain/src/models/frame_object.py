
from typing import List


class FrameObject:
    def __init__(
        self,
        object_class: str,
        bbox: List[float],
        confidence: float
    ):
        self.object_class = object_class
        self.bbox = bbox
        self.confidence = confidence

    def to_dict(self):
        return {
            "object_class": self.object_class,
            "bbox": self.bbox,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            object_class=data["object_class"],
            bbox=data["bbox"],
            confidence=data["confidence"]
        )
