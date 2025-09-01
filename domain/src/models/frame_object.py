
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
