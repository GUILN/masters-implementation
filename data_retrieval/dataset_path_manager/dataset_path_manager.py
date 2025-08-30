from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List


@dataclass
class VideoPath:
    video_path: str
    output_path: str
    extension: str

class DatasetPathManagerInterface(ABC):
    """
    Interface for managing dataset paths.
    As each dataset has it own structure, the implementation on how
    to retrieve paths will vary.
    """
    @abstractmethod
    def get_videos_path(self) -> List[VideoPath]:
        pass
