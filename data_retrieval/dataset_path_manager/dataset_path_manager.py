from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional


@dataclass
class VideoPath:
    video_path: str
    output_path: str
    extension: str
    
@dataclass
class VideoFramesPath:
    video_id: str
    frames_path: List[str]
    # this one is to indicate the set, like: test | validation | train
    sub_set: Optional[str] = None

class DatasetPathManagerInterface(ABC):
    """
    Interface for managing dataset paths.
    As each dataset has it own structure, the implementation on how
    to retrieve paths will vary.
    """
    @abstractmethod
    def get_videos_path(self) -> List[VideoPath]:
        pass
    
    @abstractmethod
    def get_frames_path(self) -> List[VideoFramesPath]:
        pass
