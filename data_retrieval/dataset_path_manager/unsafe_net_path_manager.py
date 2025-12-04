import os
from typing import List
from common_setup import CommonSetup
from dataset_path_manager.dataset_path_manager import (
    DatasetPathManagerInterface, VideoFramesPath, VideoPath
)

logger = CommonSetup.get_logger()

class UnsafeNetPathManager(DatasetPathManagerInterface):
    def __init__(
        self,
        base_path: str,
        base_output_path: str,
    ):
        self._base_path = base_path
        self._base_output_path = base_output_path
        self._format = ".mp4"

    def get_videos_path(self) -> List[VideoPath]:
        logger.info("Retrieving video paths from Unsafe Net dataset structure.")
        logger.info(f"Scanning: {self._base_path}")
        subfolders = [self._base_path]
        while len(subfolders) > 0:
            current_folder = subfolders.pop(0)
            
            for entry in os.scandir(current_folder):
                logger.info(f"Scanning entry: {entry.path}")
                if entry.is_dir():
                    subfolders.append(entry.path)
                elif entry.is_file() and entry.name.endswith(self._format):
                    relative_path = os.path.relpath(entry.path, self._base_path)
                    output_path = os.path.join(self._base_output_path, os.path.splitext(relative_path)[0])
                    logger.debug(f"Found video file: {entry.path}")
                    yield VideoPath(
                        video_path=entry.path,
                        output_path=output_path,
                        extension=self._format
                    )
        # only get subfolders
    def get_frames_path(self) -> List[VideoFramesPath]:
        """
        Extracts a unique video ID from the video path.
        For Unsafe Net, this could be the relative path from base_path without extension.
        """
        videos_frames_paths: List[VideoFramesPath] = []
        for action in os.scandir(self._base_path):
            if action.is_dir():
                logger.info(f"getting videos for action {action.name}")
                for video in os.scandir(action.path):
                    if video.is_dir():
                        frames_paths: List[str] = []
                        logger.debug(f"Getting frames for video {video.name}")
                        files = os.scandir(video.path)
                        # filter and sort
                        # get last 6 digits before extension and make it int
                        logger.debug(f"Sorting frames for video {video.name}")
                        sort_key = lambda x: int(os.path.splitext(os.path.basename(x))[0][-6:])
                        files = sorted(
                            (f.path for f in files if f.is_file() and f.name.endswith(('.jpg', '.png'))),
                            key=sort_key
                        )
                        logger.debug("sorted files")
                        for file in files:
                            frames_paths.append(file)
                        logger.debug(f"Found {len(frames_paths)} frames for video {video.name}")

                        video_frame_path = VideoFramesPath(
                            video_id=video.name,
                            frames_path=frames_paths,
                        )
                        videos_frames_paths.append(video_frame_path)
        return videos_frames_paths
