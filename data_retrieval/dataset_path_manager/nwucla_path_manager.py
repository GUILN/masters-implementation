import os
from typing import List
from common_setup import CommonSetup
from dataset_path_manager.dataset_path_manager import (
    DatasetPathManagerInterface, VideoPath
)

logger = CommonSetup.get_logger()

class NwUclaPathManager(DatasetPathManagerInterface):
    def __init__(
        self,
        base_path: str,
        base_output_path: str,
    ):
        self._base_path = base_path
        self._base_output_path = base_output_path

    def get_videos_path(self) -> List[VideoPath]:
        logger.info("Retrieving video paths from NW-UCLA dataset structure.")
        main_folder = "multiview_action_videos"
        # only get subfolders
        subfolders = [
            f.path for f in os.scandir(os.path.join(self._base_path, main_folder))
            if f.is_dir()
        ]
        logger.debug(f"Found {len(subfolders)} subfolders in {main_folder}.")
        video_paths = []
        
        # for each subfolder get only avi files
        for subfolder in subfolders:
            avi_files = [
                f.path for f in os.scandir(subfolder)
                if f.is_file() and f.name.endswith('.avi')
            ]
            logger.debug(f"Found {len(avi_files)} .avi files in {subfolder}.")
            if len(avi_files) == 0:
                logger.debug(f"No .avi files found in {subfolder}, skipping.")
                continue
            for avi_file in avi_files:
                relative_path = os.path.relpath(avi_file, self._base_path)
                output_path = os.path.join(self._base_output_path, os.path.splitext(relative_path)[0])
                video_paths.append(VideoPath(
                    video_path=avi_file,
                    output_path=output_path,
                    extension='.avi'
                ))
        return video_paths
