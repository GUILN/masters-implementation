import os
from typing import Iterator, List, Optional
import zipfile
import shutil
from common_setup import CommonSetup
from dataset_path_manager.dataset_path_manager import (
    DatasetPathManagerInterface, VideoFramesPath, VideoPath
)

logger = CommonSetup.get_logger()

class NTURGBDPathManager(DatasetPathManagerInterface):
    def __init__(
        self,
        base_path: str,
        base_output_path: str,
    ):
        self._base_path = base_path
        self._base_output_path = base_output_path
        self._format = ".avi"
        self._zip_format = ".zip"

    def get_videos_path(self) -> Iterator[VideoPath]:
        logger.info("Retrieving video paths from NTU RGB-D dataset structure.")
        logger.info(f"Scanning: {self._base_path}")
        logger.info(f"This extractor expects the base path to contain zip files.")
        zip_files = [entry for entry in os.scandir(self._base_path)
                     if entry.is_file() and entry.name.endswith(self._zip_format)]
        logger.info(f"Found {len(zip_files)} zip files to extract.")
        for entry in zip_files:
            logger.info(f"Processing zip file: {entry.path}...")
            folder_to_extract = entry.path.replace(self._zip_format, "")
            try:
                os.makedirs(folder_to_extract, exist_ok=True)
                with zipfile.ZipFile(entry.path, 'r') as zip_ref:
                    zip_ref.extractall(folder_to_extract)
                for processed in self.process_extracted_folder(folder_to_extract):
                    yield processed
                logger.info(f"Finished processing zip file: {entry.path}.")
            except Exception as e:
                logger.error(f"Error extracting {entry.path}: {e}")
                raise e
            finally:
                logger.info(f"Deleting extracted folder: {folder_to_extract}...")
                try:
                    # remove non-empty extracted folders
                    shutil.rmtree(folder_to_extract)
                except Exception as e:
                    logger.warning(f"Failed to delete extracted folder {folder_to_extract}: {e}")

    def process_extracted_folder(self, folder_path: str) -> Iterator[VideoPath]:
        subfolders = [folder_path]
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
    def get_frames_path(
        self,
        actions_filter: Optional[List[str]] = None
    ) -> Iterator[VideoFramesPath]:
        """
        Extracts a unique video ID from the video path.
        For Unsafe Net, this could be the relative path from base_path without extension.
        """
        logger.info(f"[NTURGBDPathManager] getting videos from {self._base_path} and saving to {self._base_output_path}")
        if actions_filter:
            logger.info(f"[NTURGBDPathManager] Applying actions filter: {actions_filter}")
        videos_dirs = [child.path for child in os.scandir(self._base_path)]
        while len(videos_dirs) > 0:
            current_dir = videos_dirs.pop(0)
            base_name = os.path.basename(current_dir)
            video_category = base_name.split('_')[0][-4:]
            if actions_filter and video_category not in actions_filter:
                logger.info(f"[NTURGBDPathManager] Skipping video {base_name} of category {video_category}")
                continue
            logger.info(f"Scanning video dir {current_dir}")
            files = os.scandir(current_dir)
            # filter and sort
            # get last 6 digits before extension and make it int
            logger.debug(f"Sorting frames for video {current_dir}")
            sort_key = lambda x: int(os.path.splitext(os.path.basename(x))[0][-6:])
            files = sorted(
                (f.path for f in files if f.is_file() and f.name.endswith(('.jpg', '.png'))),
                key=sort_key
            )
            logger.debug("sorted files")
            logger.info(f"Video {base_name} of category {video_category} has {len(files)} frames")
            yield VideoFramesPath(
                sub_set=None,
                video_id=base_name,
                frames_path=files,
                video_category=video_category,
            )

    
    def get_video_frames_path_from_action_dir(self, action_dir: str, video_set: str) -> List[VideoFramesPath]:
        logger.info(f"[UnsafeNetPathManager] Getting video frames paths from action directory: {action_dir}")
        videos_frames_paths: List[VideoFramesPath] = []
        for action in os.scandir(action_dir):
            if action.is_dir():
                logger.info(f"[UnsafeNetPathManager] getting videos for action {action.name}")
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
                            sub_set=video_set,
                        )
                        videos_frames_paths.append(video_frame_path)
        return videos_frames_paths