
from pathlib import Path
from typing import List, Generator
from src.models.video import Video

from settings.global_settings import GlobalSettings


logger = GlobalSettings.get_logger()


class VideoDataLoader:
    def __init__(
        self,
        path: str,
    ):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Path {self._path} does not exist.")

    def load_videos(self) -> List[Video]:
        return list(self.iter_videos())

    def iter_videos(self) -> Generator[Video, None, None]:
        # get all subdirectories - which indicates different actions
        subdirs_actions = [d for d in self._path.iterdir() if d.is_dir()]
        subdirs_actions.sort(key=lambda d: d.name)
        for action_dir in subdirs_actions:
            logger.info(
                f"[VideoDataLoader] Loding action videos for action: {action_dir.name}"
            )
            video_files = [
                f for f in action_dir.iterdir()
                if f.is_file() and f.suffix in [".json"] and not f.name.startswith(".")
            ]
            video_files.sort(key=lambda f: f.name)
            for video_file in video_files:
                logger.debug(f"Loading video file: {video_file}")
                video = Video.from_json(video_file.read_text())
                if video is None:
                    logger.warning(f"Failed to load video from file: {video_file}")
                    continue
                logger.debug(f"Loaded video: {video}")
                yield video
