
import os
from pathlib import Path
import shutil
from typing import Literal

from settings.global_settings import GlobalSettings


logger = GlobalSettings.get_logger()


DatasetType = Literal["train", "test", "validation"]


class NTURGBDDatasetScripts:
    @staticmethod
    def split_dataset(
        videos_path: str,
        split_ratio: float = 0.8,
        first_dataset: DatasetType = "train",
        second_dataset: DatasetType = "test",
        shuffle: bool = False,
    ):
        path = Path(videos_path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        subdirs_actions = [d for d in path.iterdir() if d.is_dir()]
        subdirs_actions.sort(key=lambda d: d.name)
        logger.info(f"Splitting dataset in {split_ratio*100}% first and {(1-split_ratio)*100}% second")
        first_dir = os.path.join(videos_path, first_dataset)
        second_dir = os.path.join(videos_path, second_dataset)
        logger.info(f"Total actions found: {len(subdirs_actions)}")
        for action_dir in subdirs_actions:
            action_first_dir = os.path.join(first_dir, action_dir.name)
            action_second_dir = os.path.join(second_dir, action_dir.name)
            os.makedirs(action_first_dir, exist_ok=True)
            os.makedirs(action_second_dir, exist_ok=True)
            logger.info(
                f"[NTURGBDDatasetScripts] Splitting action videos for action: {action_dir.name}"
            )
            video_files = [
                f for f in action_dir.iterdir()
                if f.is_file() and f.suffix in [".json"] and not f.name.startswith(".")
            ]
            video_files.sort(key=lambda f: f.name)
            if shuffle:
                import random
                random.shuffle(video_files)
            split_index = int(len(video_files) * split_ratio)
            first_files = video_files[:split_index]
            second_files = video_files[split_index:]

            for file in first_files:
                shutil.copy(file, os.path.join(action_first_dir, file.name))
            for file in second_files:
                shutil.copy(file, os.path.join(action_second_dir, file.name))
        logger.info("Dataset splitting completed.")
