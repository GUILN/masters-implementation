
import os
from pathlib import Path
import shutil

from settings.global_settings import GlobalSettings

TRAIN_DIR = "train"
TEST_DIR = "test"


logger = GlobalSettings.get_logger()


class NWUCLADatasetScripts:
    @staticmethod
    def split_train_and_test(
        videos_path: str,
        train_ratio: float = 0.8
    ):
        path = Path(videos_path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        subdirs_actions = [d for d in path.iterdir() if d.is_dir()]
        subdirs_actions.sort(key=lambda d: d.name)
        logger.info(f"Splitting dataset in {train_ratio*100}% train and {(1-train_ratio)*100}% test")
        train_dir = os.path.join(videos_path, TRAIN_DIR)
        test_dir = os.path.join(videos_path, TEST_DIR)
        logger.info(f"Total actions found: {len(subdirs_actions)}")
        for action_dir in subdirs_actions:
            action_train_dir = os.path.join(train_dir, action_dir.name)
            action_test_dir = os.path.join(test_dir, action_dir.name)
            os.makedirs(action_train_dir, exist_ok=True)
            os.makedirs(action_test_dir, exist_ok=True)
            logger.info(
                f"[NWUCLADatasetScripts] Splitting action videos for action: {action_dir.name}"
            )
            video_files = [
                f for f in action_dir.iterdir()
                if f.is_file() and f.suffix in [".json"] and not f.name.startswith(".")
            ]
            video_files.sort(key=lambda f: f.name)
            split_index = int(len(video_files) * train_ratio)
            train_files = video_files[:split_index]
            test_files = video_files[split_index:]

            for file in train_files:
                shutil.copy(file, os.path.join(action_train_dir, file.name))
            for file in test_files:
                shutil.copy(file, os.path.join(action_test_dir, file.name))
        logger.info("Dataset splitting completed.")
