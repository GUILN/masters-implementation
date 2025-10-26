import sys
from pathlib import Path



# Add the parent directory to the Python path first
sys.path.append(str(Path(__file__).parent.parent))

# Now import the modules
try:
    from cli.cli_args import ModelArgs, Command
except ImportError:
    from cli_args import ModelArgs, Command

try:
    from settings.global_settings import GlobalSettings
except ImportError:
    from global_settings import GlobalSettings

try:
    from dataset.video_loader import VideoDataLoader
except ImportError:
    from video_loader import VideoDataLoader

try:
    from dataset.nwucla.nwucla_scripts import NWUCLADatasetScripts
except ImportError:
    from nwucla_scripts import NWUCLADatasetScripts

# Loading CLI arguments and configuration
print("Loading CLI arguments and configuration...")  # Debug statement
args = ModelArgs.parse_args()
model_config = GlobalSettings.get_config()
logger = GlobalSettings.get_logger()


def show_settings():
    logger.debug("Current Configuration Settings:")
    logger.debug("Logging settings:")
    logger.debug(model_config.logging_settings)
    logger.debug("Model settings:")
    logger.debug(model_config.model_settings)
    logger.debug("Sentry settings:")
    logger.debug(model_config.sentry_settings)


def run_sentry_test():
    """Execute sentry mode operations."""
    logger.debug("Sentry mode specific operations:")
    logger.debug("- Monitoring and logging enabled")
    logger.debug("- Error tracking active")
    raise Exception("This is a test exception for Sentry.")


def test_video_dataloader():
    logger.debug(f"Testing VideoDataLoader from {model_config.model_settings.video_data_dir}")
    video_data_loader = VideoDataLoader(str(model_config.model_settings.video_data_dir))
    videos_iter = video_data_loader.load_videos()
    logger.info(f"[model_cli] Loaded {len(videos_iter)} videos.")
    for video in videos_iter:
        logger.debug(video)


def split_nwucla_dataset():
    NWUCLADatasetScripts.split_dataset(
        str(model_config.model_settings.video_data_dir),
        split_ratio=0.5,
        first_dataset="test",
        second_dataset="validation",
    )


def main():
    """Main CLI entry point."""
    logger.debug(f"Executing command: {args.command} with args: {args}")
    if args.verbose:
        logger.debug("Verbose mode enabled")
        show_settings()
    if args.command == Command.TEST_SENTRY:
        run_sentry_test()
    elif args.command == Command.TEST_VIDEO_DATALOADER:
        test_video_dataloader()
    elif args.command == Command.SPLIT_NWUCLA_DATASET:
        split_nwucla_dataset()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise
