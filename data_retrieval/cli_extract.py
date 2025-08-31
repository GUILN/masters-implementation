#!/usr/bin/env python3
"""
Command-line interface for video extraction with configuration override.
"""

import argparse
import os
import sys
from pathlib import Path
import time
from common_setup import CommonSetup
from config_manager import ExtractionConfig
from app_logging.application_logger import ApplicationLogger

from dataset_path_manager.dataset_path_manager_factory import DatasetPathManagerFactory, DatasetType
from retrieval.frame_retriever import FrameRetriever
import multiprocessing as mp

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract skeleton, object, or frame data from videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'extraction_type',
        nargs='?',
        choices=['skeleton', 'objects', 'frame'],
        help='Type of extraction to perform: skeleton for human pose estimation, objects for scene object detection, or frame for video frame extraction'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.ini',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        help='Input directory containing videos (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for skeleton files (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Batch size for processing (overrides config)'
    )
    
    parser.add_argument(
        '--confidence-threshold', '-t',
        type=float,
        help='Confidence threshold for skeleton detection (overrides config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    
    parser.add_argument(
        '--test-sentry',
        action='store_true',
        help='Test Sentry integration by triggering a test exception'
    )
    
    return parser

def override_config(config: ExtractionConfig, args: argparse.Namespace) -> None:
    """Override configuration with command line arguments."""
    # Override paths
    if args.input_dir:
        config.config.set('paths', 'input_dir', args.input_dir)
    
    if args.output_dir:
        config.config.set('paths', 'output_dir', args.output_dir)
    
    # Override extraction settings
    if args.batch_size:
        config.config.set('extraction', 'batch_size', str(args.batch_size))
    
    # Override skeleton settings
    if args.confidence_threshold:
        config.config.set('skeleton', 'confidence_threshold', str(args.confidence_threshold))
    
    # Override logging
    if args.verbose:
        config.config.set('logging', 'level', 'DEBUG')

def setup_logging(config: ExtractionConfig) -> ApplicationLogger:
    """Setup logging based on configuration."""
    return CommonSetup.get_logger()

def test_sentry(config: ExtractionConfig) -> None:
    """Test Sentry integration by triggering an exception."""
    logger = setup_logging(config)
    sentry_settings = config.sentry_settings
    
    logger.info("🧪 Testing Sentry integration...")
    
    if not sentry_settings['dsn']:
        logger.warning("❌ Sentry DSN not configured. Please add your DSN to secrets.ini")
        logger.info("💡 Run 'make setup-secrets' and edit secrets.ini with your Sentry DSN")
        return
    
    try:
        logger.info("🚀 Triggering test exception for Sentry...")
        # Intentionally trigger an exception
        result = 1 / 0  # This will cause a ZeroDivisionError
    except ZeroDivisionError as e:
        logger.error("💥 Test exception caught and sent to Sentry!")
        logger.info("✅ If Sentry is configured correctly, you should see this error in your Sentry dashboard")
        logger.exception(e)
    except Exception as e:
        logger.exception(f"❌ Unexpected error during Sentry test: {e}")

def extract_skeleton_data(video_path: Path, config: ExtractionConfig, logger) -> str:
    """Extract skeleton data from a video file."""
    skeleton_settings = config.skeleton_extraction_settings
    
    logger.info(f"Processing video for skeleton extraction: {video_path}")
    logger.info(f"Using model: {skeleton_settings['model_path']}")
    logger.info(f"Confidence threshold: {skeleton_settings['confidence_threshold']}")
    
    # Placeholder for actual skeleton extraction logic
    # This would integrate with your pose estimation model
    output_file = video_path.stem + ".ske"
    logger.info(f"Would create skeleton output file: {output_file}")
    
    return output_file

def extract_objects_data(config: ExtractionConfig, logger) -> str:
    """Extract object detection data from a video file."""
    object_settings = config.object_extraction_settings
    dataset_type = DatasetType.NW_UCLA
    
    logger.info(f"Using model: {object_settings['model_path']}")
    logger.info(f"Confidence threshold: {object_settings['confidence_threshold']}")
    logger.info(f"Input directory: {object_settings['input_dir']}")
    logger.info(f"Output directory: {object_settings['output_dir']}")
    logger.info(f"Dataset type: {dataset_type}")

    dataset_path_manager = DatasetPathManagerFactory.create_path_manager(
        dataset_type=dataset_type,
        base_path=str(config.object_extraction_settings['input_dir']),
        base_output_path=str(config.object_extraction_settings['output_dir']),
    )
    
    
    logger.info(f"Dataset path manager created: {type(dataset_path_manager).__name__}")

    logger.info("Getting video IDs...")
    logger.info("Starting object detection...")
    logger.info("Getting video frames paths...")
    video_frames_paths = dataset_path_manager.get_frames_path()
    logger.info(f"Got {len(video_frames_paths)} videos")
    # get first three frames paths for the first video to see if it is coming in order
    logger.info(f"{video_frames_paths[0].video_id} - {video_frames_paths[0].frames_path[:3]}")
    for video_frames_path in video_frames_paths:
        logger.debug(f" - {video_frames_path.video_id}")
        for frame_path in video_frames_path.frames_path:
            logger.debug(f"   - {frame_path}")


def extract_frames_data(config: ExtractionConfig, logger) -> str:
    """Extract frames from a video file."""
    frame_settings = config.frame_extraction_settings
    dataset_type = DatasetType.NW_UCLA
    
    logger.info("starting frame extraction")
    logger.info(f"Frame rate per second: {frame_settings['frame_rate_per_second']}")
    logger.info(f"Getting video paths from {config.frame_extraction_settings['input_dir']}...")
    logger.info(f"Dataset type: {dataset_type}")

    dataset_path_manager = DatasetPathManagerFactory.create_path_manager(
        dataset_type=dataset_type,
        base_path=str(config.frame_extraction_settings['input_dir']),
        base_output_path=str(config.frame_extraction_settings['output_dir']),
    )
    
    logger.info(f"Dataset path manager created: {type(dataset_path_manager).__name__}") 
    
    logger.info("getting videos path...")    
    videos_paths = dataset_path_manager.get_videos_path()
    logger.info(f"Found {len(videos_paths)} videos to process.")
    for video_path in videos_paths:
        logger.debug(f"Found video path: {video_path}")

    logger.info("Starting frame extraction")
    frame_retriever = FrameRetriever(
        frame_rate=frame_settings['frame_rate_per_second'],
    )
    logger.info("Running in the first video for testing")
    try:
        # get time
        start_time = time.time()
        logger.info(f"Running {len(videos_paths)} tasks in parallel using {os.cpu_count()} workers.")
        with mp.Pool(processes=os.cpu_count()) as pool:
            results = pool.starmap(frame_retriever.extract_frames, [(video_path.video_path, video_path.output_path + "_frames") for video_path in videos_paths])
        logger.info(f"Parallel execution completed. Processed {len(results)} results.")
        # start timer
        elapsed_time = time.time() - start_time
        logger.info(f"Frame extraction completed in {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")


   
def main() -> None:
    """Main CLI function."""
    parser: argparse.ArgumentParser = create_parser()
    args: argparse.Namespace = parser.parse_args()
    logger = None
    
    try:
        # Load configuration
        config: ExtractionConfig = ExtractionConfig(args.config)
        
        # Handle special test-sentry mode
        if args.test_sentry:
            test_sentry(config)
            return
        
        # Validate that extraction_type is provided when not testing Sentry
        if not args.extraction_type:
            print("Error: extraction_type is required when not using --test-sentry", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        
        logger = setup_logging(config)

        # Override with command line arguments
        override_config(config, args)
        
        # Display current configuration
        logger.info("Current Configuration:")
        logger.info(f"  Extraction type: {args.extraction_type}")
        logger.info(f"  Input directory: {config.frame_extraction_settings['input_dir']}")
        logger.info(f"  Output directory: {config.frame_extraction_settings['output_dir']}")
        logger.info(f"  Batch size: {config.frame_extraction_settings['batch_size']}")
        
        if args.extraction_type == 'skeleton':
            skeleton_settings = config.skeleton_extraction_settings
            logger.info(f"  Confidence threshold: {skeleton_settings['confidence_threshold']}")
            logger.info(f"  Max persons: {skeleton_settings['max_persons']}")
        elif args.extraction_type == 'objects':
            object_settings = config.object_extraction_settings
            logger.info(f"  Confidence threshold: {object_settings['confidence_threshold']}")
            logger.info(f"  Max objects: {object_settings['max_objects']}")
            extract_objects_data(config, logger)
        else:  # frame
            extract_frames_data(config, logger)

        if args.dry_run:
            logger.info("Dry run mode - no actual processing will occur")
            return
        
        logger.info("Video extraction process completed")
        
    except FileNotFoundError as e:
        error_msg = f"Configuration file error: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
