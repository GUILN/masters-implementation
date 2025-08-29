#!/usr/bin/env python3
"""
Command-line interface for video extraction with configuration override.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, List
from config_manager import ExtractionConfig
from app_logging.application_logger import ApplicationLogger

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
    log_settings = config.logging_settings
    sentry_settings = config.sentry_settings
    
    # Create logger using app_logging with Sentry DSN from secrets
    logger = ApplicationLogger(
        name="data_extraction",
        loglevel=log_settings['level'],
        logfile=log_settings['log_file'],
        log_to_stdout=True,
        sentry_dsn=sentry_settings['dsn']
    )
    logger.info(f"Sentry DSN set to: {sentry_settings['dsn']}")

    if sentry_settings['dsn']:
        logger.info(f"Sentry initialized with environment: {sentry_settings['environment']}")
    else:
        logger.warning("Sentry DSN not configured - errors will not be sent to Sentry")
    
    return logger

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

def extract_objects_data(video_path: Path, config: ExtractionConfig, logger) -> str:
    """Extract object detection data from a video file."""
    object_settings = config.object_extraction_settings
    
    logger.info(f"Processing video for object detection: {video_path}")
    logger.info(f"Using model: {object_settings['model_path']}")
    logger.info(f"Confidence threshold: {object_settings['confidence_threshold']}")
    
    # Placeholder for actual object detection logic
    # This would integrate with your object detection model
    output_file = video_path.stem + ".obj"
    logger.info(f"Would create objects output file: {output_file}")
    
    return output_file

def extract_frames_data(video_path: Path, config: ExtractionConfig, logger) -> str:
    """Extract frames from a video file."""
    frame_settings = config.frame_extraction_settings
    
    logger.info(f"Processing video for frame extraction: {video_path}")
    logger.info(f"Frame rate per second: {frame_settings['frame_rate_per_second']}")
    logger.info(f"Resolution: {frame_settings['resolution']}")
    
    # Placeholder for actual frame extraction logic
    # This would extract frames at specified intervals
    output_dir = video_path.stem + "_frames"
    logger.info(f"Would create frames output directory: {output_dir}")
    
    return output_dir

def process_videos(config: ExtractionConfig, extraction_type: str, logger) -> None:
    """Process all videos in the input directory."""
    paths = config.paths
    frame_settings = config.frame_extraction_settings
    
    # Ensure output directory exists
    paths['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    input_dir = paths['input_dir']
    video_extensions = ['.avi', '.mp4', '.mov']
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    video_files: List[Path] = []
    for ext in video_extensions:
        video_files.extend(input_dir.rglob(f"*{ext}"))
    
    logger.info(f"Found {len(video_files)} video files for {extraction_type} extraction")
    
    # Choose extraction function based on type
    if extraction_type == 'skeleton':
        extract_func = extract_skeleton_data
    elif extraction_type == 'objects':
        extract_func = extract_objects_data
    else:  # frame
        extract_func = extract_frames_data
    
    # Process videos in batches
    batch_size = frame_settings['batch_size']
    for i in range(0, len(video_files), batch_size):
        batch = video_files[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} videos")
        
        for video_path in batch:
            try:
                output_file = extract_func(video_path, config, logger)
                logger.info(f"Successfully processed: {video_path.name}")
            except Exception as e:
                logger.error(f"Error processing {video_path.name}: {e}")

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
        logger.info(f"  Input directory: {config.paths['input_dir']}")
        logger.info(f"  Output directory: {config.paths['output_dir']}")
        logger.info(f"  Batch size: {config.frame_extraction_settings['batch_size']}")
        
        if args.extraction_type == 'skeleton':
            skeleton_settings = config.skeleton_extraction_settings
            logger.info(f"  Confidence threshold: {skeleton_settings['confidence_threshold']}")
            logger.info(f"  Max persons: {skeleton_settings['max_persons']}")
        elif args.extraction_type == 'objects':
            object_settings = config.object_extraction_settings
            logger.info(f"  Confidence threshold: {object_settings['confidence_threshold']}")
            logger.info(f"  Max objects: {object_settings['max_objects']}")
        else:  # frame
            frame_settings = config.frame_extraction_settings
            logger.info(f"  Frame rate per second: {frame_settings['frame_rate_per_second']}")
            logger.info(f"  Resolution: {frame_settings['resolution']}")
        
        if args.dry_run:
            logger.info("Dry run mode - no actual processing will occur")
            return
        
        # Run the extraction process
        logger.info("Starting video extraction process")
        logger.info(f"Configuration loaded from: {config.config_file}")
        
        process_videos(config, args.extraction_type, logger)
        
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
