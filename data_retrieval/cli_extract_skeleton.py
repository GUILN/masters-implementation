#!/usr/bin/env python3
"""
Command-line interface for skeleton extraction from videos.
"""

import argparse
import json
import sys
from pathlib import Path
from common_setup import CommonSetup
from config_manager import ExtractionConfig
from app_logging.application_logger import ApplicationLogger

from dataset_path_manager.dataset_path_manager_factory import DatasetPathManagerFactory, DatasetType
from src.models.frame_object import FrameObject
from src.models.video import Video

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract skeleton data from videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    return parser

def override_config(config: ExtractionConfig, args: argparse.Namespace) -> None:
    if args.input_dir:
        config.config.set('paths', 'input_dir', args.input_dir)
    if args.output_dir:
        config.config.set('paths', 'output_dir', args.output_dir)
    if args.batch_size:
        config.config.set('extraction', 'batch_size', str(args.batch_size))
    if args.confidence_threshold:
        config.config.set('skeleton', 'confidence_threshold', str(args.confidence_threshold))
    if args.verbose:
        config.config.set('logging', 'level', 'DEBUG')

def setup_logging(config: ExtractionConfig) -> ApplicationLogger:
    return CommonSetup.get_logger()

def extract_skeleton_data(config: ExtractionConfig, logger: ApplicationLogger):
    object_settings = config.object_extraction_settings
    dataset_type = DatasetType.NW_UCLA
    logger.info("Extracting skeleton data...")

    logger.info(f"Using model: {object_settings['model_path']}")
    logger.info(f"Confidence threshold: {object_settings['confidence_threshold']}")
    logger.info(f"Input directory: {object_settings['input_dir']}")
    logger.info(f"Output directory: {object_settings['output_dir']}")
    logger.info(f"Dataset type: {dataset_type}")

    # read json file as dict
    input_file = "data/output/nw_ucla/extracted_features/a01/v01_s01_e00_frames_objects.json"
    with open(input_file, 'r', encoding='utf-8') as f:
        logger.info(f"Reading skeleton data from {input_file}...")
        data = json.load(f)
        logger.info(f"Data loaded: {type(data)}")
        logger.info(f"Data content keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        video_object = Video.from_dict(data)
        logger.info(f"video id: {video_object.video_id}")
        logger.info(f"video category: {video_object.category}")
        logger.info(f"video frames: {len(video_object.frames)}")
        for f in video_object.frames:
            logger.info(f"frame id: {f.frame_id}, frame_sequence: {f.frame_sequence}")
            logger.info(f"number of objects in frame: {len(f.frame_objects)}")
            for obj in f.frame_objects:
                logger.info(f"  object: {obj.to_dict()}")
        logger.info(video_object)

def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    logger = None

    try:
        config = ExtractionConfig(args.config)
        logger = setup_logging(config)
        override_config(config, args)

        logger.info("Current Configuration:")
        logger.info(f"  Input directory: {config.frame_extraction_settings['input_dir']}")
        logger.info(f"  Output directory: {config.frame_extraction_settings['output_dir']}")
        logger.info(f"  Batch size: {config.frame_extraction_settings['batch_size']}")

        skeleton_settings = config.skeleton_extraction_settings
        logger.info(f"  Confidence threshold: {skeleton_settings['confidence_threshold']}")
        logger.info(f"  Max persons: {skeleton_settings['max_persons']}")

        extract_skeleton_data(config, logger)

        if args.dry_run:
            logger.info("Dry run mode - no actual processing will occur")
            return

        logger.info("Skeleton extraction process completed")

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