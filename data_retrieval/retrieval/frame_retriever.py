import cv2
import os

from common_setup import CommonSetup


logger = CommonSetup.get_logger()


def extract_video(
    video_path: str,
    output_dir: str,
    frame_rate: int,
):
    # open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / frame_rate))
    logger.info(f"Video FPS: {video_fps} | Configured Frame Rate: {frame_rate} | Calculated Frame interval: {frame_interval}")
    if frame_interval == 0:
        logger.error(f"Frame rate {frame_rate} is higher than video FPS {video_fps} for video {video_path}. Skipping.")
        cap.release()
        raise ValueError("Frame rate is higher than video FPS")
    
    frame_count = 0
    saved_count = 0

    video_prefix = os.path.splitext(os.path.basename(video_path))[0]
    logger.info(f"🎬 Extracting frames from {video_path} at {frame_rate} FPS into {output_dir}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"{video_prefix}_frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        
        frame_count += 1
        
    cap.release()
    logger.info(f"✅ Extracted {saved_count} frames from {video_path} into {output_dir}")
    
class FrameRetriever:
    def __init__(
        self,
        frame_rate: int=8,
    ):
        self._frame_rate = frame_rate

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
    ):
        """
        Extract frames from a video at a given FPS.

        Args:
            video_path (str): Path to input video file.
            output_dir (str): Directory to save extracted frames.
        """
        os.makedirs(output_dir, exist_ok=True)
        extract_video(
            video_path=video_path,
            output_dir=output_dir,
            frame_rate=self._frame_rate,
        )
        
