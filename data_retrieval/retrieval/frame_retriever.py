import urllib
import cv2
import os
from cv2 import dnn_superres
from common_setup import CommonSetup

logger = CommonSetup.get_logger()



def get_superres_model_path() -> str:
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "EDSR_x2.pb")

    if not os.path.exists(model_path):
        logger.info("📦 Downloading EDSR_x2.pb super-resolution model...")
        url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb"
        urllib.request.urlretrieve(url, model_path)
        logger.info(f"✅ Model downloaded to {model_path}")

    return model_path


class FrameRetriever:
    def __init__(self, frame_rate: int = 8, upscale: bool = False, use_superres: bool = False):
        self._frame_rate = frame_rate
        self._upscale = upscale
        self._use_superres = use_superres

        if use_superres:
            logger.info("Using DNN Super Resolution for frame enhancement")
            model_path = get_superres_model_path()
            self.sr = dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(model_path)
            self.sr.setModel("edsr", 2)

    def extract_frames(self, video_path: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(round(video_fps / self._frame_rate)))

        frame_count = 0
        saved_count = 0
        video_prefix = os.path.splitext(os.path.basename(video_path))[0]

        logger.info(f"🎬 Extracting frames from {video_path} at {self._frame_rate} FPS into {output_dir}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                if self._use_superres:
                    logger.info("Applying super-resolution superres sampling")
                    frame = self.sr.upsample(frame)
                elif self._upscale:
                    logger.info("Applying upscaling")
                    frame = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

                frame_filename = os.path.join(output_dir, f"{video_prefix}_frame_{saved_count:06d}.jpg")
                logger.info(f"Saving frame {saved_count} to {frame_filename}")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        logger.info(f"✅ Extracted {saved_count} frames from {video_path} into {output_dir}")
