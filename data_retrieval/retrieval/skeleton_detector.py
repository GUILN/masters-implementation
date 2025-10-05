from common_setup import CommonSetup
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples
from mmdet.apis import inference_detector, init_detector
import mmcv
import os
from src.models.skeleton import Skeleton
from src.models.video_frame import VideoFrame
from mmpose.utils import register_all_modules as register_pose_modules
from mmdet.utils import register_all_modules as register_det_modules
from mmengine.registry import DefaultScope
from mmengine.structures import InstanceData
import numpy as np
import cv2


# COCO skeleton connections
COCO_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
]

# --- Configs ---

register_det_modules(init_default_scope=False)   # registers mmdet stuff
register_pose_modules(init_default_scope=False)  # registers mmpose stuff
# Important: must come before building datasets/models


logger = CommonSetup.get_logger()

class ObjectDetector:
    def __init__(
        self,
        det_config: str,
        det_checkpoint: str,
    ):
        with DefaultScope.overwrite_default_scope('mmdet'):
            self.det_model = init_detector(det_config, det_checkpoint, device='cpu')
        
        logger.info(f"Detection config: {det_config}")

    def detect_objects(
        self,
        image_path: str,
    ) -> InstanceData:
        with DefaultScope.overwrite_default_scope('mmdet'):
            logger.info("Starting object detection...")

            logger.info(f"Image path: {image_path}")
            # --- Init model ---
            logger.info("Model initialized.")
            img_path = image_path

            # 1. Detect humans
            det_result = inference_detector(self.det_model, img_path)

            # det_result is a DetDataSample
            instances = det_result.pred_instances  # type: InstanceData

            return instances
        
class PoseDetector:
    def __init__(
        self,
        pose_config: str,
        pose_checkpoint: str,
    ):
        with DefaultScope.overwrite_default_scope('mmpose'):
            self.pose_model = init_model(pose_config, pose_checkpoint, device='cpu')
        
        logger.info(f"Pose config: {pose_config}")
    
    def detect_poses(
        self,
        image_path: str,
        person_bboxes: list,
    ) -> InstanceData:
        
        with DefaultScope.overwrite_default_scope('mmpose'):
            logger.info("Starting pose estimation...")

            logger.info(f"Image path: {image_path}")
            # --- Init model ---
            logger.info("Model initialized.")
            img_path = image_path

            # 2. Extract skeleton joints
            pose_results = inference_topdown(self.pose_model, img_path, person_bboxes)
            pose_data = merge_data_samples(pose_results)

            return pose_data.pred_instances  # type: InstanceData
        
class DetectionPipeline:
    def __init__(
        self,
        det_config = '/home/guilherme/Mestrado/masters-implementation/data_retrieval/retrieval/config/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
        det_checkpoint = '/home/guilherme/Mestrado/masters-implementation/data_retrieval/retrieval/config/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        pose_config = '/home/guilherme/Mestrado/masters-implementation/data_retrieval/retrieval/mmpose/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
        pose_checkpoint = '/home/guilherme/Mestrado/masters-implementation/data_retrieval/retrieval/config/checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        conf_threshold = 0.3,
    ):
        self.object_detector = ObjectDetector(det_config, det_checkpoint)
        self.pose_detector = PoseDetector(pose_config, pose_checkpoint)
        self.conf_threshold = conf_threshold

    def run_detection_pipeline(
        self,
        image_path: str,
        visualize: bool = False,
    ) -> VideoFrame:
        skeleton = Skeleton(
            person_id=0
        )
        
        logger.info("Starting objects detection...")

        logger.info(f"Image path: {image_path}")
        
        # 1. Detect objects
        instances = self.object_detector.detect_objects(image_path)
        
        # COCO: class 0 = person
        person_mask = (instances.labels.cpu().numpy() == 0)
        bboxes = instances.bboxes.cpu().numpy()
        scores = instances.scores.cpu().numpy()

        # filter by person class and score > 0.5
        person_bboxes = [
            bbox for bbox, label, score in zip(bboxes, instances.labels, scores)
            if label == 0 and score > self.conf_threshold
        ]
        
        logger.info(f"Number of persons detected: {len(person_bboxes)}")
        
        if len(person_bboxes) == 0:
            logger.warning("No persons detected.")
            return skeleton

        # 2. Extract poses
        pose_instances = self.pose_detector.detect_poses(image_path, person_bboxes)

        # 3. Print joints
        for person in pose_instances.keypoints:
            print("Joints (x,y):", person)

        # 4. Visualization step removed due to missing API
        if visualize:
            logger.info("Visualizing detections...")
            visualize_detections(image_path, person_bboxes, pose_instances)

        # Implement skeleton detection logic here
        return None


# COCO skeleton connections (example)
COCO_SKELETON = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (5, 6),              # shoulders
    (11, 12),            # hips
    (5, 11), (6, 12)     # torso
]

def to_numpy(x):
    """Convert torch tensor to numpy if needed"""
    logger.info("Converting to numpy...")
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.array(x)

def visualize_detections(image_path: str, person_bboxes: list, pose_instances: InstanceData):
    """Visualize detected persons and keypoints on the image"""
    image = cv2.imread(image_path)

    logger.info("Visualizing detections...")
    for bbox in person_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    logger.info("Drawing keypoints...")
    if hasattr(pose_instances, "keypoints"):
        keypoints = to_numpy(pose_instances.keypoints)  # (N, K, 2) or (N, K, 3)

        logger.info("Getting keypoint scores...")
        if hasattr(pose_instances, "keypoint_scores") and pose_instances.keypoint_scores is not None:
            logger.info("Using separate keypoint scores...")
            scores = to_numpy(pose_instances.keypoint_scores)  # (N, K)
        else:
            # If scores are included in keypoints (shape (N, K, 3))
            if keypoints.shape[-1] == 3:
                scores = keypoints[..., 2]
                keypoints = keypoints[..., :2]
            else:
                scores = np.ones(keypoints.shape[:2])

        logger.info("Drawing keypoints...")
        for person_kpts, person_scores in zip(keypoints, scores):
            logger.info("Drawing individual keypoints...")
            for (x, y), score in zip(person_kpts, person_scores):
                if score > 0.3:
                    cv2.circle(image, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)

            logger.info("Drawing skeleton connections...")
            for i1, i2 in COCO_SKELETON:
                if person_scores[i1] > 0.3 and person_scores[i2] > 0.3:
                    x1, y1 = person_kpts[i1]
                    x2, y2 = person_kpts[i2]
                    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Show the image
    cv2.imshow("Detections + Keypoints", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
