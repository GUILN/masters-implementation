from common_setup import CommonSetup
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples
from mmdet.apis import inference_detector, init_detector
from retrieval.image_visualizer import visualize_detections
from retrieval.object_filter import filter_nearest_objects
from src.models.skeleton import Skeleton, SkeletonJoint
from src.models.video_frame import VideoFrame
from src.models.frame_object import FrameObject
from mmpose.utils import register_all_modules as register_pose_modules
from mmdet.utils import register_all_modules as register_det_modules
from mmengine.registry import DefaultScope
from mmengine.structures import InstanceData
from typing import List, NamedTuple

class FrameInfo(NamedTuple):
    frame_id: int
    frame_sequence: int
    timestamp: float

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
        max_per_class=2,
        top_k_objects=10,
    ):
        self.object_detector = ObjectDetector(det_config, det_checkpoint)
        self.pose_detector = PoseDetector(pose_config, pose_checkpoint)
        self._conf_threshold = conf_threshold
        self._max_per_class = max_per_class
        self._top_k_objects = top_k_objects

    def run_detection_pipeline(
        self,
        image_path: str,
        frame_info: FrameInfo,
        visualize: bool = False,
    ) -> VideoFrame:
        video_frame = VideoFrame(
            frame_info.frame_id,
            frame_info.frame_sequence,
            frame_info.timestamp
        )
        skeleton = Skeleton(
            person_id=0
        )

        logger.info("Starting objects detection...")

        logger.info(f"Image path: {image_path}")

        # 1. Detect objects
        instances = self.object_detector.detect_objects(image_path)

        bboxes = instances.bboxes.cpu().numpy()
        scores = instances.scores.cpu().numpy()

        # filter by person class and score > 0.5
        person_bboxes = [
            bbox for bbox, label, score in zip(bboxes, instances.labels, scores)
            if label == 0 and score > self._conf_threshold
        ]

        logger.info(f"Number of persons detected: {len(person_bboxes)}")

        if len(person_bboxes) == 0:
            logger.warning("⚠️ No person detected.")
            return None
        elif len(person_bboxes) > 1:
            logger.warning("⚠️ Multiple persons detected. Using the first one.")
            logger.info("Filtering to detect the biggest bounding box...")
            person_bboxes = sorted(person_bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            person_bboxes = [person_bboxes[0]]
           


        # 2. Extract poses
        logger.info("Starting pose estimation...")
        pose_instances = self.pose_detector.detect_poses(image_path, person_bboxes)

        # 3. Print joints
        logger.info("Constructing skeleton...")
        for seq, (x, y) in enumerate(pose_instances.keypoints[0]):
            skeleton.add_joint(
                SkeletonJoint(
                    joint_id=seq,
                    name=str(seq),
                    x=x,
                    y=y,
                )
            )

        logger.info("Adding skeleton to video frame...")
        video_frame.add_frame_skeleton(skeleton)
        logger.info("Filtering nearest objects...")
        filtered_objects = filter_nearest_objects(
            pred_instances=instances,
            person_bbox=person_bboxes[0],
        )
        logger.debug("built skeleton")
        
        zipped = zip(
            filtered_objects.pred_instances.labels,
            filtered_objects.pred_instances.bboxes,
            filtered_objects.pred_instances.scores
        )
        logger.debug("Creating FrameObject instances...")
        for label, bbox, score in zipped:
            video_frame.add_frame_object(
                FrameObject(
                    object_class=str(int(label)),
                    bbox=[round(float(coord), 2) for coord in bbox],
                    confidence=round(float(score), 2),
                )
            )
        logger.debug("Populated video_frame with objects.")

        # 4. Visualization step
        if visualize:
            logger.info("Visualizing detections...")
            visualize_detections(
                image_path,
                person_bboxes,
                pose_instances,
                filtered_objects.pred_instances,
            )

        return video_frame

    def extract_video_frames(
        self,
        video_paths: List[str],
        save_dir: str = './data/frames',
    ):
        pass
