from common_setup import CommonSetup
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples
from mmdet.apis import inference_detector, init_detector
import mmcv
import os
from src.models.skeleton import Skeleton
from mmpose.utils import register_all_modules as register_pose_modules
from mmdet.utils import register_all_modules as register_det_modules
from mmengine.registry import DefaultScope



# --- Configs ---

register_det_modules(init_default_scope=False)   # registers mmdet stuff
register_pose_modules(init_default_scope=False)  # registers mmpose stuff
# Important: must come before building datasets/models


logger = CommonSetup.get_logger()

class SkeletonDetector:
    @staticmethod
    def detect_skeletons(
        image_path: str,
        det_config = '/home/guilherme/Mestrado/masters-implementation/data_retrieval/retrieval/config/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
        det_checkpoint = '/home/guilherme/Mestrado/masters-implementation/data_retrieval/retrieval/config/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        pose_config = '/home/guilherme/Mestrado/masters-implementation/data_retrieval/retrieval/mmpose/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
        pose_checkpoint = '/home/guilherme/Mestrado/masters-implementation/data_retrieval/retrieval/config/checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
    ) -> Skeleton:
        skeleton = Skeleton(
            person_id=0
        )
        
        logger.info("Starting skeleton detection...")

        logger.info(f"Image path: {image_path}")
        logger.info(f"Detection config: {det_config}")
        # --- Init models ---
        with DefaultScope.overwrite_default_scope('mmdet'):
            det_model = init_detector(det_config, det_checkpoint, device='cpu')
            logger.info("Models initialized.") 
            img_path = image_path

            # 1. Detect humans
            det_result = inference_detector(det_model, img_path)
            person_detections = det_result[0]  # class 0 = person in COCO
            person_bboxes = [bbox for bbox in person_detections if bbox[4] > 0.5]  # filter by score

        with DefaultScope.overwrite_default_scope('mmpose'):
            pose_model = init_model(pose_config, pose_checkpoint, device='cpu')
            logger.info(f"Number of persons detected: {len(person_bboxes)}")
            # 2. Extract skeleton joints
            pose_results = inference_topdown(pose_model, img_path, person_bboxes)
            pose_data = merge_data_samples(pose_results)

            # 3. Print joints
            for person in pose_data.pred_instances.keypoints:
                print("Joints (x,y):", person)

            # 4. Visualization step removed due to missing API

            # Implement skeleton detection logic here
            return skeleton
