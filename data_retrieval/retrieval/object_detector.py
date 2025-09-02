from common_setup import CommonSetup
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
from mmdet.registry import VISUALIZERS
import torch
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
import numpy as np


logger = CommonSetup.get_logger()

HUMAN_CLASS = 0


def filter_nearest_objects(det_result, person_label=0, top_k=15):
    """
    Filter the top-k nearest objects to the detected human.
    Returns a DetDataSample (so it works with visualizer).
    """
    # Extract bboxes, scores, and labels
    bboxes = det_result.pred_instances.bboxes.cpu().numpy()
    scores = det_result.pred_instances.scores.cpu().numpy()
    labels = det_result.pred_instances.labels.cpu().numpy()

    # Identify persons
    person_indices = np.where(labels == person_label)[0]
    if len(person_indices) == 0:
        print("⚠️ No person detected in this frame.")
        return det_result  # return unfiltered if no person found

    # For now, take the first person (extend later for multi-person if needed)
    person_idx = person_indices[0]
    person_bbox = bboxes[person_idx]
    person_center = [(person_bbox[0] + person_bbox[2]) / 2,
                     (person_bbox[1] + person_bbox[3]) / 2]

    # Collect other objects
    object_indices = [i for i in range(len(labels)) if i != person_idx]

    if not object_indices:
        print("⚠️ No objects detected besides the person.")
        return det_result

    object_centers = [( (bboxes[i][0] + bboxes[i][2]) / 2,
                        (bboxes[i][1] + bboxes[i][3]) / 2) for i in object_indices]

    # Compute distances from person to objects
    distances = [np.linalg.norm(np.array(person_center) - np.array(obj_center))
                 for obj_center in object_centers]

    # Sort objects by distance
    sorted_indices = np.argsort(distances)

    # Keep top-k nearest objects
    keep_objects = [object_indices[i] for i in sorted_indices[:top_k]]

    # Always keep the person
    keep_indices = [person_idx] + keep_objects

    # Build new InstanceData
    filtered_instances = InstanceData(
        bboxes=det_result.pred_instances.bboxes[keep_indices],
        scores=det_result.pred_instances.scores[keep_indices],
        labels=det_result.pred_instances.labels[keep_indices]
    )

    # Wrap into a new DetDataSample
    filtered_result = DetDataSample()
    filtered_result.pred_instances = filtered_instances
    filtered_result.set_metainfo(det_result.metainfo)

    return filtered_result

def filter_limiting_number_of_objects(
    results: DetDataSample,
    limit_objects: int,
) -> DetDataSample:
    logger.info("Filtering detections...")
    scores = results.pred_instances.scores
    logger.info(f"Original scores: {scores}")
    keep_idx = torch.topk(scores, k=limit_objects).indices
    logger.info(f"Indices of top {limit_objects} scores: {keep_idx}")
    filtered_instances = results.pred_instances[keep_idx]
    logger.info(f"Filtered to top {limit_objects}")
    
    filtered_result = DetDataSample()
    filtered_result.pred_instances = filtered_instances
    logger.info(f"Setting metainfo for filtered results")
    filtered_result.set_metainfo(results.metainfo)
    logger.info("Returning filtered results")
    return filtered_result

def filter_k_elements_nearest_to_human(
    det_sample: DetDataSample,
    score_thr=0.01,
    top_k: int = 15,
) -> DetDataSample:
    instances = det_sample.pred_instances

    # Score threshold
    keep = instances.scores > score_thr
    instances = instances[keep]

    # Top-k
    if top_k is not None and len(instances) > top_k:
        top_idx = instances.scores.argsort(descending=True)[:top_k]
        instances = instances[top_idx]

    # Create a new DetDataSample with filtered instances
    filtered = DetDataSample()
    filtered.pred_instances = InstanceData()
    filtered.pred_instances = instances
    filtered.set_metainfo(det_sample.metainfo)

    return filtered

def filter_detections(results, limit_objects: int):
    # return filter_limiting_number_of_objects(results, limit_objects)
    # return filter_k_elements_nearest_to_human(results, top_k=limit_objects)
    return filter_nearest_objects(results)

class ObjectDetector:
    def __init__(
        self,
        config_file: str = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
        checkpoint_file: str = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    ):
        logger.info("Initializing object detector...")
        self._model = init_detector(config_file, checkpoint_file, device='cpu')
        
    def detect(
        self,
        image_path: str,
        output_file: str,
        score_threshold: float = 0.5,
        limit_objects: int = 10,
    ):
        logger.info(f"Running object detection on {image_path}, saving to {output_file}")
        results = inference_detector(self._model, image_path)
        logger.info(f"Filtering {limit_objects} objects...")
        filtered_results = filter_detections(results, limit_objects)
        logger.info(f"Reading image from {image_path}")
        img = mmcv.imread(image_path)
        visualizer = VISUALIZERS.build(self._model.cfg.visualizer)
        visualizer.add_datasample(
             name='result',
            image=img,
            data_sample=filtered_results,
            draw_gt=False,
            draw_pred=True,
            show=False,
            out_file=output_file
        )
        logger.info(f"Detection results saved to {output_file}")
