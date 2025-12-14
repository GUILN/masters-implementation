from collections import defaultdict
from common_setup import CommonSetup
import numpy as np
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

logger = CommonSetup.get_logger()


def filter_nearest_objects(
    pred_instances,
    person_bbox,
    max_per_class=2,
    person_label=0,
    top_k=15
) -> DetDataSample:
    """
    Keep up to top_k nearest objects total, but max_per_class per class.
    Always keeps at least one person.
    """
    logger.info("Starting filtering of nearest objects...")
    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    if len(bboxes) == 0:
        logger.warning("⚠️ No objects detected.")
        return DetDataSample()
    person_center = [(person_bbox[0] + person_bbox[2]) / 2,
                     (person_bbox[1] + person_bbox[3]) / 2]

    logger.info(f"Person bbox: {person_bbox}")
    logger.info(f"Person center: {person_center}")
    object_indices = [i for i in range(len(labels)) if labels[i] != person_label]
    if not object_indices:
        logger.warning("⚠️ No objects detected besides the person.")
        return pred_instances

    logger.info(f"Computing object centers to {len(object_indices)} objects...")
    object_centers = [((bboxes[i][0] + bboxes[i][2]) / 2,
                       (bboxes[i][1] + bboxes[i][3]) / 2) for i in object_indices]

    logger.info(f"Computing distances to {len(object_indices)} objects...")
    distances = [np.linalg.norm(np.array(person_center) - np.array(obj_center))
                 for obj_center in object_centers]

    logger.info("Sorting objects by distance...")
    sorted_indices = np.argsort(distances)
    sorted_object_indices = [object_indices[i] for i in sorted_indices]

    logger.info("Selecting top-k nearest objects with class constraints...")
    class_counts = defaultdict(int)
    final_keep = []
    for idx in sorted_object_indices:
        cls = labels[idx]
        if class_counts[cls] < max_per_class:
            final_keep.append(idx)
            class_counts[cls] += 1
        if len(final_keep) >= top_k:  # stop once we hit total cap
            break

    keep_indices = final_keep

    logger.info("Filtering instances...")
    filtered_instances = InstanceData(
        bboxes=bboxes[keep_indices],
        scores=scores[keep_indices],
        labels=labels[keep_indices]
    )
    filtered_result = DetDataSample()
    filtered_result.pred_instances = filtered_instances

    logger.info(f"Filtering complete. Kept {len(keep_indices)} objects.")
    for k_idx in keep_indices:
        logger.info(f"Kept object - Class: {labels[k_idx]}, Score: {scores[k_idx]}")
    return filtered_result
