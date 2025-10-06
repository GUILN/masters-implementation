from common_setup import CommonSetup
from mmengine.structures import InstanceData
import numpy as np
import cv2


logger = CommonSetup.get_logger()

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

# Use a list of maximally distinct colors (RGB)
DISTINCT_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (128, 0, 0),      # Maroon
    (0, 128, 0),      # Dark Green
    (0, 0, 128),      # Navy
    (128, 128, 0),    # Olive
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (0, 0, 0),        # Black
    (255, 255, 255),  # White
]

def to_numpy(x):
    """Convert torch tensor to numpy if needed"""
    logger.info("Converting to numpy...")
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.array(x)

def visualize_detections(
    image_path: str,
    person_bboxes: list,
    pose_instances: InstanceData,
    object_instances: InstanceData = None,
):
    """Visualize detected persons and keypoints on the image"""
    image = cv2.imread(image_path)

    logger.info("Visualizing detections...")
    for bbox in person_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    logger.info("Visualizing objects...")
    def color_for_label(label):
        # Simple color mapping for demonstration
        return DISTINCT_COLORS[int(label) % len(DISTINCT_COLORS)]
    if object_instances is not None:
        for bbox, label in zip(object_instances.bboxes, object_instances.labels):
            x1, y1, x2, y2 = map(int, bbox)
            # draw rectangle with different color for objects
            cv2.rectangle(image, (x1, y1), (x2, y2), color=color_for_label(label), thickness=2)
            cv2.putText(image, str(label), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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
