import json
import os
from src.models.video import Video
from src.models.video_frame import VideoFrame
from src.models.frame_object import FrameObject
from src.models.skeleton import Skeleton, SkeletonJoint

def test_video_to_json_file(tmp_path):
    # Create FrameObjects
    obj1 = FrameObject(object_class="person", bbox=[0.1, 0.2, 0.3, 0.4], confidence=0.95)
    obj2 = FrameObject(object_class="car", bbox=[0.5, 0.6, 0.7, 0.8], confidence=0.85)

    # Create Skeletons and Joints
    joint1 = SkeletonJoint(joint_id=1, name="head", x=0.1, y=0.2)
    joint2 = SkeletonJoint(joint_id=2, name="shoulder", x=0.3, y=0.4)
    skeleton = Skeleton(person_id=123)
    skeleton.add_joint(joint1)
    skeleton.add_joint(joint2)

    # Create VideoFrames and add FrameObjects and Skeletons
    frame1 = VideoFrame(frame_id="f1", frame_sequence=1, time_stamp=0.0)
    frame1.add_frame_object(obj1)
    frame1.add_frame_skeleton(skeleton)
    frame2 = VideoFrame(frame_id="f2", frame_sequence=2, time_stamp=0.04)
    frame2.add_frame_object(obj2)

    # Create Video and add VideoFrames
    video = Video(video_id="v1", category="test_category")
    video.add_frame(frame1)
    video.add_frame(frame2)

    # Write to JSON file
    json_path = tmp_path / "video.json"
    with open(json_path, "w") as f:
        json.dump(video.to_dict(), f, indent=2)

    # Read and assert
    with open(json_path, "r") as f:
        data = json.load(f)
    assert data["video_id"] == "v1"
    assert data["category"] == "test_category"
    assert isinstance(data["frames"], list)
    assert data["frames"][0]["frame_objects"][0]["object_class"] == "person"
    assert data["frames"][0]["frame_skeletons"][0]["person_id"] == 123
    assert data["frames"][0]["frame_skeletons"][0]["joints"][0]["name"] == "head"
    assert data["frames"][1]["frame_objects"][0]["object_class"] == "car"
