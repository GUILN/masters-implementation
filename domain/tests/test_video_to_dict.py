from src.models.video import Video
from src.models.video_frame import VideoFrame
from src.models.frame_object import FrameObject
from src.models.skeleton import Skeleton, SkeletonJoint
def test_video_to_dict():
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

    # Convert to dict
    video_dict = video.to_dict()

    # Assert structure
    assert video_dict["video_id"] == "v1"
    assert video_dict["category"] == "test_category"
    assert isinstance(video_dict["frames"], list)
    assert len(video_dict["frames"]) == 2
    assert video_dict["frames"][0]["frame_id"] == "f1"
    assert video_dict["frames"][0]["frame_sequence"] == 1
    assert video_dict["frames"][0]["time_stamp"] == 0.0
    assert isinstance(video_dict["frames"][0]["frame_objects"], list)
    assert video_dict["frames"][0]["frame_objects"][0]["object_class"] == "person"
    assert video_dict["frames"][0]["frame_objects"][0]["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert video_dict["frames"][0]["frame_objects"][0]["confidence"] == 0.95
    assert isinstance(video_dict["frames"][0]["frame_skeletons"], list)
    assert video_dict["frames"][0]["frame_skeletons"][0]["person_id"] == 123
    assert len(video_dict["frames"][0]["frame_skeletons"][0]["joints"]) == 2
    assert video_dict["frames"][0]["frame_skeletons"][0]["joints"][0]["name"] == "head"
    assert video_dict["frames"][1]["frame_objects"][0]["object_class"] == "car"
    assert video_dict["frames"][1]["frame_objects"][0]["bbox"] == [0.5, 0.6, 0.7, 0.8]
    assert video_dict["frames"][1]["frame_objects"][0]["confidence"] == 0.85
