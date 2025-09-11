import json
from src.models.video import Video
from src.models.video_frame import VideoFrame
from src.models.frame_object import FrameObject
from src.models.skeleton import Skeleton, SkeletonJoint

def test_video_from_dict_and_json():
    # Build nested objects
    obj = FrameObject(object_class="person", bbox=[0.1, 0.2, 0.3, 0.4], confidence=0.99)
    joint = SkeletonJoint(joint_id=1, name="head", x=0.1, y=0.2)
    skeleton = Skeleton(person_id=1)
    skeleton.add_joint(joint)
    frame = VideoFrame(frame_id="f1", frame_sequence=1, time_stamp=0.0)
    frame.add_frame_object(obj)
    frame.add_frame_skeleton(skeleton)
    video = Video(video_id="v1", category="demo")
    video.add_frame(frame)

    # to_dict and from_dict
    d = video.to_dict()
    video2 = Video.from_dict(d)
    assert video2.video_id == video.video_id
    assert video2.category == video.category
    assert len(video2.frames) == 1
    f2 = video2.frames[0]
    assert f2.frame_id == frame.frame_id
    assert f2.frame_sequence == frame.frame_sequence
    assert f2.time_stamp == frame.time_stamp
    assert len(f2.frame_objects) == 1
    assert f2.frame_objects[0].object_class == obj.object_class
    assert f2.frame_objects[0].bbox == obj.bbox
    assert f2.frame_objects[0].confidence == obj.confidence
    assert len(f2.frame_skeletons) == 1
    s2 = f2.frame_skeletons[0]
    assert s2.person_id == skeleton.person_id
    assert len(s2.joints) == 1
    j2 = s2.joints[0]
    assert j2.joint_id == joint.joint_id
    assert j2.name == joint.name
    assert j2.x == joint.x
    assert j2.y == joint.y

    # to_json and from_json
    json_str = json.dumps(d)
    video3 = Video.from_json(json_str)
    assert video3.video_id == video.video_id
    assert video3.category == video.category
    assert len(video3.frames) == 1
    f3 = video3.frames[0]
    assert f3.frame_id == frame.frame_id
    assert f3.frame_sequence == frame.frame_sequence
    assert f3.time_stamp == frame.time_stamp
    assert len(f3.frame_objects) == 1
    assert f3.frame_objects[0].object_class == obj.object_class
    assert f3.frame_objects[0].bbox == obj.bbox
    assert f3.frame_objects[0].confidence == obj.confidence
    assert len(f3.frame_skeletons) == 1
    s3 = f3.frame_skeletons[0]
    assert s3.person_id == skeleton.person_id
    assert len(s3.joints) == 1
    j3 = s3.joints[0]
    assert j3.joint_id == joint.joint_id
    assert j3.name == joint.name
    assert j3.x == joint.x
    assert j3.y == joint.y
