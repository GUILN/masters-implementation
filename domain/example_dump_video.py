import json
from src.models.video import Video
from src.models.video_frame import VideoFrame
from src.models.frame_object import FrameObject
from src.models.skeleton import Skeleton, SkeletonJoint

def main():
    # Create FrameObjects
    obj = FrameObject(object_class="person", bbox=[0.1, 0.2, 0.3, 0.4], confidence=0.99)

    # Create Skeleton and Joint
    joint = SkeletonJoint(joint_id=1, name="head", x=0.1, y=0.2)
    skeleton = Skeleton(person_id=1)
    skeleton.add_joint(joint)

    # Create VideoFrame and add FrameObject and Skeleton
    frame = VideoFrame(frame_id="f1", frame_sequence=1, time_stamp=0.0)
    frame.add_frame_object(obj)
    frame.add_frame_skeleton(skeleton)

    # Create Video and add VideoFrame
    video = Video(video_id="example_video", category="demo")
    video.add_frame(frame)

    # Dump to JSON file
    with open("example_video.json", "w", encoding="utf-8") as f:
        json.dump(video.to_dict(), f, indent=2, ensure_ascii=False)
    print("example_video.json created!")

if __name__ == "__main__":
    main()
