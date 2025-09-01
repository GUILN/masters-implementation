
from typing import List


class SkeletonJoint:
    def __init__(
        self,
        joint_id: int,
        name: str,
        x: float,
        y: float,
    ):
        self.joint_id = joint_id
        self.name = name
        self.x = x
        self.y = y

    def to_dict(self):
        return {
            "joint_id": self.joint_id,
            "name": self.name,
            "x": self.x,
            "y": self.y
        }

class Skeleton:
    def __init__(
        self,
        person_id: int,
    ):
        self.person_id = person_id
        self.joints: List[SkeletonJoint] = []
        
    def to_dict(self):
        return {
            "person_id": self.person_id,
            "joints": [joint.to_dict() for joint in self.joints]
        }

    @property
    def joints(self) -> List[SkeletonJoint]:
        return self.joints

    def add_joint(self, joint: SkeletonJoint):
        self.joints.append(joint)
