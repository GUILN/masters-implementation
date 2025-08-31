
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

class Skeleton:
    def __init__(
        self,
        person_id: int,
    ):
        self.person_id = person_id
        self.joints: List[SkeletonJoint] = []
        
    @property
    def joints(self) -> List[SkeletonJoint]:
        return self.joints

    def add_joint(self, joint: SkeletonJoint):
        self.joints.append(joint)
