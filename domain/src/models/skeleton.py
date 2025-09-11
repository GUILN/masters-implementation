
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

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            joint_id=data["joint_id"],
            name=data["name"],
            x=data["x"],
            y=data["y"]
        )

class Skeleton:
    def __init__(
        self,
        person_id: int,
    ):
        self.person_id = person_id
        self._joints: List[SkeletonJoint] = []
        
    def to_dict(self):
        return {
            "person_id": self.person_id,
            "joints": [joint.to_dict() for joint in self.joints]
        }

    @property
    def joints(self) -> List[SkeletonJoint]:
        return self._joints

    def add_joint(self, joint: SkeletonJoint):
        self._joints.append(joint)

    @classmethod
    def from_dict(cls, data: dict):
        skeleton = cls(person_id=data["person_id"])
        for joint_data in data.get("joints", []):
            skeleton.add_joint(SkeletonJoint.from_dict(joint_data))
        return skeleton
