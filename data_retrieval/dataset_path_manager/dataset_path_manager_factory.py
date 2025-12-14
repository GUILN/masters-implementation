from enum import Enum

from dataset_path_manager.dataset_path_manager import DatasetPathManagerInterface
from dataset_path_manager.ntu_rgbd_path_manager import NTURGBDPathManager
from dataset_path_manager.nwucla_path_manager import NwUclaPathManager
from dataset_path_manager.unsafe_net_path_manager import UnsafeNetPathManager

class DatasetType(Enum):
    NW_UCLA = "nw_ucla"
    UNSAFE_NET = "unsafe_net"
    NTU_RGB_D = "ntu_rgb_d"
    OTHER = "other"

class DatasetPathManagerFactory:
    @staticmethod
    def create_path_manager(dataset_type: DatasetType, base_path: str, base_output_path: str) -> DatasetPathManagerInterface:
        if dataset_type == DatasetType.NW_UCLA:
            return NwUclaPathManager(base_path, base_output_path)
        elif dataset_type == DatasetType.UNSAFE_NET:
            return UnsafeNetPathManager(base_path, base_output_path)
        elif dataset_type == DatasetType.NTU_RGB_D:
            return NTURGBDPathManager(base_path, base_output_path)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")