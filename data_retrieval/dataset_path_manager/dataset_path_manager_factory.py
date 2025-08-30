from enum import Enum

from dataset_path_manager.dataset_path_manager import DatasetPathManagerInterface
from dataset_path_manager.nwucla_path_manager import NwUclaPathManager

class DatasetType(Enum):
    NW_UCLA = "nw_ucla"
    OTHER = "other"

class DatasetPathManagerFactory:
    @staticmethod
    def create_path_manager(dataset_type: DatasetType, base_path: str, base_output_path: str) -> DatasetPathManagerInterface:
        if dataset_type == DatasetType.NW_UCLA:
            return NwUclaPathManager(base_path, base_output_path)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")