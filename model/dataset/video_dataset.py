from torch.utils.data import Dataset

from dataset.video_loader import VideoDataLoader


class VideoDataset(Dataset):
    def __init__(
        self,
        video_data_loader: VideoDataLoader,
    ):
        self._video_data_loader = video_data_loader
