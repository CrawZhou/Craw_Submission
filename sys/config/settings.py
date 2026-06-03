
from dataclasses import dataclass
from typing import Tuple


@dataclass
# video preprocessing parameters
class VideoConfig:

    target_size: Tuple[int, int] = (1280, 640)
    fps: int = 30
    device: str = 'cuda'

    detection_conf: float = 0.55
    detection_iou: float = 0.25
    detection_min_hits: int = 5

    tracker_type: str = 'botsort.yaml'

    # ROI region (x1, y1, x2, y2)
    roi: Tuple[int, int, int, int] = (0, 160, 1280, 600)

    # BoT-SORT tracking parameters
    track_high_thresh: float = 0.20
    track_low_thresh: float = 0.05
    new_track_thresh: float = 0.20
    track_buffer: int = 150
    match_thresh: float = 0.80
    fuse_score: bool = True
    gmc_method: str = 'sparseOptFlow'
    proximity_thresh: float = 0.70
    appearance_thresh: float = 0.50
    with_reid: bool = True


@dataclass
# video segmentation parameters
class MSUConfig:

    filter_min_frames: int = 10                 

    min_window_length: int = 45                  
    max_window_length: int = 225                 

    init_overlap_threshold: float = 0.3           
    init_disjoint_ratio: float = 0.4             
    max_expansion_factor: int = 5                
    expansion_threshold: float = 0.3              


@dataclass
class InteractionConfig:
    threshold_close: int = 300                   
    threshold_far: int = 1000                     
    consecutive_points: int = 45                 


@dataclass
class ClusteringConfig:
    n_T_clusters: int = 3
    n_I_clusters: int = 3
    n_F_clusters: int = 4
    linkage_method: str = 'average'


@dataclass
class NoveltyDetectionConfig:
    """Novelty detection parameters for filtering query results."""
    t_max: int = 20                    # Time window in seconds
    hit_gap: int = 4                   # HIT_GAP = T_MAX // 5
    n_clusters: int = 3                # Number of cluster centers
    novel_ratio_th: float = 0.4        # Novel feature ratio threshold
    dist_th: float = 2.5               # Distance threshold multiplier              


@dataclass
class StorageConfig:
    base_dir: str = "D:/XW2/yolov13/system/db"
    milvus_db_path: str = None  # set dynamically from base_dir
    milvus_collection: str = "matrix_collection"
    vector_dim: int = 128
    sqlite_db_path: str = None  # set dynamically from base_dir
    clip_dir: str = None  # set dynamically from base_dir
    inverted_index_path: str = None  # set dynamically from base_dir


@dataclass
class PathConfig:
    model_path: str = "D:/XW2/yolov13/system/ultralytics/runs/ytb2/weights/best.pt"
    frame_info_path: str = "D:/XW2/yolov13/main/yolov13-main/frame_info.json"
    output_dir: str = "D:/XW2/yolov13/system/output"
    ffmpeg_path: str = "D:/XW2/yolov13/main/ffmpeg-4.4.2.tar.bz2/ffmpeg.exe"


class Config:

    def __init__(self):
        self.video = VideoConfig()
        self.msu = MSUConfig()
        self.interaction = InteractionConfig()
        self.clustering = ClusteringConfig()
        self.novelty = NoveltyDetectionConfig()
        self.storage = StorageConfig()
        self.path = PathConfig()
        self._init_storage_paths()

    def _init_storage_paths(self):
        """Initialize storage paths relative to base_dir."""
        import os
        base = self.storage.base_dir
        os.makedirs(base, exist_ok=True)
        self.storage.milvus_db_path = os.path.join(base, "msu_milvus_db")
        self.storage.sqlite_db_path = os.path.join(base, "video_db.sqlite")
        self.storage.clip_dir = os.path.join(base, "clips")
        self.storage.inverted_index_path = os.path.join(base, "inverted_index.json")
        os.makedirs(self.storage.clip_dir, exist_ok=True)

    def update_video(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.video, key):
                setattr(self.video, key, value)

    def update_msu(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.msu, key):
                setattr(self.msu, key, value)

    def update_clustering(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.clustering, key):
                setattr(self.clustering, key, value)

    def update_storage(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.storage, key):
                setattr(self.storage, key, value)

    def update_path(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.path, key):
                setattr(self.path, key, value)


_default_config = Config()


def get_config() -> Config:
    return _default_config


def set_config(config: Config):
    global _default_config
    _default_config = config
