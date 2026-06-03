from typing import Dict, List, Tuple, Optional, Any
import json

from config.settings import Config
from data.frame_parser import FrameParser
from data.msu_splitter import MSUSplitter
from features.feature_builder import FeatureBuilder
from clustering.hierarchical_index import HierarchicalClusterIndex
from storage.inverted_index import InvertedIndex
from storage.milvus_storage import MilvusStorage
from storage.sqlite_manager import SQLiteManager
from query.query_manager import QueryManager
from utils.logger import setup_logger, get_logger

setup_logger(level=1)
logger = get_logger("System")


class System:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.frame_parser = None
        self.hierarchical_index = None
        self.inverted_index = None
        self.query_manager = None
        self._frame_info = None
        self._msu_ranges = None
        self._msu_durations = None
        self._F = None
        self._I = None
        self._T = None
        self._source_video_path = None

    def load_frame_info(self, json_path: str) -> 'System':
        self.frame_parser = FrameParser()
        self.frame_parser.load_from_json(json_path)
        self._frame_info = self.frame_parser.frame_info
        # Try to infer video path from json_path if possible
        self._source_video_path = None
        logger.info(f"Loaded: {self.frame_parser.total_frames} frames")
        return self

    def run_from_video(self, video_path: str, roi: Optional[Tuple[int, int, int, int]] = None) -> 'System':
        from data.tracker import Tracker
        self._source_video_path = video_path
        tracker = Tracker(
            model_path=self.config.path.model_path,
            tracker_type=self.config.video.tracker_type,
            conf=self.config.video.detection_conf,
            iou=self.config.video.detection_iou,
            device=self.config.video.device
        )
        result = tracker.track_video(video_path, roi=roi)
        self._frame_info = result['frame_info']
        logger.info(f"Tracking done: {result['total_frames']} frames")
        return self

    def run_full_pipeline(self, db_path: str = None, output_dir: str = None) -> QueryManager:
        if self._frame_info is None:
            raise ValueError("frame_info not loaded")

        splitter = MSUSplitter(self.config.msu)
        self._msu_ranges, self._msu_durations = splitter.split(self._frame_info)
        logger.info(f"Segmentation done: {len(self._msu_ranges)} MSUs")

        fb = FeatureBuilder(self.config)
        fb.set_msu_splits(self._msu_ranges, self._msu_durations)
        self._F, self._I, self._T = fb.build(self._frame_info, skip_splits=True)
        logger.info(f"Features done: {len(self._F)} MSUs")

        # Save intermediate files to output_dir (beside video, not in db)
        if output_dir:
            self._save_intermediate_files(output_dir)

        self.hierarchical_index = HierarchicalClusterIndex(self.config.clustering)
        self.hierarchical_index.build(self._F, self._I, self._T)

        self.inverted_index = InvertedIndex(db_path=self.config.storage.inverted_index_path)
        for i in range(len(self._F)):
            self.inverted_index.update(i, self._F[i], self._I[i])
        self.inverted_index.save()

        # Initialize SQLite and store video/MSU clip info
        sqlite_mgr = SQLiteManager(db_path=self.config.storage.sqlite_db_path)
        video_id = sqlite_mgr.insert_video(self._source_video_path or "")
        sqlite_mgr.insert_vsus_from_ranges(
            msu_ranges=self._msu_ranges,
            clip_dir=self.config.storage.clip_dir,
            source_video_path=self._source_video_path or "",
            source_video_id=video_id if video_id > 0 else None,
            video_fps=float(self.config.video.fps)
        )
        sqlite_mgr.close()
        logger.info(f"SQLite database saved: {self.config.storage.sqlite_db_path}")

        # Save clips (video segments) using ffmpeg
        self._save_clips()
        logger.info(f"Video clips saved: {self.config.storage.clip_dir}")

        milvus_db_path = self.config.storage.milvus_db_path
        milvus_storage = MilvusStorage(
            db_path=milvus_db_path,
            collection_name=self.config.storage.milvus_collection,
            vector_dim=self.config.storage.vector_dim
        )
        milvus_storage.batch_insert(
            F_list=self._F,
            I_list=self._I,
            T_list=self._T,
            T_labels=list(self.hierarchical_index.T_labels_),
            I_labels=list(self.hierarchical_index.I_labels_),
            F_labels=list(self.hierarchical_index.F_labels_)
        )
        milvus_storage.flush()
        milvus_storage.close()
        logger.info(f"Milvus database saved: {milvus_db_path}.db")

        self.query_manager = QueryManager(
            milvus_storage=milvus_storage,
            inverted_index=self.inverted_index,
            hierarchical_index=self.hierarchical_index
        )

        logger.info("Pipeline done")
        return self.query_manager

    def _save_intermediate_files(self, output_dir: str):
        """Save frame_info.json and object_frame_statistics.json to output_dir."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        if self._frame_info:
            # Save frame_info.json
            frame_info_path = os.path.join(output_dir, "frame_info.json")
            with open(frame_info_path, 'w', encoding='utf-8') as f:
                json.dump(self._frame_info, f, indent=2)
            logger.info(f"Saved frame_info.json: {frame_info_path}")

            # Save object_frame_statistics.json
            obj_stats = self._build_object_frame_statistics(self._frame_info)
            obj_stats_path = os.path.join(output_dir, "object_frame_statistics.json")
            with open(obj_stats_path, 'w', encoding='utf-8') as f:
                json.dump(obj_stats, f, indent=2)
            logger.info(f"Saved object_frame_statistics.json: {obj_stats_path}")

        if self._msu_ranges and self._msu_durations:
            msu_data = {
                'msu_ranges': self._msu_ranges,
                'msu_durations': [
                    {str(k): v for k, v in d.items()}
                    for d in self._msu_durations
                ]
            }
            msu_splits_path = os.path.join(output_dir, "msu_splits.json")
            with open(msu_splits_path, 'w', encoding='utf-8') as f:
                json.dump(msu_data, f, indent=2)
            logger.info(f"Saved msu_splits.json: {msu_splits_path}")

    def _build_object_frame_statistics(self, frame_info: Dict) -> Dict[int, List[int]]:
        """Build object_frame_statistics.json format (object_id -> list of frames)."""
        from collections import defaultdict
        result = defaultdict(list)
        for frame_id, detections in frame_info.items():
            frame_num = int(frame_id)
            for det in detections:
                result[det['id']].append(frame_num)
        for obj_id in result:
            result[obj_id].sort()
        return dict(result)

    def find_similar(self, msu_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.query_manager is None:
            raise ValueError("run_full_pipeline() must be called first")
        return self.query_manager.find_similar_by_id(msu_id, top_k)

    @property
    def F(self) -> List:
        return self._F

    @property
    def msu_ranges(self) -> List:
        return self._msu_ranges

    @property
    def msu_count(self) -> int:
        return len(self._F) if self._F else 0

    def summary(self) -> str:
        return (
            f"MSU: {self.msu_count} segments\n"
            f"Clusters: {'built' if self.hierarchical_index else 'none'}\n"
            f"Index: {'ready' if self.query_manager else 'none'}"
        )

    def close(self):
        if self.inverted_index:
            self.inverted_index.close()

    def _save_clips(self):
        """Save video clips for each MSU using ffmpeg."""
        import os
        import subprocess

        if not self._source_video_path or not os.path.exists(self._source_video_path):
            logger.warning(f"Source video not found: {self._source_video_path}, skipping clip generation")
            return

        clip_dir = self.config.storage.clip_dir
        os.makedirs(clip_dir, exist_ok=True)
        ffmpeg_path = self.config.path.ffmpeg_path

        # Use system ffmpeg if configured path doesn't exist
        if not os.path.exists(ffmpeg_path):
            ffmpeg_cmd = "ffmpeg"
        else:
            ffmpeg_cmd = ffmpeg_path

        fps = self.config.video.fps

        for msu_id, (start, end) in enumerate(self._msu_ranges):
            clip_name = f"msu_{msu_id:04d}_f{start}-{end}.mp4"
            clip_path = os.path.join(clip_dir, clip_name)

            if os.path.exists(clip_path):
                logger.info(f"Clip already exists: {clip_path}")
                continue

            start_sec = start / fps
            end_sec = end / fps
            duration = end_sec - start_sec

            cmd = [
                ffmpeg_cmd, "-y",
                "-i", self._source_video_path,
                "-ss", str(start_sec),
                "-t", str(duration),
                "-c:v", "libx264",
                "-an",
                clip_path
            ]

            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=120)
                logger.info(f"Saved clip: {clip_path}")
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout saving clip: {clip_path}")
            except Exception as e:
                logger.warning(f"Failed to save clip {clip_path}: {e}")