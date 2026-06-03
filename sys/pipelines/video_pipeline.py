import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from data.tracker import Tracker
from data.msu_splitter import MSUSplitter
from features.feature_builder import FeatureBuilder
from config.settings import Config
from utils.logger import get_logger

logger = get_logger("VideoPipeline")


class VideoPipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.tracker = Tracker(
            model_path=self.config.path.model_path,
            tracker_type=self.config.video.tracker_type,
            conf=self.config.video.detection_conf,
            iou=self.config.video.detection_iou,
            min_hits=self.config.video.detection_min_hits,
            device=self.config.video.device
        )
        self.feature_builder = FeatureBuilder(self.config)

        self.frame_info = None
        self.msu_ranges = None
        self.msu_durations = None
        self.F = None
        self.I = None
        self.T = None

    def run_tracking(
        self,
        video_path: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
        save_json: bool = True,
        json_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        logger.info(f"Starting tracking: {video_path}")

        # Auto-generate output paths if not provided
        if output_dir is None:
            output_dir = str(Path(video_path).parent / f"{Path(video_path).stem}_output")
        if json_path is None and save_json:
            json_path = str(Path(output_dir) / "frame_info.json")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        result = self.tracker.track_video(
            video_path=video_path,
            roi=roi,
            save_json=save_json,
            json_path=json_path
        )

        self.frame_info = result['frame_info']

        # Also save object_frame_statistics.json
        if save_json and self.frame_info:
            obj_stats = self._build_object_frame_statistics(self.frame_info)
            obj_stats_path = Path(output_dir) / "object_frame_statistics.json"
            with open(obj_stats_path, 'w', encoding='utf-8') as f:
                json.dump(obj_stats, f, indent=2)
            logger.info(f"Saved object_frame_statistics.json: {obj_stats_path}")

        logger.info(f"Tracking done: {result['total_frames']} frames, {len(self.frame_info)} valid frames")

        return result

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

    def run_segmentation(self) -> Tuple[List, List]:
        if self.frame_info is None:
            raise ValueError("Tracking must be run first")

        logger.info("Starting MSU segmentation")

        splitter = MSUSplitter(self.config.msu)
        self.msu_ranges, self.msu_durations = splitter.split(self.frame_info)

        logger.info(f"Segmentation done: {len(self.msu_ranges)} MSUs")

        return self.msu_ranges, self.msu_durations

    def run_feature_extraction(self) -> Tuple[List, List, List]:
        if self.frame_info is None:
            raise ValueError("Tracking must be run first")

        if self.msu_ranges is None:
            self.run_segmentation()

        logger.info("Starting feature extraction")

        self.feature_builder.set_msu_splits(self.msu_ranges, self.msu_durations)
        self.F, self.I, self.T = self.feature_builder.build(
            self.frame_info,
            skip_splits=True
        )

        logger.info(f"Feature extraction done: {len(self.F)} MSUs")

        return self.F, self.I, self.T

    def run_full_pipeline(
        self,
        video_path: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
        save_intermediate: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict:
        tracking_result = self.run_tracking(video_path, roi)
        self.run_segmentation()
        self.run_feature_extraction()

        if save_intermediate:
            out_dir = output_dir or str(Path(video_path).parent / f"{Path(video_path).stem}_output")
            self.save_results(out_dir)

        return {
            'tracking': tracking_result,
            'msu_ranges': self.msu_ranges,
            'msu_durations': self.msu_durations,
            'F': self.F,
            'I': self.I,
            'T': self.T
        }

    def save_results(self, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.frame_info:
            # Save frame_info.json
            with open(output_path / "frame_info.json", 'w', encoding='utf-8') as f:
                json.dump(self.frame_info, f, indent=2)

            # Save object_frame_statistics.json
            obj_stats = self._build_object_frame_statistics(self.frame_info)
            with open(output_path / "object_frame_statistics.json", 'w', encoding='utf-8') as f:
                json.dump(obj_stats, f, indent=2)

        if self.msu_ranges and self.msu_durations:
            msu_data = {
                'msu_ranges': self.msu_ranges,
                'msu_durations': [
                    {str(k): v for k, v in d.items()}
                    for d in self.msu_durations
                ]
            }
            with open(output_path / "msu_splits.json", 'w', encoding='utf-8') as f:
                json.dump(msu_data, f, indent=2)

        logger.info(f"Results saved to: {output_dir}")

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "Video Pipeline Summary",
            "=" * 50,
            f"Frame Info: {'loaded' if self.frame_info else 'not loaded'}",
            f"MSU count: {len(self.msu_ranges) if self.msu_ranges else 0}",
            f"F matrix: {'extracted' if self.F else 'not extracted'}",
            f"I matrix: {'extracted' if self.I else 'not extracted'}",
            f"T matrix: {'extracted' if self.T else 'not extracted'}",
        ]
        return "\n".join(lines)