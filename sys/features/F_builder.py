from typing import Dict, List, Tuple, Any
import numpy as np

from data.frame_parser import FrameParser
from data.msu_splitter import MSUSplitter
from .trajectory_extractor import TrajectoryExtractor


class FMatrixBuilder:
    def __init__(self, extract_keypoints: bool = True):
        self.extract_keypoints = extract_keypoints
        self.trajectory_extractor = TrajectoryExtractor() if extract_keypoints else None

    def build(
        self,
        frame_info: Dict[str, List[Dict]],
        msu_ranges: List[Tuple[int, int]],
        msu_durations: List[Dict[int, Tuple[int, int]]]
    ) -> Tuple[List[List[List]], List[List[List]]]:
        parser = FrameParser(frame_info)
        F_matrices = []
        raw_trajectories = []

        for msu_idx, (start, end) in enumerate(msu_ranges):
            trajectories = self._extract_msu_trajectories(
                parser, msu_durations[msu_idx], start, end
            )
            raw_trajectories.append(trajectories)

            if self.extract_keypoints and self.trajectory_extractor:
                F_matrix = self.trajectory_extractor.extract_keypoints_batch(trajectories)
            else:
                F_matrix = trajectories

            F_matrices.append(F_matrix)

        return F_matrices, raw_trajectories

    def _extract_msu_trajectories(
        self,
        parser: FrameParser,
        duration_map: Dict[int, Tuple[int, int]],
        start: int,
        end: int
    ) -> List[List]:
        trajectories = []

        for obj_id, (first, last) in duration_map.items():
            # Get the frames where this object actually appears
            obj_frames = parser.get_object_frames(obj_id)

            trajectory = [0]

            for frame_id in obj_frames:
                detections = parser.get_frame_detections(frame_id)
                cx, cy = None, None

                for det in detections:
                    if det['id'] == obj_id:
                        cx = round(det['cx'], 2)
                        cy = round(det['cy'], 2)
                        break

                if cx is not None and cy is not None:
                    trajectory.extend([cx, cy])

            cls = self._get_object_class(parser, obj_id, start, end)
            trajectory[0] = cls

            trajectories.append(trajectory)

        return trajectories

    def _get_object_class(
        self,
        parser: FrameParser,
        obj_id: int,
        start: int,
        end: int
    ) -> int:
        for frame_id in range(start, end + 1):
            detections = parser.get_frame_detections(frame_id)
            for det in detections:
                if det['id'] == obj_id:
                    return det['cls']
        return 0