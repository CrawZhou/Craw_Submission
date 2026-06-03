import json
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from utils.logger import get_logger

logger = get_logger("FrameParser")


class FrameParser:
    def __init__(self, frame_info: Optional[Dict[str, List[Dict]]] = None):
        self.frame_info = frame_info or {}
        self._obj_frames = None
        self._obj_ids = None

    def load_from_json(self, json_path: str) -> 'FrameParser':
        with open(json_path, 'r', encoding='utf-8') as f:
            self.frame_info = json.load(f)
        self._invalidate_cache()
        return self

    def save_to_json(self, json_path: str) -> None:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.frame_info, f, indent=2)

    def _invalidate_cache(self) -> None:
        self._obj_frames = None
        self._obj_ids = None

    @property
    def total_frames(self) -> int:
        return len(self.frame_info)

    @property
    def all_object_ids(self) -> Set[int]:
        if self._obj_ids is None:
            self._obj_ids = set()
            for frame_id, detections in self.frame_info.items():
                for det in detections:
                    self._obj_ids.add(det['id'])
        return self._obj_ids

    def get_object_frames(self, obj_id: int) -> List[int]:
        if self._obj_frames is None:
            self._obj_frames = defaultdict(list)
            for frame_id, detections in self.frame_info.items():
                frame_num = int(frame_id)
                for det in detections:
                    self._obj_frames[det['id']].append(frame_num)

            for obj_id in self._obj_frames:
                self._obj_frames[obj_id].sort()

        return self._obj_frames.get(obj_id, [])

    def get_frame_detections(self, frame_id: int) -> List[Dict]:
        return self.frame_info.get(str(frame_id), [])

    def get_object_trajectory(
        self,
        obj_id: int,
        coordinate_keys: Tuple[str, str] = ('cx', 'cy')
    ) -> List[Tuple[float, float]]:
        frames = self.get_object_frames(obj_id)
        trajectory = []

        for frame_id in frames:
            detections = self.get_frame_detections(frame_id)
            for det in detections:
                if det['id'] == obj_id:
                    trajectory.append((det[coordinate_keys[0]], det[coordinate_keys[1]]))
                    break

        return trajectory

    def filter_objects_by_min_frames(self, min_frames: int) -> Set[int]:
        filtered = set()
        for obj_id in self.all_object_ids:
            frames = self.get_object_frames(obj_id)
            if len(frames) >= min_frames:
                filtered.add(obj_id)
        return filtered

    def get_object_count_per_frame(self) -> Dict[int, int]:
        counts = {}
        for frame_id, detections in self.frame_info.items():
            counts[int(frame_id)] = len(detections)
        return counts

    def get_statistics(self) -> Dict[str, any]:
        obj_counts = self.get_object_count_per_frame()

        total_objects = len(self.all_object_ids)
        total_detections = sum(obj_counts.values())
        avg_objects = total_detections / max(self.total_frames, 1)

        frame_counts = [len(self.get_object_frames(oid)) for oid in self.all_object_ids]
        if frame_counts:
            min_frames = min(frame_counts)
            max_frames = max(frame_counts)
            avg_frames = sum(frame_counts) / len(frame_counts)
        else:
            min_frames = max_frames = avg_frames = 0

        return {
            'total_frames': self.total_frames,
            'total_objects': total_objects,
            'total_detections': total_detections,
            'avg_objects_per_frame': avg_objects,
            'min_object_appearances': min_frames,
            'max_object_appearances': max_frames,
            'avg_object_appearances': avg_frames
        }

    def to_object_frame_statistics(self) -> Dict[str, List[int]]:
        """Convert to object_frame_statistics.json format (object_id -> list of frames).

        This matches main's object_frame_statistics.json structure.
        Each key is object ID, value is list of frame numbers where that object appears.
        """
        result = defaultdict(list)
        for frame_id, detections in self.frame_info.items():
            frame_num = int(frame_id)
            for det in detections:
                result[det['id']].append(frame_num)

        # Sort each object's frame list
        for obj_id in result:
            result[obj_id].sort()

        return dict(result)


def load_frame_info(json_path: str) -> Dict[str, List[Dict]]:
    parser = FrameParser()
    parser.load_from_json(json_path)
    return parser.frame_info