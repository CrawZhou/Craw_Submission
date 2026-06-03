import json
import time
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

from utils.logger import get_logger

logger = get_logger("Tracker")


class Tracker:
    def __init__(
        self,
        model_path: str,
        tracker_type: str = 'bytetrack.yaml',
        conf: float = 0.55,
        iou: float = 0.25,
        min_hits: int = 5,
        device: str = 'cuda'
    ):
        self.model_path = model_path
        self.tracker_type = tracker_type
        self.conf = conf
        self.iou = iou
        self.min_hits = min_hits
        self.device = device
        self.model = YOLO(model_path)
        self.track_hits = defaultdict(int)
        self.prev_ids = set()

    def track_video(
        self,
        video_path: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
        output_path: Optional[str] = None,
        save_json: bool = False,
        json_path: Optional[str] = None
    ) -> Dict[str, Any]:
        frame_info = defaultdict(list)
        trajectories = defaultdict(list)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.track_hits.clear()
        self.prev_ids = set()
        frame_idx = 0
        start_time = time.perf_counter()

        results = self.model.track(
            source=video_path,
            stream=True,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker_type,
            device=self.device,
            persist=True,
            verbose=False
        )

        for result in results:
            curr_ids = set()

            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()

                for (cx, cy, w, h), tid, cls in zip(boxes, track_ids, classes):
                    offset_x, offset_y = 0, 0
                    if roi:
                        offset_x, offset_y = roi[0], roi[1]
                        cx += offset_x
                        cy += offset_y

                    self.track_hits[tid] += 1

                    if self.track_hits[tid] < self.min_hits:
                        continue

                    curr_ids.add(tid)

                    frame_info[frame_idx].append({
                        'id': int(tid),
                        'cls': int(cls),
                        'cx': float(cx),
                        'cy': float(cy),
                        'w': float(w),
                        'h': float(h)
                    })

                    trajectories[int(tid)].append((float(cx), float(cy)))

            for tid in (self.prev_ids - curr_ids):
                self.track_hits[tid] = max(0, self.track_hits[tid] - 1)

            self.prev_ids = curr_ids

            if writer:
                annotated = result.plot()
                if roi:
                    x1, y1, x2, y2 = roi
                    annotated[0:y1, :] = 0
                    annotated[y2:, :] = 0
                    annotated[:, 0:x1] = 0
                    annotated[:, x2:] = 0
                writer.write(annotated)

            frame_idx += 1

        if writer:
            writer.release()

        elapsed = time.perf_counter() - start_time

        if save_json and json_path:
            with open(json_path, 'w') as f:
                json.dump(dict(frame_info), f, indent=2)

        logger.info(f"Tracking complete: {frame_idx} frames, {elapsed:.2f}s")

        return {
            'frame_info': dict(frame_info),
            'trajectories': dict(trajectories),
            'fps': fps,
            'total_frames': frame_idx,
            'elapsed_time': elapsed
        }

    def track_frame(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Dict[str, Any]]:
        if roi:
            x1, y1, x2, y2 = roi
            roi_frame = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            roi_frame = frame
            offset_x, offset_y = 0, 0

        results = self.model.track(
            source=roi_frame,
            stream=False,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker_type,
            device=self.device,
            persist=True,
            verbose=False
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()

                for (cx, cy, w, h), tid, cls in zip(boxes, track_ids, classes):
                    detections.append({
                        'id': int(tid),
                        'cls': int(cls),
                        'cx': float(cx + offset_x),
                        'cy': float(cy + offset_y),
                        'w': float(w),
                        'h': float(h)
                    })

        return detections


def load_frame_info(json_path: str) -> Dict[str, List[Dict]]:
    with open(json_path, 'r') as f:
        return json.load(f)