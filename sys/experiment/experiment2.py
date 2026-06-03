"""
Evaluates how well MSU segmentation preserves object trajectories across segments.
This measures the global completeness ratio - what fraction of objects have
their full lifespan covered by the MSU segments.

This experiment can use different MSU segmentation methods to compare their effectiveness.
"""
import sys
import os

# Add system/ to sys.path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))  # experiment/
_system_dir = os.path.dirname(_script_dir)  # system/
_root_dir = os.path.dirname(_system_dir)  # parent (yolov13/, contains main/ and system/)

if _system_dir not in sys.path:
    sys.path.insert(0, _system_dir)

os.chdir(_system_dir)

import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from data.frame_parser import FrameParser
from data.msu_splitter import MSUSplitter
from config.settings import Config
from utils.logger import get_logger

logger = get_logger("Experiment2")


def global_completeness_ratio(
    durations: List[Dict[int, Tuple[int, int]]],
    object_frames: Dict[int, List[int]],
    threshold: float = 0.0
) -> float:
    """Calculate the ratio of objects that are fully covered by MSU segments.

    An object is considered "complete" if the fraction of its lifespan that
    falls outside all MSUs is <= threshold.

    Args:
        durations: List of MSU duration maps (object_id -> (start_frame, end_frame))
        object_frames: Map of object_id -> list of frames where object appears
        threshold: Maximum allowed outside ratio (0.0 to 1.0)

    Returns:
        Ratio of complete objects (0.0 to 1.0)
    """
    # Filter to valid objects with frames
    valid_object_frames = {oid: frames for oid, frames in object_frames.items() if frames}

    # Calculate global range for each object
    global_range = {oid: (min(frames), max(frames)) for oid, frames in valid_object_frames.items()}
    total_objs = len(global_range)

    if total_objs == 0:
        return 0.0

    complete_count = 0
    for oid, (g_first, g_last) in global_range.items():
        total_life = g_last - g_first + 1

        # Find best coverage across all MSUs
        best_inside = 0
        for dur_map in durations:
            for msu_first, msu_last in dur_map.values():
                overlap_start = max(g_first, msu_first)
                overlap_end = min(g_last, msu_last)
                inside = max(0, overlap_end - overlap_start + 1)
                best_inside = max(best_inside, inside)

        # Calculate outside ratio
        outside_ratio = 1.0 if total_life == 0 else (1 - (best_inside / total_life))
        if outside_ratio <= threshold:
            complete_count += 1

    return complete_count / total_objs


def filter_short_life_objects(
    object_frames: Dict[int, List[int]],
    min_obj_frames: int
) -> Dict[int, List[int]]:
    """Filter objects that appear in fewer than min_obj_frames frames."""
    filtered = {}
    for oid, frames in object_frames.items():
        unique_frames = len(set(frames))
        if unique_frames >= min_obj_frames:
            filtered[oid] = frames
    return filtered


def load_msu_splits(json_path: str) -> Tuple[List, List]:
    """Load MSU splits from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('msu_ranges', []), data.get('msu_durations', [])


def build_track_duration_map(
    frame_info: Dict[str, List[Dict]],
    start: int,
    end: int
) -> Dict[int, Tuple[int, int]]:
    """Build track duration map for a frame range."""
    track2frames = defaultdict(list)
    for fno in range(start, end + 1):
        for item in frame_info.get(str(fno), []):
            track2frames[item['id']].append(fno)
    return {tid: (frames[0], frames[-1]) for tid, frames in track2frames.items() if frames}


def run_msu_segmentation(frame_info: Dict[str, List[Dict]], config=None) -> Tuple[List, List]:
    """Run MSU segmentation using the configured MSU splitter."""
    if config is None:
        config = Config()
    splitter = MSUSplitter(config.msu if hasattr(config, 'msu') else config)
    msu_ranges, msu_durations = splitter.split(frame_info)
    return msu_ranges, msu_durations


def run_fixed_segmentation(
    frame_info: Dict[str, List[Dict]],
    n_segments: int,
    fps: float = 30.0
) -> Tuple[List, List]:
    """Fixed segmentation - divide video into n equal segments."""
    frame_ids = [int(fid) for fid in frame_info.keys()]
    total_frames = max(frame_ids) + 1 if frame_ids else 0

    if total_frames == 0:
        return [], []

    frames_per_segment = total_frames / n_segments
    msu_ranges = []
    msu_durations = []

    for i in range(n_segments):
        start = int(i * frames_per_segment)
        end = total_frames - 1 if i == n_segments - 1 else int((i + 1) * frames_per_segment) - 1

        msu_ranges.append((start, end))
        duration_map = build_track_duration_map(frame_info, start, end)
        msu_durations.append(duration_map)

    return msu_ranges, msu_durations


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MSU segmentation object completeness ratio"
    )
    parser.add_argument('--frame-info', type=str,
                        default=r'D:\XW2\yolov13\system\data\tmp\frame_info1.json',
                        help='Path to frame_info.json')
    parser.add_argument('--msu-splits', type=str, default=None,
                        help='Path to pre-computed msu_splits.json (optional)')
    parser.add_argument('--method', type=str, choices=['msu', 'fixed'], default='msu',
                        help='Segmentation method: msu (hierarchical) or fixed (equal segments)')
    parser.add_argument('--n-segments', type=int, default=44,
                        help='Number of segments for fixed segmentation')
    parser.add_argument('--min-obj-frames', type=int, default=10,
                        help='Minimum frames for object to be considered')
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='Outside ratio thresholds to evaluate')
    args = parser.parse_args()

    # Load frame_info
    frame_parser = FrameParser()
    frame_parser.load_from_json(args.frame_info)
    frame_info = frame_parser.frame_info

    # Load or compute MSU splits
    if args.msu_splits and args.method == 'msu':
        msu_ranges, msu_durations = load_msu_splits(args.msu_splits)
    elif args.method == 'msu':
        config = Config()
        msu_ranges, msu_durations = run_msu_segmentation(frame_info, config.msu)
    else:
        msu_ranges, msu_durations = run_fixed_segmentation(
            frame_info, args.n_segments, fps=30.0
        )

    # Get object frames
    object_frames = frame_parser.to_object_frame_statistics()
    original_count = len(object_frames)

    # Filter short-life objects
    filtered_object_frames = filter_short_life_objects(object_frames, args.min_obj_frames)
    filtered_count = len(filtered_object_frames)

    # Calculate completeness ratio for each threshold
    print(f"\n=== Object Completeness Ratio Evaluation ===")
    print(f"Method: {args.method}")
    print(f"Original objects: {original_count}")
    print(f"Filtered objects (>= {args.min_obj_frames} frames): {filtered_count}")
    print(f"MSU segments: {len(msu_ranges)}")
    print()
    print(f"{'Threshold':>12} | {'Completeness Ratio':>20}")
    print("-" * 36)

    for th in args.thresholds:
        ratio = global_completeness_ratio(msu_durations, filtered_object_frames, th)
        print(f"{th:>12.1%} | {ratio:>20.2%}")

    return 0


if __name__ == '__main__':
    exit(main())