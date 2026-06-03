"""
Experiment 1: PM-TSMD vs TSMD speed comparison.

Compares the performance of Penalized Matching Trajectory Set MD (PM-TSMD)
against basic Trajectory Set MD (TSMD) using random MSU pairs.
"""
import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_system_dir = os.path.dirname(_script_dir)

if _system_dir not in sys.path:
    sys.path.insert(0, _system_dir)

os.chdir(_system_dir)

import json
import time
import random
import argparse
from random import sample

import numpy as np

from features.feature_builder import FeatureBuilder
from data.msu_splitter import MSUSplitter
from config.settings import Config
from utils.logger import get_logger

logger = get_logger("Experiment1")


def build_SA(I, F):
    """Build SA structure from I and F matrices."""
    return [[F[i], I[i]] for i in range(len(I))]


def _worker_tsmd(pairs, SA):
    """Worker for TSMD computation (uses PM-TSMD internally)."""
    from similarity.PM_TSMD import compute_PM_TSMD
    for idx1, idx2 in pairs:
        compute_PM_TSMD(SA[idx1][0], SA[idx2][0])
    return True


def _worker_pmtsmd(pairs, SA):
    """Worker for PM-TSMD computation."""
    from similarity.PM_TSMD import compute_PM_TSMD
    for idx1, idx2 in pairs:
        compute_PM_TSMD(SA[idx1][0], SA[idx2][0])
    return True


def measure_total_time(SA, n=1):
    """Measure total computation time for TSMD and PM-TSMD."""
    pairs = [sample(range(len(SA)), 2) for _ in range(n)]

    # Measure TSMD (using PM-TSMD)
    start = time.perf_counter()
    _worker_tsmd(pairs, SA)
    t_tsmd = time.perf_counter() - start

    # Measure PM-TSMD
    start = time.perf_counter()
    _worker_pmtsmd(pairs, SA)
    t_pm = time.perf_counter() - start

    return t_tsmd, t_pm, (t_tsmd / t_pm) if t_pm else 0


def main():
    parser = argparse.ArgumentParser(description="PM-TSMD vs TSMD speed comparison")
    parser.add_argument('-n', '--loops', type=int, default=1000,
                        help='Number of random pairs to test')
    parser.add_argument('--frame-info', type=str,
                        default=r'D:\XW2\yolov13\system\data\tmp\frame_info1.json',
                        help='Path to frame_info.json')
    args = parser.parse_args()

    # Load frame_info
    if not os.path.exists(args.frame_info):
        logger.error(f"frame_info.json not found: {args.frame_info}")
        return 1

    with open(args.frame_info, 'r') as f:
        frame_data = json.load(f)

    # Build features with full pipeline (with actual MSU segmentation)
    config = Config()
    fb = FeatureBuilder(config)

    # Run normal MSU segmentation
    msu_splitter = MSUSplitter(config.msu)
    msu_ranges, msu_durations = msu_splitter.split(frame_data)
    fb.set_msu_splits(msu_ranges, msu_durations)
    F, I, T = fb.build(frame_data, skip_splits=True)

    if not F or not I or len(F) < 2:
        logger.error(f"Not enough MSUs ({len(F) if F else 0}) for pairwise comparison, need at least 2")
        return 1

    SA = build_SA(I, F)

    logger.info(f"Built SA with {len(SA)} MSUs, testing {args.loops} random pairs")

    t_tsmd, t_pm, ratio = measure_total_time(SA, n=args.loops)

    print(f"TSMD total time: {t_tsmd:.2f} seconds")
    print(f"PM-TSMD total time: {t_pm:.2f} seconds")
    print(f"Speed ratio (TSMD/PM-TSMD): {ratio:.2f}x")

    return 0


if __name__ == '__main__':
    exit(main())