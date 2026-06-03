import math
from typing import List, Tuple, Dict
import numpy as np

from utils.math_utils import (
    euclidean_distance,
    vector_angle,
    trajectory_direct_distance
)


class TrajectoryExtractor:
    def __init__(
        self,
        n_keypoints: int = 6,
        threshold_K: float = 0.10,
        threshold_angle: float = 150.0,
        max_recursion_depth: int = 3
    ):
        self.n_keypoints = n_keypoints
        self.threshold_K = threshold_K
        self.threshold_angle = threshold_angle
        self.max_recursion_depth = max_recursion_depth

    def extract_keypoints_batch(self, trajectories: List[List]) -> List[List]:
        if not trajectories:
            return []

        parsed = []
        for traj in trajectories:
            cls = traj[0]
            coords = traj[1:]
            cleaned = []
            for i in range(0, len(coords) - 1, 2):
                if coords[i] != -1 and coords[i + 1] != -1:
                    cleaned.append((coords[i], coords[i + 1]))

            if cleaned:
                parsed.append({'cls': cls, 'trajectory': cleaned})
            else:
                parsed.append({'cls': cls, 'trajectory': []})

        keypoints_list = []
        all_trajectories = []

        for item in parsed:
            if not item['trajectory']:
                keypoints_list.append([item['cls']])
                all_trajectories.append([])
                continue

            trajectory = item['trajectory']
            all_trajectories.append(trajectory)
            keypoints = self._analyze_trajectory(trajectory)
            keypoints_list.append([item['cls']] + keypoints)

        keypoints_list = self._unify_length(keypoints_list, all_trajectories)

        result = []
        for kp in keypoints_list:
            cls = kp[0]
            formatted = [cls]
            for point in kp[1:]:
                formatted.append(point[0])
                formatted.append(point[1])
            result.append(formatted)

        return result

    def _analyze_trajectory(self, trajectory: List[Tuple[float, float]]) -> List[Tuple]:
        if len(trajectory) < self.n_keypoints:
            # Return with index appended to each point so caller always gets 3-element tuples
            return [(t[0], t[1], i) for i, t in enumerate(trajectory)]

        initial_points = self._get_equidistant_points(trajectory)
        if len(initial_points) < 3:
            return trajectory

        key_points = set()

        for i in range(len(initial_points) - 2):
            p1, p2, p3 = initial_points[i], initial_points[i + 1], initial_points[i + 2]
            result = self._analyze_triangle(
                p1, p2, p3, trajectory,
                self.threshold_K, self.threshold_angle,
                self.max_recursion_depth
            )
            for point in result:
                key_points.add(point)

        sorted_points = sorted(key_points, key=lambda x: x[2] if len(x) > 2 else 0)

        return sorted_points

    def _get_equidistant_points(
        self,
        trajectory: List[Tuple[float, float]]
    ) -> List[Tuple]:
        n = len(trajectory)
        if n < self.n_keypoints:
            return [(t[0], t[1], i) for i, t in enumerate(trajectory)]

        indices = [int(i * (n - 1) / (self.n_keypoints - 1)) for i in range(self.n_keypoints)]
        return [(trajectory[idx][0], trajectory[idx][1], idx) for idx in indices]

    def _analyze_triangle(
        self,
        p1: Tuple,
        p2: Tuple,
        p3: Tuple,
        trajectory: List[Tuple],
        threshold_K: float,
        threshold_angle: float,
        max_depth: int
    ) -> List[Tuple]:
        d_13 = euclidean_distance(p1, p3)
        d_total = trajectory_direct_distance(trajectory)

        if d_total == 0:
            return [p1, p2, p3]

        K = d_13 / d_total

        if K <= threshold_K:
            return [p1, p2, p3]

        angle = vector_angle(p1, p2, p3)
        if angle >= threshold_angle:
            return [p1, p2, p3]

        d_12 = euclidean_distance(p1, p2)
        d_23 = euclidean_distance(p2, p3)

        if d_12 == d_23:
            return [p1, p2, p3]

        if max_depth <= 0:
            return [p1, p2, p3]

        if d_12 > d_23:
            mid_idx = (int(p1[2]) + int(p2[2])) // 2
            new_p2 = (trajectory[mid_idx][0], trajectory[mid_idx][1], mid_idx)
            return self._analyze_triangle(
                p1, new_p2, p3, trajectory,
                threshold_K, threshold_angle, max_depth - 1
            )
        else:
            mid_idx = (int(p2[2]) + int(p3[2])) // 2
            new_p2 = (trajectory[mid_idx][0], trajectory[mid_idx][1], mid_idx)
            return self._analyze_triangle(
                p1, new_p2, p3, trajectory,
                threshold_K, threshold_angle, max_depth - 1
            )

    def _unify_length(
        self,
        keypoints_list: List[List],
        trajectories: List[List[Tuple]]
    ) -> List[List]:
        lengths = []
        for kp in keypoints_list:
            lengths.append(len(kp) - 1)

        if not lengths:
            return keypoints_list

        max_length = max(lengths)

        if min(lengths) == max_length:
            return keypoints_list

        for i in range(len(keypoints_list)):
            current_length = lengths[i]
            if current_length >= max_length:
                continue

            to_add = max_length - current_length

            kp = keypoints_list[i]
            trajectory = trajectories[i]

            if not trajectory:
                continue

            pure_kp = kp[1:]

            for _ in range(to_add):
                if len(pure_kp) < 2:
                    break

                max_dist = 0
                max_idx = 0

                for m in range(len(pure_kp) - 1):
                    p1 = (pure_kp[m][0], pure_kp[m][1])
                    p2 = (pure_kp[m + 1][0], pure_kp[m + 1][1])
                    dist = euclidean_distance(p1, p2)

                    if dist > max_dist:
                        max_dist = dist
                        max_idx = m

                # Guard against out-of-bounds
                if max_idx >= len(pure_kp) - 1:
                    max_idx = len(pure_kp) - 2

                start_idx = int(pure_kp[max_idx][2])
                end_idx = int(pure_kp[max_idx + 1][2])
                center_idx = (start_idx + end_idx) // 2

                if 0 <= center_idx < len(trajectory):
                    center_point = (
                        trajectory[center_idx][0],
                        trajectory[center_idx][1],
                        center_idx
                    )
                else:
                    center_point = pure_kp[max_idx]

                pure_kp.insert(max_idx + 1, center_point)

            keypoints_list[i] = [kp[0]] + pure_kp

        return keypoints_list