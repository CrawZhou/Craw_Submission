from typing import Dict, List, Tuple
import numpy as np

from config.settings import InteractionConfig
from utils.math_utils import euclidean_distance


class IMatrixBuilder:
    def __init__(self, config: InteractionConfig = None):
        self.config = config or InteractionConfig()

    def build(
        self,
        raw_trajectories: List[List[List]]
    ) -> Tuple[List[np.ndarray], List[List[List]]]:
        I_matrices = []
        interaction_start_points = []

        for msu_trajectories in raw_trajectories:
            I_matrix, inter_starts = self._build_single_msu(msu_trajectories)
            I_matrices.append(I_matrix)
            interaction_start_points.append(inter_starts)

        return I_matrices, interaction_start_points

    def _build_single_msu(
        self,
        trajectories: List[List]
    ) -> Tuple[np.ndarray, List[List[int]]]:
        n_objects = len(trajectories)
        I_matrix = np.zeros((n_objects, n_objects), dtype=int)
        inter_starts = [[-1] * n_objects for _ in range(n_objects)]

        for i in range(n_objects):
            for j in range(n_objects):
                if i == j:
                    continue

                obj1 = trajectories[i]
                obj2 = trajectories[j]

                intersect, start_idx = self._check_intersection(obj1, obj2)

                if intersect:
                    I_matrix[i, j] = 1
                    inter_starts[i][j] = start_idx
                    continue

                relation = self._check_motion_relation(obj1, obj2)
                I_matrix[i, j] = relation

        return I_matrix, inter_starts

    def _check_intersection(self, obj1: List, obj2: List) -> Tuple[bool, int]:
        consecutive_count = 0
        start_index = -1

        # main/I_build.py: for k in range(0, len(obj1), 2)
        # But must use min_len-1 to handle odd-length trajectories (need k+1)
        min_len = min(len(obj1), len(obj2))
        for k in range(0, min_len - 1, 2):
            if obj1[k] != -1 and obj1[k + 1] != -1 and obj2[k] != -1 and obj2[k + 1] != -1:
                if consecutive_count == 0:
                    start_index = k // 2
                consecutive_count += 1
                if consecutive_count >= self.config.consecutive_points:
                    return True, start_index
            else:
                consecutive_count = 0

        return False, -1

    def _check_motion_relation(self, obj1: List, obj2: List) -> int:
        # main/I_build.py: direct indexing of first and last valid coordinates
        # But must check length to avoid index errors
        if len(obj1) < 4 or len(obj2) < 4:
            return 0

        start_distance = euclidean_distance((obj1[0], obj1[1]), (obj2[0], obj2[1]))
        end_distance = euclidean_distance((obj1[-2], obj1[-1]), (obj2[-2], obj2[-1]))

        dist_diff = abs(start_distance - end_distance)

        if dist_diff < self.config.threshold_close:
            return 4
        elif dist_diff < self.config.threshold_far:
            return 2
        else:
            return 3

