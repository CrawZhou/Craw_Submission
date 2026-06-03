from typing import List, Dict, Any, Tuple
import numpy as np


def validate_frame_info(data: Dict[str, List[Dict]]) -> bool:
    required_keys = {'id', 'cls', 'cx', 'cy', 'w', 'h'}

    for frame_id, detections in data.items():
        if not isinstance(frame_id, str):
            raise ValueError(f"Frame ID must be string, got {type(frame_id)}")

        if not isinstance(detections, list):
            raise ValueError(f"Detections for frame {frame_id} must be list")

        for det in detections:
            if not isinstance(det, dict):
                raise ValueError(f"Detection must be dict, got {type(det)}")

            missing_keys = required_keys - set(det.keys())
            if missing_keys:
                raise ValueError(f"Missing keys in detection: {missing_keys}")

    return True


def validate_trajectory(trajectory: List) -> bool:
    if not isinstance(trajectory, list):
        return False
    if len(trajectory) < 3:
        return False
    return True


def validate_matrix_2d(matrix: Any) -> bool:
    if not isinstance(matrix, list):
        return False
    if len(matrix) == 0:
        return False
    if not all(isinstance(row, list) for row in matrix):
        return False
    return True


def validate_msu_tuple(msu_tuple: Tuple) -> bool:
    if not isinstance(msu_tuple, tuple) or len(msu_tuple) != 2:
        return False
    start, end = msu_tuple
    if not (isinstance(start, int) and isinstance(end, int)):
        return False
    if start < 0 or end < 0:
        return False
    if start > end:
        return False
    return True


def validate_feature_matrices(
    F: List[List],
    I: List[List],
    T: List[List]
) -> bool:
    if len(F) != len(I) or len(F) != len(T):
        raise ValueError(f"Feature matrix count mismatch: F={len(F)}, I={len(I)}, T={len(T)}")

    for i, (f_mat, i_mat, t_mat) in enumerate(zip(F, I, T)):
        if not validate_matrix_2d(f_mat):
            raise ValueError(f"F_matrix[{i}] is not a valid 2D matrix")
        if not validate_matrix_2d(i_mat):
            raise ValueError(f"I_matrix[{i}] is not a valid 2D matrix")
        if not validate_matrix_2d(t_mat):
            raise ValueError(f"T_matrix[{i}] is not a valid 2D matrix")

    return True