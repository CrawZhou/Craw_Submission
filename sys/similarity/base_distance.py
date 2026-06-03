import numpy as np
from typing import List, Tuple

from utils.math_utils import euclidean_distance


def trajectory_to_points(trajectory: List) -> List[Tuple[float, float]]:
    points = []

    if isinstance(trajectory[0], (int, float)) and len(trajectory) > 1:
        if isinstance(trajectory[0], (int, float)) and trajectory[0] != -1:
            coords = trajectory[1:]
        else:
            coords = trajectory

        for i in range(0, len(coords) - 1, 2):
            points.append((coords[i], coords[i + 1]))
    else:
        points = trajectory

    return points


def flatten_trajectory_for_distance(trajectory: List) -> np.ndarray:
    points = trajectory_to_points(trajectory)
    valid_points = [(x, y) for x, y in points if x != -1 and y != -1]

    if not valid_points:
        return np.array([])

    return np.array(valid_points).flatten()


def vectors_to_distance_matrix(vectors: List[np.ndarray], distance_func=None) -> np.ndarray:
    n = len(vectors)
    matrix = np.zeros((n, n))

    if distance_func is None:
        from utils.math_utils import cosine_distance_custom
        distance_func = cosine_distance_custom

    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 0
            else:
                d = distance_func(vectors[i], vectors[j])
                matrix[i, j] = d
                matrix[j, i] = d

    return matrix