import math
from typing import List, Tuple, Union
import numpy as np


def euclidean_distance(p1: Union[List, Tuple], p2: Union[List, Tuple]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return 1 - cosine_similarity(vec1, vec2)


def cosine_distance_custom(vec1: np.ndarray, vec2: np.ndarray) -> float:
    min_len = min(len(vec1), len(vec2))
    vec1_cut = vec1[:min_len]
    vec2_cut = vec2[:min_len]

    dot_product = np.dot(vec1_cut, vec2_cut)
    norm1 = np.linalg.norm(vec1_cut)
    norm2 = np.linalg.norm(vec2_cut)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return 1 - (dot_product / (norm1 * norm2))


def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.sum(np.abs(vec1 - vec2))


def vector_angle(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    point_c: Tuple[float, float]
) -> float:
    x1, y1 = point_a[0] - point_b[0], point_a[1] - point_b[1]
    x2, y2 = point_c[0] - point_b[0], point_c[1] - point_b[1]

    len1 = math.sqrt(x1**2 + y1**2)
    len2 = math.sqrt(x2**2 + y2**2)

    if len1 == 0 or len2 == 0:
        return 0.0

    cos_theta = (x1 * x2 + y1 * y2) / (len1 * len2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)

    return math.degrees(math.acos(cos_theta))


def trajectory_direct_distance(trajectory: List[Tuple[float, float]]) -> float:
    if len(trajectory) < 2:
        return 0.0

    return euclidean_distance(trajectory[0], trajectory[-1])