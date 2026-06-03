import numpy as np

from .base_distance import flatten_trajectory_for_distance
from utils.math_utils import cosine_distance_custom


def compute_TMD(traj1: list, traj2: list) -> float:
    """Compute Trajectory Matching Distance. Matches main/F_cluster.py TMD logic.

    Note: Unlike system/similarity/base_distance.py flatten_trajectory_for_distance,
    this does NOT filter -1 values - it flattens the raw trajectory directly
    to match main/F_cluster.py behavior.
    """
    vec1 = np.array(traj1).flatten()
    vec2 = np.array(traj2).flatten()
    return cosine_distance_custom(vec1, vec2)


def compute_batch_TMD(trajectories1: list, trajectories2: list) -> np.ndarray:
    n1 = len(trajectories1)
    n2 = len(trajectories2)
    matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            matrix[i, j] = compute_TMD(trajectories1[i], trajectories2[j])

    return matrix