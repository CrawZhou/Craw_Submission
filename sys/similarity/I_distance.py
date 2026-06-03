import numpy as np
from utils.math_utils import manhattan_distance


def count_matrix(I):
    """Count occurrences of interaction types 0-4 in I matrix. Uses full matrix like main/I_cluster."""
    V = np.zeros((5,), dtype=int)
    for val in I.flatten():
        if 0 <= val <= 4:
            V[int(val)] += 1
    return V


def compute_I_distance(IA: np.ndarray, IB: np.ndarray) -> float:
    """Compute I distance using normalized histogram Manhattan distance. Matches main/I_cluster.matrix_analysis."""
    V1 = count_matrix(IA)
    V2 = count_matrix(IB)

    sum1 = V1.sum()
    sum2 = V2.sum()

    if sum1 == 0:
        sum1 = 1
    if sum2 == 0:
        sum2 = 1

    v1 = np.round(V1 / sum1, 2)
    v2 = np.round(V2 / sum2, 2)

    return round(np.sum(np.abs(v1 - v2)), 3)


def compute_I_distance_fast(IA: np.ndarray, IB: np.ndarray) -> float:
    """Compute I distance without rounding. Matches main/I_cluster.matrix_analysis pattern."""
    V1 = count_matrix(IA)
    V2 = count_matrix(IB)

    sum1 = V1.sum()
    sum2 = V2.sum()

    if sum1 == 0:
        sum1 = 1
    if sum2 == 0:
        sum2 = 1

    v1 = V1 / sum1
    v2 = V2 / sum2

    return manhattan_distance(v1, v2)


def compute_I_distance_matrix(I_list: list) -> np.ndarray:
    n = len(I_list)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 0
            else:
                d = compute_I_distance_fast(I_list[i], I_list[j])
                matrix[i, j] = d
                matrix[j, i] = d

    return matrix