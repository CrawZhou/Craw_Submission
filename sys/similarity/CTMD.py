import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple


def pad_T_matrix(T: np.ndarray, target_rows: int) -> np.ndarray:
    padded = np.zeros((target_rows, 2))
    padded[:T.shape[0], :] = T

    if T.shape[0] < target_rows:
        padded[T.shape[0]:, 0] = 0
        padded[T.shape[0]:, 1] = 1

    return padded


def compute_emd_1d(d1: np.ndarray, d2: np.ndarray) -> float:
    cost_matrix = np.abs(d1[:, None] - d2[None, :])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].sum()


def compute_CTMD(TA: np.ndarray, TB: np.ndarray) -> Tuple[float, float]:
    import time
    start = time.perf_counter()

    c1 = TA.shape[0]
    c2 = TB.shape[0]
    c_max = max(c1, c2)

    TA_padded = pad_T_matrix(TA, c_max)
    TB_padded = pad_T_matrix(TB, c_max)

    Du1, Dv1 = TA_padded[:, 0], TA_padded[:, 1]
    Du2, Dv2 = TB_padded[:, 0], TB_padded[:, 1]

    emd_u = compute_emd_1d(Du1, Du2)
    emd_v = compute_emd_1d(Dv1, Dv2)

    L = 1.0 / c_max
    ctmd = L * (emd_u + emd_v) / 2.0

    duration = time.perf_counter() - start

    return ctmd, duration


def compute_CTMD_fast(TA: np.ndarray, TB: np.ndarray) -> float:
    c1 = TA.shape[0]
    c2 = TB.shape[0]
    c_max = max(c1, c2)

    TA_padded = pad_T_matrix(TA, c_max)
    TB_padded = pad_T_matrix(TB, c_max)

    Du1, Dv1 = TA_padded[:, 0], TA_padded[:, 1]
    Du2, Dv2 = TB_padded[:, 0], TB_padded[:, 1]

    emd_u = compute_emd_1d(Du1, Du2)
    emd_v = compute_emd_1d(Dv1, Dv2)

    L = 1.0 / c_max
    return L * (emd_u + emd_v) / 2.0