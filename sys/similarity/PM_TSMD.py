import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

from .TMD import compute_TMD
from utils.math_utils import euclidean_distance


def parse_trajectory(row) -> Tuple[int, List[List[float]]]:
    class_id = int(row[0])
    coords = row[1:]

    if len(coords) % 2 != 0:
        coords = coords[:-1]

    trajectory = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
    return class_id, trajectory


def group_by_class(F: List) -> Dict[int, List[List]]:
    groups = defaultdict(list)
    for row in F:
        class_id, traj = parse_trajectory(row)
        groups[class_id].append(traj)
    return groups


def match_trajectories(
    group1: List[List],
    group2: List[List],
    w1: float = 0.5,
    w2: float = 0.5
) -> List[Tuple[int, int]]:
    matched = []
    used_indices = set()

    for i, traj1 in enumerate(group1):
        if len(traj1) < 2:
            continue

        best_j = None
        best_score = float('inf')

        start1, end1 = traj1[0], traj1[-1]

        for j, traj2 in enumerate(group2):
            if j in used_indices or len(traj2) < 2:
                continue

            start2, end2 = traj2[0], traj2[-1]

            dist_start = euclidean_distance(start1, start2)
            dist_end = euclidean_distance(end1, end2)
            score = w1 * dist_start + w2 * dist_end

            if score < best_score:
                best_score = score
                best_j = j

        if best_j is not None:
            matched.append((i, best_j))
            used_indices.add(best_j)

    return matched


def compute_PM_TSMD(FA: List, FB: List, base_penalty: float = 1.0) -> float:
    groups_A = group_by_class(FA)
    groups_B = group_by_class(FB)

    total_distance = 0.0
    total_pairs = 0

    for class_id in groups_A:
        if class_id not in groups_B:
            total_distance += base_penalty * len(groups_A[class_id])
            continue

        group_A = groups_A[class_id]
        group_B = groups_B[class_id]

        matched = match_trajectories(group_A, group_B)

        max_count = max(len(group_A), len(group_B))
        match_rate = len(matched) / max_count if max_count > 0 else 0.0
        penalty_weight = 1.0 - match_rate

        for i, j in matched:
            dist = compute_TMD(group_A[i], group_B[j])
            total_distance += dist
            total_pairs += 1

        unmatched_A = len(group_A) - len(matched)
        unmatched_B = len(group_B) - len(matched)
        total_distance += base_penalty * penalty_weight * (unmatched_A + unmatched_B)

    return max(0.0, total_distance / max(total_pairs, 1))