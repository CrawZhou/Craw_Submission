from .TMD import compute_TMD
from .base_distance import trajectory_to_points


def extract_trajectories(F: list) -> list:
    trajectories = []
    for row in F:
        coords = row[1:] if len(row) > 1 else []
        traj = []
        for i in range(0, len(coords) - 1, 2):
            x, y = coords[i], coords[i + 1]
            traj.append([x, y])
        trajectories.append(traj)
    return trajectories


def compute_TSMD(FA: list, FB: list) -> tuple:
    trajs_A = extract_trajectories(FA)
    trajs_B = extract_trajectories(FB)

    m = len(trajs_A)
    n = len(trajs_B)

    if m == 0 or n == 0:
        return 0.0, 0.0

    import time
    start_time = time.perf_counter()
    total_tmd = 0.0

    for traj_a in trajs_A:
        for traj_b in trajs_B:
            total_tmd += compute_TMD(traj_a, traj_b)

    duration = time.perf_counter() - start_time

    return total_tmd / (m * n), duration


def compute_TSMD_fast(FA: list, FB: list) -> float:
    trajs_A = extract_trajectories(FA)
    trajs_B = extract_trajectories(FB)

    m = len(trajs_A)
    n = len(trajs_B)

    if m == 0 or n == 0:
        return 0.0

    total_tmd = 0.0
    for traj_a in trajs_A:
        for traj_b in trajs_B:
            total_tmd += compute_TMD(traj_a, traj_b)

    return total_tmd / (m * n)