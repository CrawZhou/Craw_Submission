from typing import Dict, List, Tuple


class TMatrixBuilder:
    def build(
        self,
        msu_ranges: List[Tuple[int, int]],
        msu_durations: List[Dict[int, Tuple[int, int]]]
    ) -> List[List[List[float]]]:
        T_matrices = []

        for i, (start, end) in enumerate(msu_ranges):
            total_frames = end - start + 1
            duration_map = msu_durations[i]

            T_matrix = []

            for obj_id, (first, last) in duration_map.items():
                start_relative = first - start
                end_relative = last - start

                start_norm = round(max(0, min(1, start_relative / total_frames)), 2)
                end_norm = round(max(0, min(1, end_relative / total_frames)), 2)

                T_matrix.append([start_norm, end_norm])

            if not T_matrix:
                T_matrix = [[0, 1]]

            T_matrices.append(T_matrix)

        return T_matrices