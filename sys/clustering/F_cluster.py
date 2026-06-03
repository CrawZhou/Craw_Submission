import numpy as np
from typing import List
from sklearn.cluster import AgglomerativeClustering

from similarity.PM_TSMD import compute_PM_TSMD
from utils.logger import get_logger

logger = get_logger("FCluster")


class FClusterer:
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.representatives_ = {}
        self.distance_matrix_ = None
        self.F_list_ = None

    def fit(self, F_list: List) -> 'FClusterer':
        self.F_list_ = F_list
        n = len(F_list)

        self.distance_matrix_ = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    continue
                dist = compute_PM_TSMD(F_list[i], F_list[j])
                self.distance_matrix_[i, j] = dist
                self.distance_matrix_[j, i] = dist

        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric='precomputed',
            linkage='average'
        )
        self.labels_ = clustering.fit_predict(self.distance_matrix_)
        self._compute_representatives()

        return self

    def _compute_representatives(self):
        for cluster_id in np.unique(self.labels_):
            indices = np.where(self.labels_ == cluster_id)[0]

            if len(indices) == 1:
                self.representatives_[cluster_id] = indices[0]
                continue

            avg_distances = []
            for idx in indices:
                dist_sum = np.sum(self.distance_matrix_[idx, indices])
                avg_dist = dist_sum / (len(indices) - 1)
                avg_distances.append((idx, avg_dist))

            avg_distances.sort(key=lambda x: x[1])
            self.representatives_[cluster_id] = avg_distances[0][0]

    def predict(self, F: np.ndarray) -> int:
        if self.labels_ is None:
            raise ValueError("fit() must be called first")

        min_dist = float('inf')
        pred_label = 0

        for cluster_id, rep_idx in self.representatives_.items():
            dist = compute_PM_TSMD(F, self.F_list_[rep_idx])
            if dist < min_dist:
                min_dist = dist
                pred_label = cluster_id

        return pred_label

    def get_cluster_members(self, cluster_id: int) -> List[int]:
        if self.labels_ is None:
            return []
        return np.where(self.labels_ == cluster_id)[0].tolist()

    def get_cluster_stats(self) -> dict:
        if self.labels_ is None:
            return {}

        stats = {}
        for cluster_id in np.unique(self.labels_):
            members = self.get_cluster_members(cluster_id)
            stats[cluster_id] = {
                'size': len(members),
                'members': members,
                'representative': self.representatives_.get(cluster_id)
            }

        return stats