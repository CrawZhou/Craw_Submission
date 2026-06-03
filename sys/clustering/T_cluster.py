import numpy as np
from typing import List
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from similarity.CTMD import compute_CTMD_fast, pad_T_matrix
from utils.logger import get_logger

logger = get_logger("TCluster")


class TClusterer:
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.representatives_ = {}
        self.distance_matrix_ = None
        self.T_list_ = None

    def fit(self, T_list: List[np.ndarray]) -> 'TClusterer':
        self.T_list_ = T_list
        max_rows = max([t.shape[0] for t in T_list]) if T_list else 0
        T_padded = [pad_T_matrix(t, max_rows) for t in T_list]
        T_flattened = np.array([t.flatten() for t in T_padded])

        def ctmd_metric(X, Y=None):
            if Y is None:
                return pairwise_distances(
                    X,
                    metric=lambda a, b: compute_CTMD_fast(
                        a.reshape(max_rows, 2),
                        b.reshape(max_rows, 2)
                    )
                )
            return np.array([
                compute_CTMD_fast(x.reshape(max_rows, 2), y.reshape(max_rows, 2))
                for x, y in zip(X, Y)
            ])

        self.distance_matrix_ = ctmd_metric(T_flattened)

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

    def predict(self, T: np.ndarray) -> int:
        if self.labels_ is None:
            raise ValueError("fit() must be called first")

        min_dist = float('inf')
        pred_label = 0

        for cluster_id, rep_idx in self.representatives_.items():
            dist = compute_CTMD_fast(T, np.array(self.T_list_[rep_idx]))
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