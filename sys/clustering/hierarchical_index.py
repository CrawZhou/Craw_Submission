from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

from .T_cluster import TClusterer
from .I_cluster import IClusterer
from .F_cluster import FClusterer
from similarity.CTMD import pad_T_matrix
from config.settings import ClusteringConfig
from utils.logger import get_logger

logger = get_logger("HierarchicalIndex")


class HierarchicalClusterIndex:
    def __init__(self, config: ClusteringConfig = None):
        self.config = config or ClusteringConfig()

        self.T_labels_ = None
        self.I_labels_ = None
        self.F_labels_ = None

        self.T_reps_ = {}
        self.I_reps_ = {}
        self.F_reps_ = {}

        self.index_ = None
        self.feature_data_ = None

    def build(self, F: List, I: List, T: List) -> 'HierarchicalClusterIndex':
        n_samples = len(F)

        if not (len(F) == len(I) == len(T) == n_samples):
            raise ValueError("F/I/T matrix count mismatch")

        T_raw = [np.array(t) for t in T]
        max_rows = max([t.shape[0] for t in T_raw]) if T_raw else 0
        T_padded = [pad_T_matrix(t, max_rows) for t in T_raw]
        T_flattened = np.array([t.flatten() for t in T_padded])

        self.feature_data_ = {
            # Keep F as list of lists (not numpy array) since row lengths are inconsistent
            'F': list(F),
            'I': [np.array(i) for i in I],
            'T': T_raw,
            'T_flattened': T_flattened,
            'T_max_rows': max_rows
        }

        logger.info(f"Building hierarchical index: {n_samples} samples")

        # T clustering
        T_clusterer = TClusterer(n_clusters=self.config.n_T_clusters)
        T_clusterer.fit([np.array(t) for t in T])
        self.T_labels_ = T_clusterer.labels_
        self.T_reps_ = T_clusterer.representatives_

        # I clustering
        I_clusterer = IClusterer(n_clusters=self.config.n_I_clusters)
        I_clusterer.fit([np.array(i) for i in I])
        self.I_labels_ = I_clusterer.labels_
        self.I_reps_ = I_clusterer.representatives_

        # F clustering
        F_clusterer = FClusterer(n_clusters=self.config.n_F_clusters)
        F_clusterer.fit(F)
        self.F_labels_ = F_clusterer.labels_
        self.F_reps_ = F_clusterer.representatives_

        self._build_index(n_samples)

        logger.info("Hierarchical index built")

        return self

    def _build_index(self, n_samples: int):
        self.index_ = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for msu_id in range(n_samples):
            t_label = self.T_labels_[msu_id]
            i_label = self.I_labels_[msu_id]
            f_label = self.F_labels_[msu_id]

            self.index_[t_label][i_label][f_label].append(msu_id)

        self.index_ = self._dict_to_regular(self.index_)

    def _dict_to_regular(self, d):
        if isinstance(d, defaultdict):
            d = dict(d)
        if isinstance(d, dict):
            for k, v in d.items():
                d[k] = self._dict_to_regular(v)
        return d

    def query(
        self,
        t_cluster: Optional[int] = None,
        i_cluster: Optional[int] = None,
        f_cluster: Optional[int] = None
    ) -> List[int]:
        if self.index_ is None:
            return []

        result = []

        if t_cluster is not None:
            t_level = {t_cluster: self.index_.get(t_cluster, {})}
        else:
            t_level = self.index_

        for t_c, i_level in t_level.items():
            if i_cluster is not None:
                i_level = {i_cluster: i_level.get(i_cluster, {})}

            for i_c, f_level in i_level.items():
                if f_cluster is not None:
                    f_level = {f_cluster: f_level.get(f_cluster, [])}

                for f_c, msu_ids in f_level.items():
                    result.extend(msu_ids)

        return list(set(result))

    def get_labels(self, msu_id: int) -> Dict[str, int]:
        """Get cluster labels for an MSU. Mirrors main HierarchicalClusterManager.get_cluster_labels()."""
        if msu_id < 0 or msu_id >= len(self.T_labels_):
            return {}

        return {
            'T_label': int(self.T_labels_[msu_id]),
            'I_label': int(self.I_labels_[msu_id]),
            'F_label': int(self.F_labels_[msu_id])
        }

    def get_representatives(self) -> Dict[str, Dict]:
        """Get cluster representatives. Mirrors main HierarchicalClusterManager.get_cluster_representatives()."""
        return {
            'T_reps': self.T_reps_,
            'I_reps': self.I_reps_,
            'F_reps': self.F_reps_
        }

    def query_by_cluster(self, t_cluster: Optional[int] = None, i_cluster: Optional[int] = None, f_cluster: Optional[int] = None) -> List[int]:
        """Query MSU IDs by cluster labels. Mirrors main HierarchicalClusterManager.query_by_cluster()."""
        if self.index_ is None:
            return []

        result = []
        t_level = self.index_

        if t_cluster is not None:
            if t_cluster not in t_level:
                return result
            t_level = {t_cluster: t_level[t_cluster]}

        for t_c, i_level in t_level.items():
            if i_cluster is not None:
                if i_cluster not in i_level:
                    continue
                i_level = {i_cluster: i_level[i_cluster]}

            for i_c, f_level in i_level.items():
                if f_cluster is not None:
                    if f_cluster not in f_level:
                        continue
                    f_level = {f_cluster: f_level[f_cluster]}

                for f_c, msu_ids in f_level.items():
                    result.extend(msu_ids)

        return list(set(result))

    def print_statistics(self):
        if self.index_ is None:
            print("Index not built")
            return

        print("\n" + "=" * 60)
        print("Hierarchical Clustering Index Statistics")
        print("=" * 60)

        print("\nRepresentatives:")
        print(f"  T: {self.T_reps_}")
        print(f"  I: {self.I_reps_}")
        print(f"  F: {self.F_reps_}")

        print("\nIndex structure (T->I->F):")
        for t_c in sorted(self.index_.keys()):
            t_level = self.index_[t_c]
            t_count = sum(
                len(f_ids)
                for i_level in t_level.values()
                for f_ids in i_level.values()
            )
            print(f"\n  T-{t_c} (total: {t_count})")

            for i_c in sorted(t_level.keys()):
                i_level = t_level[i_c]
                i_count = sum(len(f_ids) for f_ids in i_level.values())
                print(f"    |- I-{i_c} ({i_count})")

                for f_c in sorted(i_level.keys()):
                    f_ids = i_level[f_c]
                    print(f"    |  |- F-{f_c}: {len(f_ids)} IDs: {f_ids}")