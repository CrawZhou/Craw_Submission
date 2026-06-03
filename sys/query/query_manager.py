import time
from typing import List, Dict, Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from storage.milvus_storage import MilvusStorage

from storage.inverted_index import InvertedIndex
from clustering.hierarchical_index import HierarchicalClusterIndex
from similarity.PM_TSMD import compute_PM_TSMD
from similarity.CTMD import compute_CTMD_fast
from similarity.I_distance import count_matrix
from utils.math_utils import manhattan_distance
from utils.logger import get_logger

logger = get_logger("QueryManager")


class QueryManager:
    def __init__(
        self,
        milvus_storage,
        inverted_index: InvertedIndex,
        hierarchical_index: HierarchicalClusterIndex = None
    ):
        self.milvus = milvus_storage
        self.inverted_index = inverted_index
        self.hierarchical_index = hierarchical_index

    def get_by_id(self, msu_id: int) -> Optional[Dict[str, Any]]:
        if self.milvus is not None:
            return self.milvus.get_by_id(msu_id)

        if self.hierarchical_index is None or self.hierarchical_index.feature_data_ is None:
            return None

        feature_data = self.hierarchical_index.feature_data_
        if msu_id < 0 or msu_id >= len(feature_data['F']):
            return None

        return {
            'id': msu_id,
            'F_matrix': feature_data['F'][msu_id].tolist(),
            'I_matrix': feature_data['I'][msu_id].tolist(),
            'T_matrix': feature_data['T'][msu_id].tolist(),
            'F_label': int(self.hierarchical_index.F_labels_[msu_id]),
            'I_label': int(self.hierarchical_index.I_labels_[msu_id]),
            'T_label': int(self.hierarchical_index.T_labels_[msu_id]),
        }

    def get_by_label(self, label_type: str, label_value: int) -> List[Dict[str, Any]]:
        if self.milvus is not None:
            return self.milvus.query_by_label(label_type, label_value)

        if self.hierarchical_index is None:
            return []

        labels_map = {
            'T': self.hierarchical_index.T_labels_,
            'I': self.hierarchical_index.I_labels_,
            'F': self.hierarchical_index.F_labels_,
        }

        labels = labels_map.get(label_type)
        if labels is None:
            return []

        result = []
        for i, label in enumerate(labels):
            if label == label_value:
                msu_data = self.get_by_id(i)
                if msu_data:
                    result.append(msu_data)
        return result

    def get_by_labels(
        self,
        t_label: Optional[int] = None,
        i_label: Optional[int] = None,
        f_label: Optional[int] = None
    ) -> List[int]:
        if self.hierarchical_index is None:
            logger.warning("Hierarchical index not initialized")
            return []

        return self.hierarchical_index.query(t_label, i_label, f_label)

    def search_by_element(self, matrix_type: str, value: Any) -> List[int]:
        return self.inverted_index.query(matrix_type, value)

    def find_similar_by_id(self, input_msu_id: int, top_k: int = 5, include_self: bool = True) -> List[Dict[str, Any]]:
        """Find similar MSUs by MSU ID. Matches main/mil_query.py find_similar_by_cluster() logic.

        Uses T→I→F hierarchical filtering: find nearest cluster at each level by distance,
        then compute F distances within the final T+I+F cluster.
        """
        start_time = time.perf_counter()

        # Step 1: Get input MSU data and validate
        input_msu = self.get_by_id(input_msu_id)
        if not input_msu:
            logger.warning(f"MSU {input_msu_id} not found")
            return []

        # Keep F as list of lists (not numpy array) for PM-TSMD which expects varying row lengths
        input_F = input_msu['F_matrix']
        input_I = np.array(input_msu['I_matrix'])
        input_T = np.array(input_msu['T_matrix'])

        # Get cluster representatives
        T_reps = self.hierarchical_index.T_reps_
        I_reps = self.hierarchical_index.I_reps_
        F_reps = self.hierarchical_index.F_reps_
        T_list = self.hierarchical_index.feature_data_['T']
        I_list = self.hierarchical_index.feature_data_['I']

        if not T_reps:
            logger.warning("No T cluster representatives")
            return []

        # ---------- Step 1: Find nearest T cluster (CTMD) ----------
        t_dist_map = {}
        for t_cid, t_rid in T_reps.items():
            rep_T = T_list[t_rid]
            dist = compute_CTMD_fast(input_T, np.array(rep_T))
            t_dist_map[t_cid] = dist

        min_t_dist = min(t_dist_map.values()) if t_dist_map else float("inf")
        target_t_cluster = min(t_dist_map, key=t_dist_map.get)

        # ---------- Step 2: Find nearest I cluster within target T ----------
        t_msus = self.hierarchical_index.query(t_cluster=target_t_cluster)
        if not t_msus:
            logger.warning(f"Target T cluster {target_t_cluster} has no samples")
            return []

        valid_i_reps = {cid: rid for cid, rid in I_reps.items() if rid in t_msus}
        if not valid_i_reps:
            logger.warning(f"No valid I cluster representatives in T cluster {target_t_cluster}")
            return []

        i_dist_map = {}
        for i_cid, i_rid in valid_i_reps.items():
            rep_I = I_list[i_rid]
            v1 = count_matrix(input_I)
            v2 = count_matrix(np.array(rep_I))
            sum1 = v1.sum() or 1
            sum2 = v2.sum() or 1
            dist = manhattan_distance(v1 / sum1, v2 / sum2)
            i_dist_map[i_cid] = dist

        min_i_dist = min(i_dist_map.values()) if i_dist_map else float("inf")
        target_i_cluster = min(i_dist_map, key=i_dist_map.get)

        # ---------- Step 3: Find nearest F cluster within T+I (or skip if no valid F reps) ----------
        ti_msus = self.hierarchical_index.query(
            t_cluster=target_t_cluster,
            i_cluster=target_i_cluster
        )
        if not ti_msus:
            logger.warning(f"Target T+I cluster has no samples")
            return []

        target_f_cluster = None
        valid_f_reps = {cid: rid for cid, rid in F_reps.items() if rid in ti_msus}
        if valid_f_reps:
            f_dist_map = {}
            for f_cid, f_rid in valid_f_reps.items():
                msu_data = self.get_by_id(f_rid)
                if not msu_data:
                    continue
                rep_F = msu_data['F_matrix']  # Keep as list for PM-TSMD
                dist = compute_PM_TSMD(input_F, rep_F)
                f_dist_map[f_cid] = dist

            if f_dist_map:
                target_f_cluster = min(f_dist_map, key=f_dist_map.get)

        # ---------- Step 4: Get all MSUs in target T+I+F cluster (or T+I if no F cluster found) ----------
        if target_f_cluster is not None:
            target_cluster_ids = self.hierarchical_index.query(
                t_cluster=target_t_cluster,
                i_cluster=target_i_cluster,
                f_cluster=target_f_cluster
            )
        else:
            # If no valid F cluster found, use T+I cluster directly (main/mil_query.py logic)
            target_cluster_ids = ti_msus

        if not target_cluster_ids:
            return []

        # Exclude self if include_self=False
        if not include_self:
            target_cluster_ids = [cid for cid in target_cluster_ids if cid != input_msu_id]
        if not target_cluster_ids:
            return []

        # ---------- Step 5: Compute F distances within final cluster ----------
        f_result_list = []
        for msu_id in target_cluster_ids:
            msu_data = self.get_by_id(msu_id)
            if not msu_data:
                continue
            dist = compute_PM_TSMD(input_F, msu_data['F_matrix'])  # Keep as list for PM-TSMD
            f_result_list.append({
                'msu_id': msu_id,
                'msu_data': msu_data,
                'f_distance': dist
            })

        # Sort by F distance (ascending) and return top-k
        f_result_list.sort(key=lambda x: x['f_distance'])
        result = f_result_list[:top_k]

        elapsed = time.perf_counter() - start_time
        logger.info(f"Query done: {elapsed*1000:.2f}ms, {len(result)} results")

        return result

    def find_similar_by_features(
        self,
        F_matrix: List[List],
        I_matrix: List[List],
        T_matrix: List[List],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar MSUs by F/I/T features. Uses T→I→F hierarchical filtering.

        Matches the logic of main/mil_query.py MSUQueryManager.find_similar_by_msu_data().
        """
        start_time = time.perf_counter()

        # Validate input matrices
        if not (self._is_2d_matrix(F_matrix) and
                self._is_2d_matrix(I_matrix) and
                self._is_2d_matrix(T_matrix)):
            logger.warning("F/I/T matrices must be 2D lists")
            return []

        # Prepare input matrices as numpy arrays
        input_F = np.array(F_matrix)
        input_I = np.array(I_matrix)
        input_T = np.array(T_matrix)

        # Get T_max_rows from hierarchical index (set during build)
        t_max_rows = self.hierarchical_index.feature_data_.get('T_max_rows', 1)
        T_list = self.hierarchical_index.feature_data_['T']
        T_reps = self.hierarchical_index.T_reps_
        I_reps = self.hierarchical_index.I_reps_

        if not T_reps:
            logger.warning("No T cluster representatives")
            return []

        # ---------- Step 1: Find nearest T cluster (CTMD) ----------
        t_dist_map = {}
        for t_cid, t_rid in T_reps.items():
            rep_T = T_list[t_rid]
            dist = compute_CTMD_fast(input_T, np.array(rep_T))
            t_dist_map[t_cid] = dist

        if not t_dist_map:
            logger.warning("No valid T cluster distances")
            return []

        target_t_cluster = min(t_dist_map, key=t_dist_map.get)

        # ---------- Step 2: Find nearest I cluster within T (I_distance) ----------
        t_msus = self.hierarchical_index.query(t_cluster=target_t_cluster)
        if not t_msus:
            logger.warning("Target T cluster has no samples")
            return []

        # Filter I_reps to only those in current T cluster
        valid_i_reps = {cid: rid for cid, rid in I_reps.items() if rid in t_msus}
        if not valid_i_reps:
            logger.warning("No valid I cluster representatives in T cluster")
            return []

        I_list = self.hierarchical_index.feature_data_['I']
        i_dist_map = {}
        for i_cid, i_rid in valid_i_reps.items():
            rep_I = I_list[i_rid]
            # I_distance: normalized histogram Manhattan distance
            v1 = count_matrix(input_I)
            v2 = count_matrix(np.array(rep_I))
            sum1 = v1.sum() or 1
            sum2 = v2.sum() or 1
            dist = manhattan_distance(v1 / sum1, v2 / sum2)
            i_dist_map[i_cid] = dist

        target_i_cluster = min(i_dist_map, key=i_dist_map.get)

        # ---------- Step 3: Get T+I cluster members and compute F distances ----------
        ti_msus = self.hierarchical_index.query(
            t_cluster=target_t_cluster,
            i_cluster=target_i_cluster
        )
        if not ti_msus:
            logger.warning("Target T+I cluster has no samples")
            return []

        f_dist_map = {}
        for msu_id in ti_msus:
            msu_data = self.get_by_id(msu_id)
            if not msu_data:
                continue
            dist = compute_PM_TSMD(input_F, msu_data['F_matrix'])  # Keep as list for PM-TSMD
            f_dist_map[msu_id] = dist

        if not f_dist_map:
            logger.warning("Cannot compute F distances in T+I cluster")
            return []

        # Sort by F distance and return top-k
        sorted_msu = sorted(f_dist_map.items(), key=lambda x: x[1])[:top_k]

        results = []
        for msu_id, f_dist in sorted_msu:
            msu_data = self.get_by_id(msu_id)
            if msu_data:
                results.append({
                    'msu_id': msu_id,
                    'msu_data': msu_data,
                    'f_distance': f_dist
                })

        elapsed = time.perf_counter() - start_time
        logger.info(f"find_similar_by_features done: {elapsed*1000:.2f}ms, {len(results)} results")

        return results

    @staticmethod
    def _is_2d_matrix(matrix) -> bool:
        """Check if matrix is a 2D list."""
        if not isinstance(matrix, list) or len(matrix) == 0:
            return False
        return all(isinstance(row, list) for row in matrix)

    def count_by_label(self, label_type: str) -> Dict[int, int]:
        if self.milvus is not None:
            all_data = self.milvus.get_all(limit=10000)
            counts = {}
            for item in all_data:
                label = item[f"{label_type}_label"]
                counts[label] = counts.get(label, 0) + 1
            return counts

        if self.hierarchical_index is None:
            return {}

        labels_map = {
            'T': self.hierarchical_index.T_labels_,
            'I': self.hierarchical_index.I_labels_,
            'F': self.hierarchical_index.F_labels_,
        }

        labels = labels_map.get(label_type)
        if labels is None:
            return {}

        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

        return counts