from typing import List, Dict, Any, Optional

from query.query_manager import QueryManager
from utils.logger import get_logger

logger = get_logger("QueryPipeline")


class QueryPipeline:
    def __init__(self, query_manager: QueryManager):
        self.query_manager = query_manager

    def find_similar_videos(self, msu_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.query_manager.find_similar_by_id(msu_id, top_k)
        logger.info(f"Found {len(results)} similar MSUs")
        return results

    def find_similar_by_features(
        self,
        F_matrix: List[List],
        I_matrix: List[List],
        T_matrix: List[List],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        results = self.query_manager.find_similar_by_features(
            F_matrix, I_matrix, T_matrix, top_k
        )
        logger.info(f"Found {len(results)} similar MSUs")
        return results

    def find_by_trajectory_pattern(self, cls: int) -> List[int]:
        return self.query_manager.search_by_element('F', cls)

    def find_by_interaction_type(self, interaction_value: int) -> List[int]:
        return self.query_manager.search_by_element('I', interaction_value)

    def get_cluster_info(
        self,
        t_label: Optional[int] = None,
        i_label: Optional[int] = None,
        f_label: Optional[int] = None
    ) -> Dict[str, Any]:
        msu_ids = self.query_manager.get_by_labels(t_label, i_label, f_label)

        if not msu_ids:
            return {'count': 0, 'msu_ids': []}

        samples = []
        for msu_id in msu_ids[:10]:
            data = self.query_manager.get_by_id(msu_id)
            if data:
                samples.append(data)

        return {
            'count': len(msu_ids),
            'msu_ids': msu_ids,
            'samples': samples,
            'labels': {'T': t_label, 'I': i_label, 'F': f_label}
        }

    def get_statistics(self) -> Dict[str, Any]:
        total_count = self.query_manager.milvus.count()

        t_counts = self.query_manager.count_by_label('T')
        i_counts = self.query_manager.count_by_label('I')
        f_counts = self.query_manager.count_by_label('F')

        index_stats = self.query_manager.inverted_index.get_stats()

        return {
            'total_msu': total_count,
            'T_cluster_distribution': t_counts,
            'I_cluster_distribution': i_counts,
            'F_cluster_distribution': f_counts,
            'inverted_index_stats': index_stats
        }