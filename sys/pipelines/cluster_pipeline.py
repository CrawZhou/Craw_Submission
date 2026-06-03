from typing import List, Dict, Tuple

from clustering.hierarchical_index import HierarchicalClusterIndex
from storage.milvus_storage import MilvusStorage
from storage.inverted_index import InvertedIndex
from query.query_manager import QueryManager
from config.settings import Config
from utils.logger import get_logger

logger = get_logger("ClusterPipeline")


class ClusterPipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.cluster_config = self.config.clustering

        self.hierarchical_index = None
        self.milvus_storage = None
        self.inverted_index = None
        self.query_manager = None

    def run_clustering(self, F: List, I: List, T: List) -> HierarchicalClusterIndex:
        logger.info("Starting clustering")

        self.hierarchical_index = HierarchicalClusterIndex(self.cluster_config)
        self.hierarchical_index.build(F, I, T)

        self.hierarchical_index.print_statistics()

        return self.hierarchical_index

    def run_storage(
        self,
        F: List,
        I: List,
        T: List,
        db_path: str = "./msu_milvus_db"
    ) -> Tuple[MilvusStorage, InvertedIndex]:
        if self.hierarchical_index is None:
            raise ValueError("Clustering must be run first")

        logger.info("Starting storage")

        self.milvus_storage = MilvusStorage(
            db_path=db_path,
            collection_name="matrix_collection",
            vector_dim=128
        )

        self.inverted_index = InvertedIndex(
            db_path=f"{db_path}_inverted.json"
        )

        T_labels = self.hierarchical_index.T_labels_
        I_labels = self.hierarchical_index.I_labels_
        F_labels = self.hierarchical_index.F_labels_

        self.milvus_storage.batch_insert(
            F_list=F,
            I_list=I,
            T_list=T,
            T_labels=T_labels.tolist() if hasattr(T_labels, 'tolist') else list(T_labels),
            I_labels=I_labels.tolist() if hasattr(I_labels, 'tolist') else list(I_labels),
            F_labels=F_labels.tolist() if hasattr(F_labels, 'tolist') else list(F_labels)
        )

        for i in range(len(F)):
            self.inverted_index.update(i, F[i], I[i])

        self.inverted_index.save()

        logger.info(f"Storage done: {len(F)} MSUs")

        return self.milvus_storage, self.inverted_index

    def run_full_pipeline(
        self,
        F: List,
        I: List,
        T: List,
        db_path: str = "./msu_milvus_db"
    ) -> QueryManager:
        self.run_clustering(F, I, T)
        self.run_storage(F, I, T, db_path)

        self.query_manager = QueryManager(
            milvus_storage=self.milvus_storage,
            inverted_index=self.inverted_index,
            hierarchical_index=self.hierarchical_index
        )

        logger.info("Cluster and storage pipeline done")

        return self.query_manager

    def get_query_manager(self) -> QueryManager:
        if self.query_manager is None:
            raise ValueError("Pipeline must be run first")
        return self.query_manager

    def close(self):
        if self.milvus_storage:
            self.milvus_storage.close()
        if self.inverted_index:
            self.inverted_index.close()