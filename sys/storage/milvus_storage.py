import json
import os
from typing import List, Dict, Optional, Any
import numpy as np
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

from config.settings import StorageConfig
from utils.logger import get_logger

logger = get_logger("MilvusStorage")


class MilvusStorage:
    def __init__(
        self,
        db_path: str = "./msu_milvus_db",
        collection_name: str = "matrix_collection",
        vector_dim: int = 128
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.collection = None

        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._connect()
        self._create_collection()

    def _connect(self):
        if "default" in connections.list_connections():
            current_uri = connections.get_connection_addr("default")["uri"]
            if current_uri == f"{self.db_path}.db":
                return
            connections.disconnect("default")

        connections.connect("default", uri=f"{self.db_path}.db")

    def _create_collection(self):
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(f"Collection '{self.collection_name}' loaded")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="placeholder_vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name="F_matrix", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="I_matrix", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="T_matrix", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="T_label", dtype=DataType.INT64),
            FieldSchema(name="I_label", dtype=DataType.INT64),
            FieldSchema(name="F_label", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(fields=fields, description="MSU Feature Matrix Storage")
        self.collection = Collection(self.collection_name, schema)

        self.collection.create_index(
            "placeholder_vector",
            {"index_type": "FLAT", "params": {}, "metric_type": "L2"}
        )
        self.collection.load()
        logger.info(f"Collection '{self.collection_name}' created")

    def insert(
        self,
        msu_id: int,
        F_matrix: List,
        I_matrix: List,
        T_matrix: List,
        t_label: int,
        i_label: int,
        f_label: int
    ) -> int:
        placeholder_vector = [0.0] * self.vector_dim

        # Convert all types to JSON-serializable Python native types
        f_serializable = self._to_serializable(F_matrix)
        i_serializable = self._to_serializable(I_matrix)
        t_serializable = self._to_serializable(T_matrix)

        data = [
            [msu_id],
            [placeholder_vector],
            [json.dumps(f_serializable)],
            [json.dumps(i_serializable)],
            [json.dumps(t_serializable)],
            [t_label],
            [i_label],
            [f_label]
        ]

        self.collection.insert(data)
        return msu_id

    def _to_serializable(self, obj):
        """Recursively convert any nested object to JSON-serializable Python native types.

        Handles: numpy.ndarray, numpy scalar, tuple, list, dict, float, int, etc.
        """
        if obj is None:
            return None

        # Handle numpy types
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return self._to_serializable(obj.tolist())

        # Handle tuple (convert to list)
        if isinstance(obj, tuple):
            return [self._to_serializable(item) for item in obj]

        # Handle list/array - recursively convert each element
        if isinstance(obj, list):
            result = []
            for item in obj:
                result.append(self._to_serializable(item))
            return result

        # Handle dict
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}

        # For scalar types (str, int, float, bool), return as-is
        return obj

    def batch_insert(
        self,
        F_list: List[List],
        I_list: List[List],
        T_list: List[List],
        T_labels: List[int],
        I_labels: List[int],
        F_labels: List[int]
    ) -> int:
        if not (len(F_list) == len(I_list) == len(T_list) == len(T_labels) == len(I_labels) == len(F_labels)):
            raise ValueError("All lists must have same length")

        for i in range(len(F_list)):
            self.insert(
                msu_id=i,
                F_matrix=F_list[i],
                I_matrix=I_list[i],
                T_matrix=T_list[i],
                t_label=T_labels[i],
                i_label=I_labels[i],
                f_label=F_labels[i]
            )

        logger.info(f"Batch insert complete: {len(F_list)} records")
        return len(F_list)

    def flush(self):
        if self.collection:
            try:
                self.collection.flush()
            except Exception as e:
                logger.warning(f"Flush failed (ignored): {e}")

    def get_by_id(self, msu_id: int) -> Optional[Dict[str, Any]]:
        result = self.collection.query(
            expr=f"id == {msu_id}",
            output_fields=["id", "F_matrix", "I_matrix", "T_matrix",
                          "T_label", "I_label", "F_label"]
        )

        if not result:
            return None

        item = result[0]
        return {
            'id': item['id'],
            'F_matrix': json.loads(item['F_matrix']),
            'I_matrix': json.loads(item['I_matrix']),
            'T_matrix': json.loads(item['T_matrix']),
            'T_label': item['T_label'],
            'I_label': item['I_label'],
            'F_label': item['F_label']
        }

    def query_by_label(self, label_type: str, label_value: int) -> List[Dict[str, Any]]:
        label_field = f"{label_type}_label"
        result = self.collection.query(
            expr=f"{label_field} == {label_value}",
            output_fields=["id", "F_matrix", "I_matrix", "T_matrix",
                          "T_label", "I_label", "F_label"]
        )

        return [
            {
                'id': item['id'],
                'F_matrix': json.loads(item['F_matrix']),
                'I_matrix': json.loads(item['I_matrix']),
                'T_matrix': json.loads(item['T_matrix']),
                'T_label': item['T_label'],
                'I_label': item['I_label'],
                'F_label': item['F_label']
            }
            for item in result
        ]

    def get_all(self, limit: int = 1000) -> List[Dict[str, Any]]:
        result = self.collection.query(
            expr="id >= 0",
            output_fields=["id", "F_matrix", "I_matrix", "T_matrix",
                          "T_label", "I_label", "F_label"],
            limit=limit
        )

        return [
            {
                'id': item['id'],
                'F_matrix': json.loads(item['F_matrix']),
                'I_matrix': json.loads(item['I_matrix']),
                'T_matrix': json.loads(item['T_matrix']),
                'T_label': item['T_label'],
                'I_label': item['I_label'],
                'F_label': item['F_label']
            }
            for item in result
        ]

    def count(self) -> int:
        return self.collection.num_entities

    def close(self):
        if self.collection:
            # Don't flush here - data already flushed after batch_insert
            # Flush on already-loaded collection causes Windows file lock issues with milvus-lite
            try:
                self.collection.release()
            except Exception as e:
                logger.warning(f"Release failed (ignored): {e}")
        if "default" in connections.list_connections():
            connections.disconnect("default")
        logger.info("Milvus connection closed")