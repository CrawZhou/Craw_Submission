import json
import os
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger("InvertedIndex")


class InvertedIndex:
    """Inverted index mapping matrix element values to MSU IDs.
    Mirrors main/mil_storage.py InvertedIndexManager pattern."""

    def __init__(self, db_path: str = "./msu_db"):
        self.db_path = db_path
        self.collection_name = "matrix_collection"
        self.collection = None
        self.index = defaultdict(list)
        # Handle both cases:
        # - db_path is a directory like "D:/XW2/yolov13/system/msu_db" -> index_file = db_path/inverted_index.json
        # - db_path is already a file path like "D:/XW2/yolov13/system/msu_db/inverted_index.json" -> use directly
        if "inverted_index.json" in db_path:
            self.index_file = db_path
        else:
            self.index_file = os.path.join(db_path, "inverted_index.json")
        self._build_time = 0.0
        self._build_count = 0

        self._load()

    def _load(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
                logger.info(f"Loaded inverted index from {self.index_file}")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")

    def save(self):
        try:
            with open(self.index_file, 'w') as f:
                json.dump(dict(self.index), f)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def close(self):
        self.save()

    def update(self, msu_id: int, F_matrix: List[List], I_matrix: List[List]):
        """Update index with an MSU's F and I matrices.
        Mirrors main/mil_storage.py InvertedIndexManager.update_index()."""
        import time
        start = time.perf_counter()

        # Index F matrix - only the first element of each row (class id)
        for row in F_matrix:
            if row:
                key = f"F:{row[0]}"
                if key not in self.index:
                    self.index[key] = []
                if msu_id not in self.index[key]:
                    self.index[key].append(msu_id)

        # Index I matrix - all values
        for row in I_matrix:
            for val in row:
                key = f"I:{int(val)}"
                if key not in self.index:
                    self.index[key] = []
                if msu_id not in self.index[key]:
                    self.index[key].append(msu_id)

        toc = time.perf_counter()
        self._build_time += toc - start
        self._build_count += 1
        self.save()

    def query(self, matrix_type: str, value: Any) -> List[int]:
        # Try exact key first
        key = f"{matrix_type}:{value}"
        result = self.index.get(key, [])

        # If no result and value is a float that equals an integer (e.g., 2.0), try integer key
        if not result and isinstance(value, float) and value == int(value):
            int_key = f"{matrix_type}:{int(value)}"
            result = self.index.get(int_key, [])

        return result

    def query_and(self, matrix_type: str, values: List[Any]) -> List[int]:
        if not values:
            return []
        sets = [set(self.query(matrix_type, v)) for v in values]
        result = sets[0]
        for s in sets[1:]:
            result = result & s
        return list(result)

    def query_or(self, matrix_type: str, values: List[Any]) -> List[int]:
        if not values:
            return []
        sets = [set(self.query(matrix_type, v)) for v in values]
        result = sets[0]
        for s in sets[1:]:
            result = result | s
        return list(result)

    def get_stats(self) -> Dict:
        return {
            'total_keys': len(self.index),
            'total_updates': self._build_count,
            'total_build_time': self._build_time,
            'avg_build_time': self._build_time / max(self._build_count, 1)
        }

    def print_stats(self):
        stats = self.get_stats()
        print("\n" + "=" * 40)
        print("Inverted Index Statistics")
        print("=" * 40)
        print(f"  Total keys: {stats['total_keys']}")
        print(f"  Total updates: {stats['total_updates']}")
        print(f"  Total build time: {stats['total_build_time']:.4f}s")
        print(f"  Avg build time: {stats['avg_build_time']:.6f}s")