"""
Storage module for MSU system.
Mirrors the pattern from main/mil_storage.py
"""

from .inverted_index import InvertedIndex
from .milvus_storage import MilvusStorage
from .sqlite_manager import SQLiteManager

__all__ = [
    'InvertedIndex',
    'MilvusStorage',
    'SQLiteManager'
]