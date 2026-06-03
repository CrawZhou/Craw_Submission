# 04_query_tree.py - Query SVS using PERCH tree index only

import pickle
import numpy as np
import os
import time
import json
import sys

sys.path.append('../runs/scripts')
from pyemd import emd  # Import EMD computation dependency


# -------------------------- 1. Core class definitions (Node/PERCH, required for tree query) --------------------------

class Node:
    def __init__(self, data=None, svs_id=None):
        self.data = data          # SVS feature data stored in node
        self.svs_id = svs_id      # Corresponding SVS ID
        self.left = None          # Left child node
        self.right = None         # Right child node
        self.parent = None        # Parent node
        self.node_id = None       # Node ID (for identifying internal nodes)
        self.is_leaf = (data is not None)  # Whether is leaf node
        self.centroid = None      # Node centroid (for distance computation)


class PERCH:
    def __init__(self):
        self.root = None
        self.node_counter = 0     # Used to assign IDs to internal nodes

    def _compute_centroid(self, data):
        """Compute centroid of data"""
        if len(data) == 0:
            return None
        return np.mean(data, axis=0)

    def _compute_distance(self, data1, data2):
        """Compute distance between two datasets (Euclidean distance, based on average features)"""
        if len(data1) == 0 or len(data2) == 0:
            return float('inf')

        max_features = 20
        if len(data1) > max_features:
            indices = np.linspace(0, len(data1)-1, max_features, dtype=int)
            data1 = data1[indices]
        if len(data2) > max_features:
            indices = np.linspace(0, len(data2)-1, max_features, dtype=int)
            data2 = data2[indices]

        avg1 = np.mean(data1, axis=0)
        avg2 = np.mean(data2, axis=0)
        return np.linalg.norm(avg1 - avg2)


# -------------------------- 2. Common utility functions (required for tree query) --------------------------

def load_perch_tree(filepath):
    """Load PERCH tree, ensure class definitions available"""
    import __main__
    __main__.Node = Node
    __main__.PERCH = PERCH
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def omd(a, b):
    """Compute Object Movement Distance between two SVS"""
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    if np.array_equal(a, b):
        return 0.0

    # Limit feature count and dimension
    max_features = 30
    if len(a) > max_features:
        a = a[np.linspace(0, len(a)-1, max_features, dtype=int)]
    if len(b) > max_features:
        b = b[np.linspace(0, len(b)-1, max_features, dtype=int)]

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    max_dim = 256
    if a.shape[1] > max_dim:
        a = a[:, :max_dim]
    if b.shape[1] > max_dim:
        b = b[:, :max_dim]

    # Ensure matching feature count
    min_features = min(a.shape[0], b.shape[0])
    if min_features == 0:
        return float('inf')

    a = a[:min_features]
    b = b[:min_features]

    # Compute EMD
    dist_mat = np.linalg.norm(a[:, None] - b[None, :], axis=2).astype(np.float32)
    w1 = np.ones(len(a), dtype=np.float32) / len(a)
    w2 = np.ones(len(b), dtype=np.float32) / len(b)
    return float(emd(w1, w2, dist_mat))


def normalized_omd(a, b):
    """Normalize OMD distance to [0,1]"""
    distance = omd(a, b)
    return 1.0 if distance == float('inf') else min(distance / 10.0, 1.0)


def compute_ocd(a, b):
    """Compute Object Centroid Distance (lower bound for OMD, used for pruning)"""
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    return np.linalg.norm(np.mean(a, axis=0) - np.mean(b, axis=0))


def calculate_accuracy(predicted_id, actual_id):
    """Calculate query accuracy"""
    if predicted_id == actual_id:
        return 1.0
    return 0.5 if abs(predicted_id - actual_id) <= 1 else 0.0


# -------------------------- 3. Tree query core function (wrapped) --------------------------

def tree_based_query(query_features, perch_tree_path, actual_svs_id, segment_duration):
    """
    PERCH tree index query main logic
    Params: query features, tree path, expected ID, SVS duration
    Returns: query result dict
    """
    # Load PERCH tree
    perch = load_perch_tree(perch_tree_path)
    if perch.root is None:
        return {"status": "failed", "reason": "PERCH tree is empty, no SVS data"}

    # Tree recursive search (with OCD pruning)
    start_time = time.time()

    def _search(node, best_dist=float('inf')):
        if not node:
            return None, float('inf')
        if node.is_leaf:
            if node.data and len(query_features) > 0:
                ocd = compute_ocd(query_features, node.data)
                if ocd > best_dist:
                    return None, float('inf')
                omd_dist = omd(query_features, node.data)
                return node.svs_id, omd_dist
            return None, float('inf')

        # Recursively search left/right subtrees
        left_id, left_dist = _search(node.left, best_dist)
        right_id, right_dist = _search(node.right, min(best_dist, left_dist) if left_dist != float('inf') else best_dist)
        return (left_id, left_dist) if left_dist <= right_dist else (right_id, right_dist)

    best_id, best_dist = _search(perch.root)
    query_time = time.time() - start_time

    # Compute supplementary info
    matched_svs_path = f'../data/svs/svs_{best_id:03d}.pkl'
    with open(matched_svs_path, 'rb') as f:
        matched_feat = pickle.load(f)
    norm_dist = normalized_omd(query_features, matched_feat)
    accuracy = calculate_accuracy(best_id, actual_id)

    # Time range conversion
    start = best_id * segment_duration
    end = (best_id + 1) * segment_duration
    time_range = f"{start//60}m{start%60}s ~ {end//60}m{end%60}s"

    return {
        "status": "success",
        "query_type": "PERCH tree index query",
        "best_svs_id": best_id,
        "omd_distance": round(best_dist, 4),
        "normalized_omd": round(norm_dist, 4),
        "query_time": round(query_time, 4),
        "accuracy": round(accuracy, 4),
        "time_range": time_range
    }


# -------------------------- 4. Main function (in-code params + execution) --------------------------

def main():
    # -------------------------- In-code parameter definitions (user modifiable) --------------------------
    query_path = '/home/nanchang/ZZQ/yolov13/main/data/svs/svs_001.pkl'  # Target query SVS path
    metadata_path = '../data/query/metadata.json'                        # Metadata path (for accuracy)
    perch_tree_path = '../runs/intra_index.pkl'                         # PERCH tree path
    default_svs_duration = 30                                          # Default SVS duration (seconds)
    default_start_time = 1200                                          # Default query start time (seconds)

    # -------------------------- Step 1: Load query SVS features --------------------------
    print("=" * 60)
    print("[PERCH Tree Index Query] - Start")
    print(f"1. Loading query SVS: {query_path}")
    with open(query_path, 'rb') as f:
        query_features = pickle.load(f)
    print(f"   Successfully loaded: {len(query_features)} object features")

    # -------------------------- Step 2: Read metadata (determine expected SVS ID) --------------------------
    print("\n2. Reading query metadata")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            content = f.read().strip()
            if content:
                meta = json.loads(content)
                svs_duration = meta.get('segment_duration', default_svs_duration)
                actual_svs_id = meta.get('expected_svs_id', meta.get('start_time', default_start_time) // svs_duration)
            else:
                print("   Metadata file empty, using default parameters")
                svs_duration = default_svs_duration
                actual_svs_id = default_start_time // svs_duration
    else:
        print(f"   Metadata file not found ({metadata_path}), using default parameters")
        svs_duration = default_svs_duration
        actual_svs_id = default_start_time // svs_duration

    print(f"   Current params: SVS duration={svs_duration}s, expected match ID={actual_svs_id}")

    # -------------------------- Step 3: Execute tree query --------------------------
    print("\n3. Execute PERCH tree index query")
    result = tree_based_query(query_features, perch_tree_path, actual_svs_id, svs_duration)

    # -------------------------- Step 4: Output results --------------------------
    print("\n4. Query results")
    if result["status"] == "success":
        print(f"   Query type: {result['query_type']}")
        print(f"   Best matching SVS ID: {result['best_svs_id']:03d}")
        print(f"   OMD distance: {result['omd_distance']} (smaller = more similar)")
        print(f"   Normalized OMD: {result['normalized_omd']} (0=identical, 1=completely different)")
        print(f"   Query time: {result['query_time']} seconds ({result['query_time']*1000:.1f} ms)")
        print(f"   Accuracy: {result['accuracy']:.2%}")
        print(f"   Corresponding video time: {result['time_range']}")
    else:
        print(f"   Query failed: {result['reason']}")

    print("\n[PERCH Tree Index Query] - End")
    print("=" * 60)


if __name__ == "__main__":
    main()
