"""
Query script for MSU video similarity retrieval.

Supports:
1. Query from existing database by MSU ID
2. Query from existing database by F/I/T features (for new video segments)
3. Element search (inverted index)
4. Label search

Usage:
    # Query similar MSUs to MSU 0 in existing database
    python -m query_by_features --db "./msu_db" --msu-id 0 --top-k 5

    # Query with new video segment features
    python -m query_by_features --db "./msu_db" --input-f "[[0, 100, 200, 110, 210], [1, 300, 400, 310, 410]]" --input-i "[[0, 1], [1, 0]]" --input-t "[[0, 0.5], [0.2, 0.8]]"

    # Element search
    python -m query_by_features --db "./msu_db" --element-type F --element-value 0

    # Label search
    python -m query_by_features --db "./msu_db" --label-type T --label-value 0
"""

import argparse
import sys
import os
import json
import numpy as np

parent_dir = r"D:\XW2\yolov13"
yolov13_main = r"D:\XW2\yolov13\main\yolov13-main"
for p in [parent_dir, yolov13_main]:
    if p not in sys.path:
        sys.path.insert(0, p)


def load_db(db_path):
    """Load storage components from existing database."""
    from storage.inverted_index import InvertedIndex
    from storage.milvus_storage import MilvusStorage

    milvus_path = os.path.join(db_path, "msu_milvus_db")
    inverted_path = db_path

    # Load storage
    milvus = MilvusStorage(db_path=milvus_path)
    inverted_index = InvertedIndex(db_path=inverted_path)

    # Load all data from Milvus
    all_data = milvus.get_all(limit=10000)
    if not all_data:
        raise ValueError("No MSU data found in database")

    # Build F/I/T lists and labels
    F_list = [np.array(m['F_matrix']) for m in all_data]
    I_list = [np.array(m['I_matrix']) for m in all_data]
    T_list = [np.array(m['T_matrix']) for m in all_data]
    T_labels = np.array([m['T_label'] for m in all_data])
    I_labels = np.array([m['I_label'] for m in all_data])
    F_labels = np.array([m['F_label'] for m in all_data])

    # Build hierarchical index (lazy import to avoid sklearn dependency issues)
    from clustering.hierarchical_index import HierarchicalClusterIndex

    hierarchical_index = HierarchicalClusterIndex()
    hierarchical_index.T_labels_ = T_labels
    hierarchical_index.I_labels_ = I_labels
    hierarchical_index.F_labels_ = F_labels
    hierarchical_index.T_reps_ = {}
    hierarchical_index.I_reps_ = {}
    hierarchical_index.F_reps_ = {}

    # Compute representatives (average within cluster)
    for label_arr, name in [(T_labels, 'T'), (I_labels, 'I'), (F_labels, 'F')]:
        for cluster_id in np.unique(label_arr):
            indices = np.where(label_arr == cluster_id)[0]
            if len(indices) > 0:
                rep_idx = indices[len(indices) // 2]  # Middle element as representative
                if name == 'T':
                    hierarchical_index.T_reps_[cluster_id] = rep_idx
                elif name == 'I':
                    hierarchical_index.I_reps_[cluster_id] = rep_idx
                else:
                    hierarchical_index.F_reps_[cluster_id] = rep_idx

    hierarchical_index.feature_data_ = {
        'F': F_list, 'I': I_list, 'T': T_list,
        'T_max_rows': max((t.shape[0] for t in T_list), default=1)
    }
    hierarchical_index._build_index(len(F_list))

    from query.query_manager import QueryManager
    query_manager = QueryManager(milvus, inverted_index, hierarchical_index)

    return query_manager, milvus


def main():
    parser = argparse.ArgumentParser(description="Query MSU database by features")
    parser.add_argument('--db', type=str, default='./msu_db', help='Database path')
    parser.add_argument('--msu-id', type=int, help='Query MSU ID in database')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    # Input features (JSON strings)
    parser.add_argument('--input-f', type=str, help='F matrix as JSON string')
    parser.add_argument('--input-i', type=str, help='I matrix as JSON string')
    parser.add_argument('--input-t', type=str, help='T matrix as JSON string')
    # Element search
    parser.add_argument('--element-type', type=str, choices=['F', 'I'], help='Element type')
    parser.add_argument('--element-value', type=float, help='Element value')
    # Label search
    parser.add_argument('--label-type', type=str, choices=['T', 'I', 'F'], help='Label type')
    parser.add_argument('--label-value', type=int, help='Label value')

    args = parser.parse_args()

    # Check if database exists
    milvus_path = os.path.join(args.db, "msu_milvus_db")
    if not os.path.exists(milvus_path + ".db"):
        print(f"Error: Database not found at {milvus_path}. Run preprocessing first.")
        return 1

    try:
        query_manager, milvus = load_db(args.db)
    except Exception as e:
        print(f"Error loading database: {e}")
        return 1

    # Execute query
    if args.element_type is not None and args.element_value is not None:
        # Element search (inverted index - fast)
        msu_ids = query_manager.search_by_element(args.element_type, args.element_value)
        print(f"\n=== Element Search: {args.element_type}={args.element_value} ===")
        print(f"Found {len(msu_ids)} MSUs:")
        for msu_id in msu_ids[:20]:
            print(f"  MSU {msu_id}")
        if len(msu_ids) > 20:
            print(f"  ... and {len(msu_ids) - 20} more")

    elif args.label_type is not None and args.label_value is not None:
        # Label search
        results = query_manager.get_by_label(args.label_type, args.label_value)
        print(f"\n=== Label Search: {args.label_type}_label={args.label_value} ===")
        print(f"Found {len(results)} MSUs:")
        for res in results[:20]:
            print(f"  MSU {res['id']}")
        if len(results) > 20:
            print(f"  ... and {len(results) - 20} more")

    elif args.msu_id is not None:
        # Query by MSU ID in database
        print(f"\n=== Query by MSU ID: {args.msu_id} ===")
        results = query_manager.find_similar_by_id(args.msu_id, args.top_k)
        print(f"Found {len(results)} similar MSUs:")
        for i, res in enumerate(results):
            print(f"  {i+1}. MSU {res['msu_id']}, F distance: {res['f_distance']:.4f}")

    elif args.input_f and args.input_i and args.input_t:
        # Query by input F/I/T features (new video segment)
        try:
            F_matrix = json.loads(args.input_f)
            I_matrix = json.loads(args.input_i)
            T_matrix = json.loads(args.input_t)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON input: {e}")
            milvus.close()
            return 1

        print(f"\n=== Query by Input Features ===")
        print(f"F matrix: {len(F_matrix)} objects")
        print(f"I matrix: {len(I_matrix)}x{len(I_matrix) if I_matrix else 0}")
        print(f"T matrix: {len(T_matrix)} objects")

        results = query_manager.find_similar_by_features(F_matrix, I_matrix, T_matrix, args.top_k)
        print(f"\nFound {len(results)} similar MSUs:")
        for i, res in enumerate(results):
            print(f"  {i+1}. MSU {res['msu_id']}, F distance: {res['f_distance']:.4f}")

    else:
        print("Error: Specify --msu-id, --input-f/--input-i/--input-t, --element-type, or --label-type")
        print("\nExamples:")
        print("  # Query by MSU ID")
        print("  python -m query_by_features --db ./msu_db --msu-id 0 --top-k 5")
        print("  # Query by input features (new video segment)")
        print("  python -m query_by_features --db ./msu_db --input-f '[[0,100,200]]' --input-i '[[0,1],[1,0]]' --input-t '[[0,0.5]]'")
        print("  # Element search")
        print("  python -m query_by_features --db ./msu_db --element-type F --element-value 0")
        milvus.close()
        return 1

    milvus.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())