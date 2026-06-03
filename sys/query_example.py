"""
Fast query script for MSU video similarity retrieval.

直接加载已有数据库文件进行查询，无需重新运行pipeline。
倒排索引查询瞬间完成。

Usage:
    python -m query_example --msu-id 0 --top-k 5
    python -m query_example --element-type F --element-value 0
    python -m query_example --list
    python -m query_example --label-type T --label-value 0
"""

import argparse
import sys
import os
import json

parent_dir = r"D:\XW2\yolov13"
yolov13_main = r"D:\XW2\yolov13\main\yolov13-main"
for p in [parent_dir, yolov13_main]:
    if p not in sys.path:
        sys.path.insert(0, p)

from storage.inverted_index import InvertedIndex
from storage.sqlite_manager import SQLiteManager
from storage.milvus_storage import MilvusStorage
from similarity.CTMD import pad_T_matrix


def load_inverted_index(db_dir: str):
    """加载倒排索引。"""
    # Mirror main: pass db_dir (milvus db directory), index file is in parent dir
    return InvertedIndex(db_path=db_dir)


def load_milvus(db_dir: str):
    """加载Milvus存储。"""
    milvus_path = os.path.join(db_dir, "msu_milvus_db")
    try:
        return MilvusStorage(db_path=milvus_path)
    except Exception as e:
        print(f"Warning: Milvus load failed: {e}")
        return None


def query_by_element(matrix_type: str, value: float, db_dir: str):
    """倒排索引查询，瞬间完成。"""
    index = load_inverted_index(db_dir)
    msu_ids = index.query(matrix_type, value)
    index.close()

    print(f"\n{'='*60}")
    print(f"倒排索引查询: {matrix_type}矩阵包含value={value}")
    print(f"找到 {len(msu_ids)} 个MSU (耗时: 瞬间)")
    print(f"{'='*60}")

    if not msu_ids:
        return

    sqlite_path = os.path.join(db_dir, "video_db.sqlite")
    sqlite_mgr = SQLiteManager(db_path=sqlite_path) if os.path.exists(sqlite_path) else None

    for msu_id in msu_ids[:20]:
        vsu_info = sqlite_mgr.get_vsu_by_id(msu_id) if sqlite_mgr else None
        clip = vsu_info['clip_path'] if vsu_info else 'N/A'
        frame = f"{vsu_info['frame_start']}-{vsu_info['frame_end']}" if vsu_info else 'N/A'
        print(f"  MSU {msu_id}: clip={os.path.basename(clip) if clip else 'N/A'}, frames={frame}")

    if len(msu_ids) > 20:
        print(f"  ... 还有 {len(msu_ids) - 20} 个结果")

    if sqlite_mgr:
        sqlite_mgr.close()


def _prepare_t_matrix(t_matrix, t_max_rows):
    """Pad T matrix to t_max_rows height, then flatten."""
    import numpy as np
    t_np = np.array(t_matrix)
    if t_np.size == 0:
        return np.zeros((t_max_rows, 2)).flatten()
    if t_np.shape[0] < t_max_rows:
        padding = np.zeros((t_max_rows - t_np.shape[0], t_np.shape[1]))
        t_padded = np.vstack([t_np, padding])
    else:
        t_padded = t_np[:t_max_rows]
    return t_padded.flatten()


def query_similar_by_id(msu_id: int, top_k: int, db_dir: str):
    """基于Milvus的层次聚类查询。使用T->I->F三层过滤。"""
    milvus = load_milvus(db_dir)
    if milvus is None:
        print("Error: Milvus database not available")
        return

    all_data = milvus.get_all(limit=10000)
    if not all_data:
        print("No data in Milvus database")
        milvus.close()
        return

    # Get input MSU
    input_msu = milvus.get_by_id(msu_id)
    if not input_msu:
        print(f"Error: MSU {msu_id} not found in database")
        milvus.close()
        return

    t_label = input_msu['T_label']
    i_label = input_msu['I_label']
    f_label = input_msu['F_label']

    print(f"\n{'='*60}")
    print(f"层次聚类查询: MSU {msu_id} (T={t_label}, I={i_label}, F={f_label})")
    print(f"找 top-{top_k} 相似片段")
    print(f"{'='*60}")

    # Step 1: Filter by T cluster (same T_label)
    candidates = [m for m in all_data if m['T_label'] == t_label]
    print(f"  T聚类过滤: {len(candidates)} 个候选")

    # Step 2: Filter by I cluster within T cluster
    if candidates:
        candidates = [m for m in candidates if m['I_label'] == i_label]
        print(f"  T+I聚类过滤: {len(candidates)} 个候选")

    # Step 3: Filter by F cluster within T+I cluster
    if candidates:
        candidates = [m for m in candidates if m['F_label'] == f_label]
        print(f"  T+I+F聚类过滤: {len(candidates)} 个候选")

    if not candidates:
        print("  No candidates in same cluster, using T+I filter only")
        candidates = [m for m in all_data if m['T_label'] == t_label and m['I_label'] == i_label]

    # Compute F distance (PM-TSMD) to all candidates
    import numpy as np
    from similarity.PM_TSMD import compute_PM_TSMD

    input_F = np.array(input_msu['F_matrix'])
    distances = []

    for m in candidates:
        if m['id'] == msu_id:
            continue
        try:
            dist = compute_PM_TSMD(input_F, np.array(m['F_matrix']))
            distances.append((m['id'], dist, m))
        except Exception:
            continue

    distances.sort(key=lambda x: x[1])
    top_results = distances[:top_k]

    sqlite_path = os.path.join(db_dir, "video_db.sqlite")
    sqlite_mgr = SQLiteManager(db_path=sqlite_path) if os.path.exists(sqlite_path) else None

    for i, (res_id, dist, msu_data) in enumerate(top_results, 1):
        vsu_info = sqlite_mgr.get_vsu_by_id(res_id) if sqlite_mgr else None
        clip = os.path.basename(vsu_info['clip_path']) if vsu_info and vsu_info['clip_path'] else 'N/A'
        frame = f"{vsu_info['frame_start']}-{vsu_info['frame_end']}" if vsu_info and vsu_info['frame_start'] is not None else 'N/A'
        print(f"\n  Rank {i}: MSU {res_id}, F距离={dist:.4f}")
        print(f"    Clip: {clip}")
        print(f"    Frames: {frame}")

    if sqlite_mgr:
        sqlite_mgr.close()
    milvus.close()


def list_all_msus(db_dir: str):
    """列出所有MSU。"""
    milvus = load_milvus(db_dir)
    if milvus is None:
        print("Error: Milvus database not available")
        return

    all_data = milvus.get_all(limit=10000)
    print(f"\n{'='*60}")
    print(f"所有MSU列表 (共 {len(all_data)} 个)")
    print(f"{'='*60}")

    sqlite_path = os.path.join(db_dir, "video_db.sqlite")
    sqlite_mgr = SQLiteManager(db_path=sqlite_path) if os.path.exists(sqlite_path) else None

    for msu_data in all_data:
        msu_id = msu_data['id']
        vsu_info = sqlite_mgr.get_vsu_by_id(msu_id) if sqlite_mgr else None
        clip = os.path.basename(vsu_info['clip_path']) if vsu_info and vsu_info['clip_path'] else 'N/A'
        frame = f"{vsu_info['frame_start']}-{vsu_info['frame_end']}" if vsu_info and vsu_info['frame_start'] is not None else 'N/A'
        print(f"  MSU {msu_id}: T={msu_data['T_label']}, I={msu_data['I_label']}, F={msu_data['F_label']}, clip={clip}, frames={frame}")

    if sqlite_mgr:
        sqlite_mgr.close()
    milvus.close()


def query_by_label(label_type: str, label_value: int, db_dir: str):
    """按聚类标签查询。"""
    milvus = load_milvus(db_dir)
    if milvus is None:
        print("Error: Milvus database not available")
        return

    results = milvus.query_by_label(label_type, label_value)
    print(f"\n{'='*60}")
    print(f"按{label_type}标签查询: {label_type}_label={label_value}, 找到 {len(results)} 个MSU")
    print(f"{'='*60}")

    sqlite_path = os.path.join(db_dir, "video_db.sqlite")
    sqlite_mgr = SQLiteManager(db_path=sqlite_path) if os.path.exists(sqlite_path) else None

    for msu_data in results:
        msu_id = msu_data['id']
        vsu_info = sqlite_mgr.get_vsu_by_id(msu_id) if sqlite_mgr else None
        clip = os.path.basename(vsu_info['clip_path']) if vsu_info and vsu_info['clip_path'] else 'N/A'
        frame = f"{vsu_info['frame_start']}-{vsu_info['frame_end']}" if vsu_info and vsu_info['frame_start'] is not None else 'N/A'
        print(f"  MSU {msu_id}: T={msu_data['T_label']}, I={msu_data['I_label']}, F={msu_data['F_label']}")
        print(f"    Clip: {clip}, Frames: {frame}")

    if sqlite_mgr:
        sqlite_mgr.close()
    milvus.close()


def main():
    parser = argparse.ArgumentParser(description="MSU Video Fast Query")
    parser.add_argument('--db', type=str, default='D:/XW2/yolov13/system/msu_db',
                        help='Database base directory')
    parser.add_argument('--msu-id', type=int, default=0,
                        help='MSU ID to query')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of results')
    parser.add_argument('--list', action='store_true',
                        help='List all MSUs')
    parser.add_argument('--label-type', type=str, choices=['T', 'I', 'F'],
                        help='Query by cluster label type')
    parser.add_argument('--label-value', type=int,
                        help='Cluster label value')
    parser.add_argument('--element-type', type=str, choices=['F', 'I'],
                        help='Query by matrix element (F or I) - inverted index query (instant)')
    parser.add_argument('--element-value', type=float,
                        help='Element value to search for')
    args = parser.parse_args()

    db_dir = args.db

    if not os.path.exists(db_dir):
        print(f"Error: Database directory not found: {db_dir}")
        print("Please run the pipeline first: python -m run --video ...")
        return 1

    if args.list:
        list_all_msus(db_dir)

    elif args.label_type is not None and args.label_value is not None:
        query_by_label(args.label_type, args.label_value, db_dir)

    elif args.element_type is not None and args.element_value is not None:
        query_by_element(args.element_type, args.element_value, db_dir)

    else:
        query_similar_by_id(args.msu_id, args.top_k, db_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())