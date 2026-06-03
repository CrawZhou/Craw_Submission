"""
Query video clip similarity against existing MSU database.

输入一个视频片段，整个作为一个MSU（不分割），然后用T/I/F特征匹配索引库中的MSU。

Usage:
    python -m query_video_clip --video "path/to/video.mp4" --db ./msu_db --top-k 5
    python -m query_video_clip --frame-info "path/to/frame_info.json" --db ./msu_db --top-k 5
"""
import argparse
import sys
import os
from collections import defaultdict

parent_dir = r"D:\XW2\yolov13"
yolov13_main = r"D:\XW2\yolov13\main\yolov13-main"
for p in [parent_dir, yolov13_main]:
    if p not in sys.path:
        sys.path.insert(0, p)


def build_full_duration(frame_info):
    """Build duration map treating entire video as one MSU (same as video_split1.py)."""
    track2frames = defaultdict(list)
    for frame_str, objs in frame_info.items():
        frame = int(frame_str)
        for obj in objs:
            track2frames[obj['id']].append(frame)

    durations = {}
    for tid, frames in track2frames.items():
        durations[tid] = (min(frames), max(frames))
    return durations


def main():
    parser = argparse.ArgumentParser(description="Query video clip as single MSU")
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--frame-info', type=str, help='frame_info JSON file path')
    parser.add_argument('--db', type=str, default='./msu_db', help='Database path')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--model-path', type=str, help='YOLO model path')
    args = parser.parse_args()

    if not args.video and not args.frame_info:
        print("Error: --video or --frame-info required")
        return 1

    # Step 1: Get frame_info (tracking or loading)
    if args.frame_info:
        import json
        with open(args.frame_info, 'r', encoding='utf-8') as f:
            frame_info = json.load(f)
        print(f"Loaded frame_info: {len(frame_info)} frames")
    else:
        from data.tracker import Tracker
        from config.settings import Config

        config = Config()
        if args.model_path:
            config.path.model_path = args.model_path

        tracker = Tracker(
            model_path=config.path.model_path,
            tracker_type=config.video.tracker_type,
            conf=config.video.detection_conf,
            iou=config.video.detection_iou,
            device=config.video.device
        )
        result = tracker.track_video(args.video)
        frame_info = result['frame_info']
        print(f"Tracking done: {result['total_frames']} frames")

    # Step 2: Build MSU ranges and durations (single MSU = entire video)
    all_frames = sorted(int(fid) for fid in frame_info.keys())
    if not all_frames:
        print("Error: No valid frames")
        return 1

    msu_ranges = [(all_frames[0], all_frames[-1])]
    durations = [build_full_duration(frame_info)]
    print(f"MSU range: [{msu_ranges[0][0]}, {msu_ranges[0][1]}], {len(durations[0])} objects")

    # Step 3: Build F/I/T features
    from features.feature_builder import FeatureBuilder
    from config.settings import Config

    config = Config()
    config.storage.base_dir = args.db
    # _init_storage_paths is on Config, not StorageConfig
    os.makedirs(args.db, exist_ok=True)
    config.storage.milvus_db_path = os.path.join(args.db, "msu_milvus_db")
    config.storage.sqlite_db_path = os.path.join(args.db, "video_db.sqlite")
    config.storage.clip_dir = os.path.join(args.db, "clips")
    config.storage.inverted_index_path = os.path.join(args.db, "inverted_index.json")
    os.makedirs(config.storage.clip_dir, exist_ok=True)

    fb = FeatureBuilder(config)
    fb.set_msu_splits(msu_ranges, durations)
    F, I, T = fb.build(frame_info, skip_splits=True)
    print(f"Features built: F={len(F[0]) if F else 0} objects, I shape={I[0].shape if hasattr(I[0], 'shape') else len(I[0])}, T shape={T[0].shape if hasattr(T[0], 'shape') else len(T[0])}")

    # Step 4: Load existing database
    from storage.milvus_storage import MilvusStorage
    from storage.inverted_index import InvertedIndex
    from clustering.hierarchical_index import HierarchicalClusterIndex
    from query.query_manager import QueryManager

    milvus_path = os.path.join(args.db, "msu_milvus_db")
    if not os.path.exists(milvus_path + ".db"):
        print(f"Error: Database not found at {milvus_path}. Run with --full first.")
        return 1

    import numpy as np
    from similarity.CTMD import compute_CTMD_fast
    from similarity.I_distance import compute_I_distance_fast
    from similarity.PM_TSMD import compute_PM_TSMD

    milvus = MilvusStorage(db_path=milvus_path)
    inverted_index = InvertedIndex(db_path=args.db)

    all_data = milvus.get_all(limit=10000)
    if not all_data:
        print("Error: No MSU data in database")
        milvus.close()
        return 1
    print(f"Database has {len(all_data)} MSUs")

    # Build hierarchical index from database
    F_list = [m['F_matrix'] for m in all_data]
    I_list = [np.array(m['I_matrix']) for m in all_data]
    T_list = [np.array(m['T_matrix']) for m in all_data]
    T_labels = np.array([m['T_label'] for m in all_data])
    I_labels = np.array([m['I_label'] for m in all_data])
    F_labels = np.array([m['F_label'] for m in all_data])

    hierarchical_index = HierarchicalClusterIndex()
    hierarchical_index.T_labels_ = T_labels
    hierarchical_index.I_labels_ = I_labels
    hierarchical_index.F_labels_ = F_labels
    t_max_rows = max([t.shape[0] for t in T_list]) if T_list else 1
    hierarchical_index.feature_data_ = {
        'F': F_list, 'I': I_list, 'T': T_list, 'T_max_rows': t_max_rows
    }
    hierarchical_index._build_index(len(F_list))

    n = len(T_list)
    t_dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = compute_CTMD_fast(T_list[i], T_list[j])
            t_dist_matrix[i, j] = t_dist_matrix[j, i] = d

    t_reps = {}
    for t_cid in np.unique(T_labels):
        indices = np.where(T_labels == t_cid)[0]
        if len(indices) == 1:
            t_reps[int(t_cid)] = int(indices[0])
        else:
            min_sum = float('inf')
            best_idx = indices[0]
            for idx in indices:
                dist_sum = sum(t_dist_matrix[idx, other] for other in indices)
                if dist_sum < min_sum:
                    min_sum = dist_sum
                    best_idx = idx
            t_reps[int(t_cid)] = int(best_idx)
    hierarchical_index.T_reps_ = t_reps

    i_dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = compute_I_distance_fast(I_list[i], I_list[j])
            i_dist_matrix[i, j] = i_dist_matrix[j, i] = d

    i_reps = {}
    for i_cid in np.unique(I_labels):
        indices = np.where(I_labels == i_cid)[0]
        if len(indices) == 1:
            i_reps[int(i_cid)] = int(indices[0])
        else:
            min_sum = float('inf')
            best_idx = indices[0]
            for idx in indices:
                dist_sum = sum(i_dist_matrix[idx, other] for other in indices)
                if dist_sum < min_sum:
                    min_sum = dist_sum
                    best_idx = idx
            i_reps[int(i_cid)] = int(best_idx)
    hierarchical_index.I_reps_ = i_reps

    f_dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = compute_PM_TSMD(F_list[i], F_list[j])
            f_dist_matrix[i, j] = f_dist_matrix[j, i] = d

    f_reps = {}
    for f_cid in np.unique(F_labels):
        indices = np.where(F_labels == f_cid)[0]
        if len(indices) == 1:
            f_reps[int(f_cid)] = int(indices[0])
        else:
            min_sum = float('inf')
            best_idx = indices[0]
            for idx in indices:
                dist_sum = sum(f_dist_matrix[idx, other] for other in indices)
                if dist_sum < min_sum:
                    min_sum = dist_sum
                    best_idx = idx
            f_reps[int(f_cid)] = int(best_idx)
    hierarchical_index.F_reps_ = f_reps

    query_manager = QueryManager(milvus, inverted_index, hierarchical_index)

    # Step 5: Query similar MSUs
    # F[0] is list of lists, I[0] and T[0] may be list or array - ensure list format
    I_input = I[0].tolist() if hasattr(I[0], 'tolist') else I[0]
    T_input = T[0].tolist() if hasattr(T[0], 'tolist') else T[0]
    results = query_manager.find_similar_by_features(F[0], I_input, T_input, top_k=args.top_k)

    print(f"\n{'='*60}")
    print(f"Query: video clip as single MSU, top-{args.top_k} similar segments")
    print(f"{'='*60}")

    from storage.sqlite_manager import SQLiteManager
    sqlite_path = os.path.join(args.db, "video_db.sqlite")
    sqlite_mgr = SQLiteManager(db_path=sqlite_path) if os.path.exists(sqlite_path) else None

    for i, res in enumerate(results, 1):
        res_id = res['msu_id']
        dist = res['f_distance']
        vsu_info = sqlite_mgr.get_vsu_by_id(res_id) if sqlite_mgr else None
        clip = os.path.basename(vsu_info['clip_path']) if vsu_info and vsu_info['clip_path'] else 'N/A'
        frame = f"{vsu_info['frame_start']}-{vsu_info['frame_end']}" if vsu_info and vsu_info['frame_start'] is not None else 'N/A'
        print(f"\n  Rank {i}: MSU {res_id}, F distance={dist:.4f}")
        print(f"    Clip: {clip}")
        print(f"    Frames: {frame}")

    if sqlite_mgr:
        sqlite_mgr.close()
    milvus.close()

    return 0


if __name__ == '__main__':
    import numpy as np
    sys.exit(main())