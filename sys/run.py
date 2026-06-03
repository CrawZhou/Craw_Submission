import argparse
import sys
import os
import numpy as np

# Only add parent_dir, not yolov13_main, to avoid ultralytics conflict
parent_dir = r"D:\XW2\yolov13"
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def main():
    parser = argparse.ArgumentParser(description="MSU Video MOT and Similar Segment Retrieval")
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--frame-info', type=str, help='frame_info JSON file path')
    parser.add_argument('--db', type=str, default='./msu_db', help='Database path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for intermediate JSON files (default: beside video)')
    parser.add_argument('--msu-id', type=int, help='Query MSU ID')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--model-path', type=str, help='YOLO model path')
    parser.add_argument('--roi', type=str, help='ROI region x1,y1,x2,y2')
    parser.add_argument('--full', action='store_true', help='Run full pipeline (preprocessing)')
    parser.add_argument('--query', action='store_true', help='Query mode (requires --msu-id or --element or --label)')
    parser.add_argument('--query-video', type=str, default=None,
                        help='Query mode: input video treated as single MSU, query existing index')
    parser.add_argument('--element-type', type=str, choices=['F', 'I'], help='Element type for search')
    parser.add_argument('--element-value', type=float, help='Element value for search')
    parser.add_argument('--label-type', type=str, choices=['T', 'I', 'F'], help='Label type for search')
    parser.add_argument('--label-value', type=int, help='Label value for search')
    args = parser.parse_args()

    # --- PREPROCESSING MODE ---
    if args.full:
        if not args.video and not args.frame_info:
            print("Error: --video or --frame-info required for --full")
            return 1

        from main import System
        system = System()

        if args.model_path:
            system.config.path.model_path = args.model_path

        system.config.storage.base_dir = args.db
        system.config._init_storage_paths()

        # Output dir for intermediate JSONs (beside video or frame_info, or custom)
        if args.output_dir:
            output_dir = args.output_dir
        elif args.frame_info:
            output_dir = os.path.dirname(os.path.abspath(args.frame_info)) or '.'
        else:
            output_dir = os.path.dirname(os.path.abspath(args.video)) or '.'

        if args.frame_info:
            system.load_frame_info(args.frame_info)
        else:
            roi = tuple(map(int, args.roi.split(','))) if args.roi else None
            system.run_from_video(args.video, roi=roi)

        query_manager = system.run_full_pipeline(args.db, output_dir=output_dir)
        print(system.summary())
        system.close()
        return 0

    # --- QUERY MODE ---
    if args.query or args.msu_id is not None or args.element_type is not None or args.label_type is not None or args.query_video is not None:
        # Query from existing database (no preprocessing)
        from storage.inverted_index import InvertedIndex
        from storage.milvus_storage import MilvusStorage
        from query.query_manager import QueryManager
        from clustering.hierarchical_index import HierarchicalClusterIndex
        from similarity.CTMD import compute_CTMD_fast
        from similarity.I_distance import compute_I_distance_fast

        db_path = args.db
        milvus_path = os.path.join(db_path, "msu_milvus_db")
        inverted_path = db_path

        # Check if database exists
        if not os.path.exists(milvus_path + ".db"):
            print(f"Error: Database not found at {milvus_path}. Run with --full first.")
            return 1

        # Load storage
        milvus = MilvusStorage(db_path=milvus_path)
        inverted_index = InvertedIndex(db_path=inverted_path)

        # Load all data from Milvus to build hierarchical index
        all_data = milvus.get_all(limit=10000)
        if not all_data:
            print("Error: No MSU data found in database")
            milvus.close()
            return 1

        # Build F/I/T lists and labels from loaded data
        # Keep F as list of lists (not numpy array) since rows have varying lengths
        def _safe_matrix(data):
            """Convert to 2D numpy array, handling empty matrices serialized as []."""
            arr = np.array(data)
            if arr.ndim == 1 and len(arr) == 0:
                return arr.reshape(0, 0)
            return arr

        F_list = [m['F_matrix'] for m in all_data]
        I_list = [_safe_matrix(m['I_matrix']) for m in all_data]
        T_list = [_safe_matrix(m['T_matrix']) for m in all_data]
        T_labels = np.array([m['T_label'] for m in all_data])
        I_labels = np.array([m['I_label'] for m in all_data])
        F_labels = np.array([m['F_label'] for m in all_data])

        # Build hierarchical index
        hierarchical_index = HierarchicalClusterIndex()
        hierarchical_index.T_labels_ = T_labels
        hierarchical_index.I_labels_ = I_labels
        hierarchical_index.F_labels_ = F_labels
        # Get T_max_rows from actual T data
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

        # Compute I distance matrix and representatives
        from similarity.I_distance import compute_I_distance_fast
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

        # Compute F distance matrix and representatives
        from similarity.PM_TSMD import compute_PM_TSMD
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

        # Execute query
        if args.element_type is not None and args.element_value is not None:
            msu_ids = query_manager.search_by_element(args.element_type, args.element_value)
            print(f"\nFound {len(msu_ids)} MSUs with {args.element_type}={args.element_value}:")
            for msu_id in msu_ids[:20]:
                print(f"  MSU {msu_id}")
            if len(msu_ids) > 20:
                print(f"  ... and {len(msu_ids) - 20} more")
        elif args.label_type is not None and args.label_value is not None:
            results = query_manager.get_by_label(args.label_type, args.label_value)
            print(f"\nFound {len(results)} MSUs with {args.label_type}_label={args.label_value}:")
            for res in results[:20]:
                print(f"  MSU {res['id']}")
            if len(results) > 20:
                print(f"  ... and {len(results) - 20} more")
        elif args.msu_id is not None:
            results = query_manager.find_similar_by_id(args.msu_id, args.top_k)
            print(f"\nFound {len(results)} MSUs similar to MSU {args.msu_id}:")
            for i, res in enumerate(results):
                print(f"  {i+1}. MSU {res['msu_id']}, F distance: {res['f_distance']:.4f}")
        elif args.query_video is not None:
            # Query by input video as single MSU (no segmentation)
            from main import System

            # Build system and run tracking only (skip MSU segmentation)
            system = System()
            if args.model_path:
                system.config.path.model_path = args.model_path

            video_path = args.query_video
            roi = tuple(map(int, args.roi.split(','))) if args.roi else None
            system.run_from_video(video_path, roi=roi)

            # Treat entire video as single MSU: range = (0, last_frame)
            from collections import defaultdict
            frame_info = system._frame_info
            all_frames = sorted(int(k) for k in frame_info.keys())
            if not all_frames:
                print("Error: No frames found in video")
                return 1
            last_frame = max(all_frames)

            # Build duration map for all objects
            obj_frames = defaultdict(list)
            for frame_id, detections in frame_info.items():
                frame_num = int(frame_id)
                for det in detections:
                    obj_frames[det['id']].append(frame_num)
            for oid in obj_frames:
                obj_frames[oid].sort()

            msu_ranges = [(0, last_frame)]
            msu_durations = [{
                oid: (min(frames), max(frames))
                for oid, frames in obj_frames.items()
            }]

            # Build feature matrices
            from features.feature_builder import FeatureBuilder
            fb = FeatureBuilder(system.config)
            fb.set_msu_splits(msu_ranges, msu_durations)
            F_list, I_list, T_list = fb.build(frame_info, skip_splits=True)

            if not F_list:
                print("Error: Failed to build features for query video")
                return 1

            F = F_list[0]
            I = I_list[0].tolist() if hasattr(I_list[0], 'tolist') else I_list[0]
            T = T_list[0].tolist() if hasattr(T_list[0], 'tolist') else T_list[0]

            # Query using hierarchical filtering
            results = query_manager.find_similar_by_features(F, I, T, top_k=args.top_k)
            print(f"\nQuery video treated as single MSU (frames 0-{last_frame}):")
            print(f"Found {len(results)} similar MSUs from index:")
            for i, res in enumerate(results):
                print(f"  {i+1}. MSU {res['msu_id']}, F distance: {res['f_distance']:.4f}")

            system.close()
        else:
            print("Error: Query mode requires --msu-id, --element-type, --label-type, or --query-video")
            return 1

        milvus.close()
        return 0

    # --- NO MODE SPECIFIED ---
    print("Error: Specify --full for preprocessing or --query for querying")
    print("Examples:")
    print("  Preprocessing: python -m run --video video.mp4 --full")
    print("  Query video:   python -m run --db ./msu_db --query-video video.mp4")
    return 1


if __name__ == '__main__':
    sys.exit(main())