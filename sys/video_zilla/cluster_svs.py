import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import glob
import os
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA  # Import PCA early to avoid import overhead in timing


def omd(a, b):
    if len(a) == 0 or len(b) == 0:
        return float('inf')

    if np.array_equal(a, b):
        return 0.0

    max_features = 30
    if len(a) > max_features:
        indices = np.linspace(0, len(a)-1, max_features, dtype=int)
        a = a[indices]

    if len(b) > max_features:
        indices = np.linspace(0, len(b)-1, max_features, dtype=int)
        b = b[indices]

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    max_dim = 256
    if a.shape[1] > max_dim:
        a = a[:, :max_dim]
    if b.shape[1] > max_dim:
        b = b[:, :max_dim]

    min_features = min(a.shape[0], b.shape[0])
    if min_features == 0:
        return float('inf')

    if a.shape[0] > min_features:
        a = a[:min_features]
    if b.shape[0] > min_features:
        b = b[:min_features]

    try:
        dist_mat = np.zeros((len(a), len(b)), dtype=np.float32)
        for i in range(len(a)):
            for j in range(len(b)):
                dist_mat[i, j] = np.linalg.norm(a[i] - b[j])

        w1 = np.ones(len(a), dtype=np.float32) / len(a)
        w2 = np.ones(len(b), dtype=np.float32) / len(b)

        from pyemd import emd
        distance = emd(w1, w2, dist_mat)
        return float(distance)
    except Exception as e:
        try:
            avg_a = np.mean(a, axis=0)
            avg_b = np.mean(b, axis=0)
            distance = np.linalg.norm(avg_a - avg_b)
            return float(distance)
        except:
            return float('inf')


def find_optimal_clusters(distance_matrix, max_clusters=10):
    """Automatically determine optimal clusters using silhouette score maximization"""
    distance_matrix = np.nan_to_num(distance_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    max_clusters = min(max_clusters, len(distance_matrix)-1)
    if max_clusters < 4:
        return 4

    silhouette_scores = []
    k_range = range(4, max_clusters + 1)

    print("Calculating silhouette scores for different cluster counts...")

    for k in k_range:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=k,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)

            if len(np.unique(labels)) > 1:
                silhouette_avg = silhouette_score(distance_matrix, labels, metric='precomputed')
                silhouette_scores.append(silhouette_avg)
                print(f"  k={k}: silhouette score={silhouette_avg:.4f}")
            else:
                silhouette_scores.append(-1)
                print(f"  k={k}: silhouette score=-1.0000 (invalid)")
        except Exception as e:
            print(f"  k={k}: calculation error - {e}")
            silhouette_scores.append(-1)

    if silhouette_scores:
        optimal_k = k_range[np.argmax(silhouette_scores)]
        max_silhouette = max(silhouette_scores)
        print(f"Optimal clusters: {optimal_k} (silhouette: {max_silhouette:.4f})")
        return optimal_k
    else:
        print("Cannot determine optimal clusters, using default value 4")
        return 4


def compute_omd_distance_matrix(svs_features_list):
    """Compute OMD distance matrix between all SVS"""
    n = len(svs_features_list)
    distance_matrix = np.zeros((n, n))

    print("Computing OMD distance matrix...")
    for i in range(n):
        for j in range(i+1, n):
            distance = omd(svs_features_list[i], svs_features_list[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
        if (i+1) % 10 == 0 or i+1 == n:
            print(f"  Progress: {i+1}/{n}")

    return distance_matrix


def cluster_svs(svs_features_list, num_clusters=None):
    """Execute core clustering logic (pre-loaded features to avoid file I/O in timing)"""
    # Core clustering timing start
    cluster_start_time = time.perf_counter()

    # 1. Compute OMD distance matrix (core step 1)
    distance_matrix = compute_omd_distance_matrix(svs_features_list)

    # 2. Determine cluster count (core step 2)
    if num_clusters is None:
        actual_num_clusters = find_optimal_clusters(distance_matrix)
    else:
        actual_num_clusters = min(num_clusters, len(svs_features_list))

    print(f"Executing clustering, cluster count: {actual_num_clusters}")

    # 3. Execute hierarchical clustering (core step 3)
    try:
        clustering = AgglomerativeClustering(
            n_clusters=actual_num_clusters,
            metric='precomputed',
            linkage='average'
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=actual_num_clusters,
            affinity='precomputed',
            linkage='average'
        )
    labels = clustering.fit_predict(distance_matrix)

    cluster_end_time = time.perf_counter()
    core_cluster_duration = cluster_end_time - cluster_start_time

    print(f"\n[Core clustering total time]: {core_cluster_duration:.2f} seconds")

    return labels, distance_matrix, actual_num_clusters, core_cluster_duration


def visualize_clusters(svs_features_list, labels, save_path):
    """
    Optimized cluster visualization function:
    1. Use pre-loaded feature list to avoid repeated file reads
    2. Check save path directory, create if not exists
    3. Enhanced exception handling, output detailed error info
    """
    try:
        # 1. Check if save directory exists, create if not
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created image save directory: {save_dir}")

        # 2. Compute average feature per SVS (for PCA dimensionality reduction)
        all_avg_features = []
        for features in svs_features_list:
            avg_feature = np.mean(features, axis=0)
            all_avg_features.append(avg_feature)
        all_avg_features = np.array(all_avg_features)

        # 3. PCA reduce to 2D (for visualization)
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(all_avg_features)

        # 4. Compute cluster centers (2D space)
        unique_labels = np.unique(labels)
        cluster_centers_2d = []
        for label in unique_labels:
            cluster_2d_features = features_2d[labels == label]
            cluster_center_2d = np.mean(cluster_2d_features, axis=0)
            cluster_centers_2d.append(cluster_center_2d)
        cluster_centers_2d = np.array(cluster_centers_2d)

        # 5. Draw visualization
        plt.figure(figsize=(12, 8))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
                  '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']

        # Plot SVS points for each cluster
        for idx, label in enumerate(unique_labels):
            cluster_points = features_2d[labels == label]
            plt.scatter(
                cluster_points[:, 0], cluster_points[:, 1],
                c=colors[idx % len(colors)],
                label=f'Cluster {label} (n={len(cluster_points)})',
                alpha=0.8,
                s=80
            )

        # Plot cluster centers (black X marks, prominent display)
        plt.scatter(
            cluster_centers_2d[:, 0], cluster_centers_2d[:, 1],
            c='black', marker='x', s=300, linewidths=4,
            label='Cluster Centers', zorder=5
        )

        # Legend and format optimization
        plt.title('SVS Clustering Result Visualization (PCA 2D)', fontsize=16, fontweight='bold')
        plt.xlabel('PCA Component 1', fontsize=12)
        plt.ylabel('PCA Component 2', fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        # Save image (high resolution)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nCluster visualization saved to: {save_path}")

    except Exception as e:
        print(f"\nCluster visualization failed:")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error details: {str(e)}")


if __name__ == "__main__":
    # ---------------------- 1. Load SVS feature files ----------------------
    file_load_start = time.perf_counter()

    svs_dir = '/home/nanchang/ZZQ/yolov13/main/yolov13-main/Video-zilla/warsaw/origin'
    svs_files = sorted(glob.glob(f"{svs_dir}/*.pkl"))
    svs_features_list = []
    valid_svs_files = []

    if not svs_files:
        exit()

    for svs_file in svs_files:
        try:
            with open(svs_file, 'rb') as f:
                features = pickle.load(f)
                if len(features) > 0 and features.ndim == 2:
                    svs_features_list.append(features)
                    valid_svs_files.append(svs_file)
                else:
                    print(f"Skipping invalid file {os.path.basename(svs_file)}: feature empty or format error")
        except Exception as e:
            print(f"Reading file {os.path.basename(svs_file)}: {str(e)}")

    if len(svs_features_list) == 0:
        print("No valid SVS feature data loaded, exiting")
        exit()

    file_load_end = time.perf_counter()
    print(f"\nFile loading complete:")
    print(f"  - Valid SVS count: {len(svs_features_list)}")
    print(f"  - Loading time: {file_load_end - file_load_start:.2f} seconds")

    # ---------------------- 2. Execute core clustering ----------------------
    labels, distance_matrix, actual_num_clusters, core_cluster_time = cluster_svs(svs_features_list)

    # ---------------------- 3. Select cluster representatives & save clustering results ----------------------
    result_process_start = time.perf_counter()

    # Select cluster "center representative" (SVS with minimum average distance)
    cluster_representatives = []
    for i in range(actual_num_clusters):
        cluster_indices = np.where(labels == i)[0]
        min_avg_distance = float('inf')
        representative_idx = cluster_indices[0]

        for idx in cluster_indices:
            avg_distance = np.mean(distance_matrix[idx][cluster_indices])
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                representative_idx = idx

        representative_file = valid_svs_files[representative_idx]
        cluster_representatives.append(representative_file)
        print(f"\nCluster {i}:")
        print(f"  - Representative SVS: {os.path.basename(representative_file)}")
        print(f"  - SVS count in cluster: {len(cluster_indices)}")

    # Save clustering result (includes visualization info)
    cluster_result_path = '/home/nanchang/ZZQ/yolov13/main/yolov13-main/Video-zilla/warsaw/origin/cluster_result.pkl'
    cluster_info = {
        'labels': labels,
        'svs_files': valid_svs_files,
        'representatives': cluster_representatives,
        'core_cluster_time_seconds': core_cluster_time,
        'optimal_clusters': actual_num_clusters
    }
    with open(cluster_result_path, 'wb') as f:
        pickle.dump(cluster_info, f)
    print(f"\nClustering result saved to: {cluster_result_path}")

    result_process_end = time.perf_counter()

    cluster_img_path = '/home/nanchang/ZZQ/yolov13/main/yolov13-main/Video-zilla/warsaw/origin/cluster_visualization.png'
    visualize_clusters(svs_features_list=svs_features_list, labels=labels, save_path=cluster_img_path)

    # ---------------------- 5. Output final statistics ----------------------
    print(f"\nFinal statistics:")
    print(f"  - Core clustering time: {core_cluster_time:.2f} seconds (distance matrix + clustering)")
    print(f"  - Result processing time: {result_process_end - result_process_start:.2f} seconds (representative + saving)")
    print(f"  - Total time: {time.perf_counter() - file_load_start:.2f} seconds")
    print(f"  - Visualization path: {cluster_img_path}")
