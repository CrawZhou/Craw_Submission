import pickle
import numpy as np
import glob
import os


class Node:
    def __init__(self, data=None, svs_id=None):
        self.data = data
        self.svs_id = svs_id
        self.left = None
        self.right = None
        self.parent = None
        self.node_id = None
        self.is_leaf = (data is not None)
        self.centroid = None


class PERCH:
    def __init__(self):
        self.root = None
        self.node_counter = 0

    def _compute_centroid(self, data):
        if len(data) == 0:
            return None
        return np.mean(data, axis=0)

    def _compute_distance(self, data1, data2):
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

    def insert(self, svs, svs_id):
        new_node = Node(data=svs, svs_id=svs_id)
        new_node.centroid = self._compute_centroid(svs)
        if self.root is None:
            self.root = new_node
            print(f"  Created root node: SVS {svs_id}")
        else:
            self._insert_node(self.root, new_node)
            print(f"  Inserted node: SVS {svs_id}")

    def _insert_node(self, root, new_node):
        if root.is_leaf:
            internal_node = Node(svs_id=None)
            internal_node.node_id = self.node_counter
            self.node_counter += 1
            internal_node.is_leaf = False
            internal_node.centroid = (root.centroid + new_node.centroid) / 2
            if np.random.rand() < 0.5:
                internal_node.left = root
                internal_node.right = new_node
            else:
                internal_node.left = new_node
                internal_node.right = root
            internal_node.left.parent = internal_node
            internal_node.right.parent = internal_node
            if root == self.root:
                self.root = internal_node
                internal_node.parent = None
            else:
                if root.parent.left == root:
                    root.parent.left = internal_node
                else:
                    root.parent.right = internal_node
                internal_node.parent = root.parent
        else:
            if root.left is None:
                root.left = new_node
                new_node.parent = root
            elif root.right is None:
                root.right = new_node
                new_node.parent = root
            else:
                dist_to_left = self._compute_distance(new_node.data, root.left.data)
                dist_to_right = self._compute_distance(new_node.data, root.right.data)
                if dist_to_left <= dist_to_right:
                    self._insert_node(root.left, new_node)
                else:
                    self._insert_node(root.right, new_node)

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        if node is not None:
            indent = "  " * level
            if node.is_leaf:
                print(f"{indent}Leaf: SVS {node.svs_id}")
            else:
                print(f"{indent}Internal Node {node.node_id}")
                if node.left:
                    print(f"{indent}├─ Left:")
                    self.print_tree(node.left, level + 1)
                if node.right:
                    print(f"{indent}└─ Right:")
                    self.print_tree(node.right, level + 1)


def load_svs_features(svs_dir):
    svs_files = sorted(glob.glob(f"{svs_dir}/*.pkl"))
    svs_data = []
    for i, svs_file in enumerate(svs_files):
        try:
            with open(svs_file, 'rb') as f:
                features = pickle.load(f)
                svs_data.append((features, i, svs_file))
        except Exception as e:
            print(f"Error loading {svs_file}: {e}")
    return svs_data


if __name__ == "__main__":
    os.makedirs('../runs', exist_ok=True)
    print("Loading SVS data...")
    svs_data = load_svs_features('../data/svs')
    if not svs_data:
        print("No SVS data loaded")
        exit()
    print(f"Successfully loaded {len(svs_data)} SVS files")
    print("Building PERCH tree...")
    perch = PERCH()
    for features, svs_id, svs_file in svs_data:
        print(f"Processing SVS {svs_id}: {os.path.basename(svs_file)}")
        perch.insert(features, svs_id)
    print("\nBuilt tree structure:")
    perch.print_tree()
    with open('../runs/intra_index.pkl', 'wb') as f:
        pickle.dump(perch, f)
    print("\nIndex saved to ../runs/intra_index.pkl")
    print("Tree structure construction complete!")
