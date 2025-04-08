import numpy as np
from collections import Counter
import queue

class ClassificationTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        # _fit()_ builds and sets this tree from scratch
        self.tree = None
        # Maximum depth of the tree
        self.max_depth = max_depth
        # Minimum numbers of data required to split further
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        root = {
            "X": X,
            "y": y,
            "depth": 0,
            "is_leaf": False,
            "leaf_value": None,
            "feature_index": None,
            "threshold": None,
            "left": None,
            "right": None
        }
        nodes_queue = queue.Queue()
        nodes_queue.put(root)

        while not nodes_queue.empty():
            node = nodes_queue.get()

            X_nodes = node["X"]
            y_nodes = node["y"]
            depth = node["depth"]

            if self._should_stop_growing(depth, y_nodes):
                node["is_leaf"] = True
                # Note: We use the majority value when the node is impure
                node["leaf_value"] = Counter(y_nodes).most_common(1)[0][0]
                self._cleanup_node(node)
                continue

            best_feature_index, best_threshold, best_left_mask, best_right_mask = self._best_split(X_nodes, y_nodes)
            if best_feature_index is None:
                node["is_leaf"] = True
                node["leaf_value"] = Counter(y_nodes).most_common(1)[0][0]
                self._cleanup_node(node)
                continue
            self._cleanup_node(node)

            # Set split information
            node["feature_index"] = best_feature_index
            node["threshold"] = best_threshold

            # Prepare leaf nodes
            left_child = {
                "X": X_nodes[best_left_mask],
                "y": y_nodes[best_left_mask],
                "depth": depth + 1,
                "is_leaf": False,
                "leaf_value": None,
                "feature_index": None,
                "threshold": None,
                "left": None,
                "right": None
            }
            right_child = {
                "X": X_nodes[best_right_mask],
                "y": y_nodes[best_right_mask],
                "depth": depth + 1,
                "is_leaf": False,
                "leaf_value": None,
                "feature_index": None,
                "threshold": None,
                "left": None,
                "right": None
            }
            node["left"] = left_child
            node["right"] = right_child

            nodes_queue.put(left_child)
            nodes_queue.put(right_child)

        return root

    def _should_stop_growing(self, depth, y_nodes):
        if self.max_depth is not None and depth >= self.max_depth: return True 
        if len(y_nodes) < self.min_samples_split: return True
        if self._are_nodes_pure(y_nodes): return True

        return False

    # We can stop growing the tree when the node is pure - All labels are either 0 or 1
    def _are_nodes_pure(self, y_nodes):
        return len(np.unique(y_nodes)) == 1

    # There is no need to keep X and y after feature has determined, reducing the memory by cleaning it up
    def _cleanup_node(self, node):
        node["X"] = None
        node["y"] = None

    # Try all possible thresholds across all rows and features to determine the best pair of feature and threshold.
    # "best" means the lowest impurity and this statement should go to the very root of the tree.
    def _best_split(self, X, y):
        best_gini = float("inf")
        best_feature_index = None
        best_threshold = None
        best_left_mask, best_right_mask = None, None

        _, n_features = X.shape
        for feature_index in range(n_features):
            feature_samples = X[:, feature_index]
            # get the sorted unique values from all the rows for the current feature column.
            thresholds = np.unique(feature_samples)
            for threshold in thresholds:
                # Create boolean masks for the splits.
                # Note: In decision tree, left nodes always mean 'True' and right nodes mean 'False'.
                left_mask, right_mask = [], []
                for feature_sample in feature_samples:
                    if feature_sample <= threshold: # This is the statement we are checking
                        left_mask.append(True)
                        right_mask.append(False)
                    else:
                        left_mask.append(False)
                        right_mask.append(True)

                # Skip this threshold if either child node will become empty.
                # Since False is treated as 0, using its total sum here. 
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Get the target values for each split.
                # This is using NumPy's boolean indexing which returns items from 'y' that only masked index is 'True'.
                left_y = y[left_mask]
                right_y = y[right_mask]

                gini = self._gini_impurity(left_y, right_y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask

        return best_feature_index, best_threshold, best_left_mask, best_right_mask

    # Calculates Gini Impurity.
    def _gini_impurity(self, left_y, right_y):
        def gini(values):
            total = len(values)
            if total == 0:
                return 0
            _, occurrences = np.unique(values, return_counts=True)
            probs = occurrences / total
            return 1 - np.sum(probs**2)

        n_left, n_right = len(left_y), len(right_y)
        total_samples = n_left + n_right
        
        # Weighted average of Gini impurity for both splits
        weighted_gini = (n_left / total_samples) * gini(left_y) + (n_right / total_samples) * gini(right_y)
        return weighted_gini

    def predict(self, X):
        return np.array([self._predict_single(sample, self.tree) for sample in X])

    def _predict_single(self, sample, node):
        if node["is_leaf"] is False:
            if sample[node["feature_index"]] <= node["threshold"]:
                return self._predict_single(sample, node["left"])
            else:
                return self._predict_single(sample, node["right"])
        else:
            return node["leaf_value"]

    def __str__(self):
        """Generate a string representation of the tree using iterative approach."""
        if self.tree is None:
            return "Tree not fitted yet"
        
        result = []
        stack = [(self.tree, "", False)]
        while stack:
            node, indent, is_right_child = stack.pop()
            
            if node["is_leaf"]:
                result.append(f"{indent}Predict({'True' if is_right_child else 'False'}): {node['leaf_value']}")
            else:
                feature = node["feature_index"]
                threshold = node["threshold"]
                
                if is_right_child:
                    result.append(f"{indent}Feature {feature} > {threshold:.4f}")
                else:
                    result.append(f"{indent}Feature {feature} <= {threshold:.4f}")

                if node["right"] is not None:
                    stack.append((node["right"], indent + "  ", True))
                if node["left"] is not None:
                    stack.append((node["left"], indent + "  ", False))
        
        return "\n".join(result)
